from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minileet.features import (
    FeatureSpec,
    extract_features,
    fit_feature_spec,
    load_rows,
    row_task_name,
    row_visible_all_pass,
    target_hidden_pass_rate,
)


@dataclass(frozen=True)
class SelectionMetrics:
    tasks: int
    selected: int
    solved: int
    visible_overfit: int
    mean_hidden_pass_rate: float
    mean_candidate_index: float


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.rerank")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-visible-fail", action="store_true")
    parser.add_argument("--model-out", default="")
    args = parser.parse_args()

    train_rows = load_rows(Path(args.train))
    eval_rows = load_rows(Path(args.eval))
    result = train_and_score(
        train_rows,
        eval_rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if args.model_out:
        _save_model(Path(args.model_out), result)

    raw = evaluate_first_visible_pass(eval_rows)
    learned = evaluate_predicted_selection(
        eval_rows,
        result["eval_predictions"],
        visible_gated=not args.allow_visible_fail,
    )
    oracle = evaluate_oracle_best(eval_rows, visible_gated=not args.allow_visible_fail)

    print(f"device={result['device']}")
    print(f"features={result['feature_count']}")
    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    print(f"train_mse={result['train_mse']:.5f} eval_mse={result['eval_mse']:.5f}")
    _print_metrics("first_visible", raw)
    _print_metrics("learned", learned)
    _print_metrics("oracle_visible_gated", oracle)


def train_and_score(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    epochs: int,
    lr: float,
    hidden: int,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = fit_feature_spec(train_rows, max_tokens=max_tokens)
    x_train_raw = _tensor([extract_features(row, spec) for row in train_rows], device)
    y_train = _tensor([[target_hidden_pass_rate(row)] for row in train_rows], device)
    x_eval_raw = _tensor([extract_features(row, spec) for row in eval_rows], device)
    y_eval = _tensor([[target_hidden_pass_rate(row)] for row in eval_rows], device)

    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std
    x_eval = (x_eval_raw - mean) / std

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 1),
        nn.Sigmoid(),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        eval_pred = model(x_eval)
        train_mse = float(loss_fn(train_pred, y_train).cpu())
        eval_mse = float(loss_fn(eval_pred, y_eval).cpu())

    return {
        "device": str(device),
        "spec": spec,
        "model": model,
        "mean": mean.detach().cpu().tolist()[0],
        "std": std.detach().cpu().tolist()[0],
        "feature_count": x_train.shape[1],
        "train_mse": train_mse,
        "eval_mse": eval_mse,
        "eval_predictions": [float(value) for value in eval_pred.detach().cpu().flatten().tolist()],
    }


def evaluate_first_visible_pass(rows: list[dict[str, Any]]) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    for _, group in _group_by_task(rows).items():
        chosen = next((row for row in group if row_visible_all_pass(row)), None)
        if chosen is not None:
            selected.append(chosen)
    return _metrics(rows, selected)


def evaluate_predicted_selection(
    rows: list[dict[str, Any]],
    predictions: list[float],
    visible_gated: bool,
) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    grouped = _group_rows_and_scores(rows, predictions)
    for _, group in grouped.items():
        candidates = [(row, score) for row, score in group if not visible_gated or row_visible_all_pass(row)]
        if candidates:
            selected.append(max(candidates, key=lambda item: item[1])[0])
    return _metrics(rows, selected)


def evaluate_oracle_best(rows: list[dict[str, Any]], visible_gated: bool) -> SelectionMetrics:
    selected: list[dict[str, Any]] = []
    for _, group in _group_by_task(rows).items():
        candidates = [row for row in group if not visible_gated or row_visible_all_pass(row)]
        if candidates:
            selected.append(max(candidates, key=target_hidden_pass_rate))
    return _metrics(rows, selected)


def _metrics(all_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> SelectionMetrics:
    task_count = len(_group_by_task(all_rows))
    solved = sum(int(row["labels"]["solved"]) for row in selected)
    overfit = sum(int(row_visible_all_pass(row) and target_hidden_pass_rate(row) < 1.0) for row in selected)
    hidden_sum = sum(target_hidden_pass_rate(row) for row in selected)
    candidate_sum = sum(float(row["candidate_index"]) for row in selected)
    return SelectionMetrics(
        tasks=task_count,
        selected=len(selected),
        solved=solved,
        visible_overfit=overfit,
        mean_hidden_pass_rate=hidden_sum / len(selected) if selected else 0.0,
        mean_candidate_index=candidate_sum / len(selected) if selected else 0.0,
    )


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_task_name(row)].append(row)
    return groups


def _group_rows_and_scores(
    rows: list[dict[str, Any]],
    predictions: list[float],
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    groups: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, predictions):
        groups[row_task_name(row)].append((row, score))
    return groups


def _tensor(values: list[list[float]], device: object):
    import torch

    return torch.tensor(values, dtype=torch.float32, device=device)


def _print_metrics(name: str, metrics: SelectionMetrics) -> None:
    print(
        f"{name}: "
        f"selected={metrics.selected}/{metrics.tasks} "
        f"solved={metrics.solved}/{metrics.tasks} "
        f"mean_hidden={metrics.mean_hidden_pass_rate:.3f} "
        f"visible_overfit={metrics.visible_overfit} "
        f"mean_candidate={metrics.mean_candidate_index:.1f}"
    )


def _save_model(path: Path, result: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "feature_spec": result["spec"].to_json(),
            "mean": result["mean"],
            "std": result["std"],
        },
        path,
    )


if __name__ == "__main__":
    main()

