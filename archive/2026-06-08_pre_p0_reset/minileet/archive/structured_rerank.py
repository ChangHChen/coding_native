from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minileet.features import FeatureSpec, extract_features, load_rows, target_hidden_pass_rate
from minileet.rerank import (
    evaluate_first_visible_pass,
    evaluate_oracle_best,
    evaluate_predicted_selection,
)
from minileet.sequence import (
    TokenSpec,
    encode_tokens,
    fit_structured_token_spec,
    row_to_structured_tokens,
)


@dataclass(frozen=True)
class StructuredConfig:
    epochs: int
    batch_size: int
    lr: float
    max_vocab: int
    task_len: int
    program_len: int
    test_len: int
    max_tests: int
    d_model: int
    heads: int
    encoder_layers: int
    fusion_layers: int
    ff_mult: int
    dropout: float
    numeric_hidden: int
    registers: int
    bce_weight: float
    rank_weight: float
    eval_every: int
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.structured_rerank")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-vocab", type=int, default=4096)
    parser.add_argument("--task-len", type=int, default=64)
    parser.add_argument("--program-len", type=int, default=256)
    parser.add_argument("--test-len", type=int, default=160)
    parser.add_argument("--max-tests", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--numeric-hidden", type=int, default=64)
    parser.add_argument("--registers", type=int, default=2)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-visible-fail", action="store_true")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    config = StructuredConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_vocab=args.max_vocab,
        task_len=args.task_len,
        program_len=args.program_len,
        test_len=args.test_len,
        max_tests=args.max_tests,
        d_model=args.d_model,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        fusion_layers=args.fusion_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        numeric_hidden=args.numeric_hidden,
        registers=args.registers,
        bce_weight=args.bce_weight,
        rank_weight=args.rank_weight,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    train_rows = load_rows(Path(args.train))
    eval_rows = load_rows(Path(args.eval))
    run_dir = Path(args.run_dir) if args.run_dir else None
    result = train_and_score(
        train_rows,
        eval_rows,
        config,
        run_dir=run_dir,
        visible_gated=not args.allow_visible_fail,
        train_path=args.train,
        eval_path=args.eval,
    )

    raw = evaluate_first_visible_pass(eval_rows)
    learned = evaluate_predicted_selection(
        eval_rows,
        result["eval_predictions"],
        visible_gated=not args.allow_visible_fail,
    )
    oracle = evaluate_oracle_best(eval_rows, visible_gated=not args.allow_visible_fail)

    print(f"device={result['device']}")
    print(f"params={result['params']}")
    print(f"vocab={len(result['token_spec'].vocab)}")
    print(
        "shape="
        f"d:{config.d_model} heads:{config.heads} shared_layers:{config.encoder_layers} "
        f"fusion_layers:{config.fusion_layers} registers:{config.registers}"
    )
    print(
        "lengths="
        f"task:{config.task_len} program:{config.program_len} "
        f"test:{config.test_len} max_tests:{config.max_tests}"
    )
    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    print(f"train_mse={result['train_mse']:.5f} eval_mse={result['eval_mse']:.5f}")
    for index, loss in enumerate(result["epoch_losses"], start=1):
        print(f"epoch={index} train_loss={loss:.5f}")
    _print_metrics("first_visible", raw)
    _print_metrics("structured_shared", learned)
    _print_metrics("oracle_visible_gated", oracle)


def train_and_score(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    config: StructuredConfig,
    run_dir: Path | None = None,
    visible_gated: bool = True,
    train_path: str = "",
    eval_path: str = "",
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    token_spec = fit_structured_token_spec(train_rows, max_vocab=config.max_vocab)
    feature_spec = FeatureSpec(())

    train_tensors, train_task_ids = _encode_rows(train_rows, token_spec, feature_spec, config)
    eval_tensors, _ = _encode_rows(eval_rows, token_spec, feature_spec, config)
    dataset = TensorDataset(*train_tensors, train_task_ids)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)

    numeric_dim = train_tensors[-3].shape[1]
    model = SharedStructuredReranker(
        vocab_size=len(token_spec.vocab),
        numeric_dim=numeric_dim,
        config=config,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    epoch_losses: list[float] = []
    history: list[dict[str, Any]] = []
    started = time.time()
    if run_dir is not None:
        _init_run_dir(
            run_dir,
            config,
            train_path=train_path,
            eval_path=eval_path,
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            params=sum(parameter.numel() for parameter in model.parameters()),
            vocab=len(token_spec.vocab),
            device=str(device),
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            task_ids_for_loss = batch[-1].to(device, non_blocking=True)
            batch = [tensor.to(device, non_blocking=True) for tensor in batch[:-1]]
            hidden_targets = batch[-2]
            solved_targets = batch[-1]

            opt.zero_grad(set_to_none=True)
            hidden_pred, solved_pred, rank_score = model(*batch[:-2])
            loss = mse_loss(hidden_pred, hidden_targets)
            loss = loss + config.bce_weight * bce_loss(solved_pred, solved_targets)
            loss = loss + config.rank_weight * _pairwise_rank_loss(
                rank_score,
                hidden_targets,
                task_ids_for_loss,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.detach().cpu()) * hidden_targets.shape[0]
            total_items += hidden_targets.shape[0]
        train_loss = total_loss / total_items
        epoch_losses.append(train_loss)
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            record = _evaluate_epoch(
                model,
                eval_rows,
                eval_tensors,
                device,
                config,
                epoch=epoch,
                train_loss=train_loss,
                mse_loss=mse_loss,
                visible_gated=visible_gated,
                started=started,
            )
            history.append(record)
            if run_dir is not None:
                _append_jsonl(run_dir / "history.jsonl", record)
                _write_plot(run_dir / "loss.png", history)
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.5f} "
                f"eval_mse={record['eval_mse']:.5f} "
                f"solved={record['selected_solved']}/{record['tasks']} "
                f"mean_hidden={record['selected_mean_hidden']:.3f} "
                f"elapsed={record['elapsed_sec']:.1f}s"
            )

    model.eval()
    with torch.no_grad():
        train_pred = _predict(model, train_tensors, device, config.batch_size)
        eval_pred = _predict(model, eval_tensors, device, config.batch_size)
        train_mse = float(mse_loss(train_pred, train_tensors[-2].to(device)).detach().cpu())
        eval_mse = float(mse_loss(eval_pred, eval_tensors[-2].to(device)).detach().cpu())

    result = {
        "device": str(device),
        "token_spec": token_spec,
        "model": model,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "train_mse": train_mse,
        "eval_mse": eval_mse,
        "epoch_losses": epoch_losses,
        "history": history,
        "eval_predictions": [float(value) for value in eval_pred.detach().cpu().flatten().tolist()],
    }
    if run_dir is not None:
        _write_summary(run_dir, result, config, eval_rows, visible_gated)
    return result


def _make_bert_config(
    vocab_size: int,
    hidden_size: int,
    layers: int,
    heads: int,
    ff_mult: int,
    dropout: float,
    max_position_embeddings: int,
):
    from transformers import BertConfig

    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden_size * ff_mult,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
        pad_token_id=0,
    )


def _make_bert(config: object):
    from transformers import BertModel

    return BertModel(config, add_pooling_layer=False)


class SharedStructuredReranker:
    pass


def _make_shared_structured_reranker_class():
    import torch
    from torch import nn

    class _SharedStructuredReranker(nn.Module):
        def __init__(self, vocab_size: int, numeric_dim: int, config: StructuredConfig) -> None:
            super().__init__()
            self.config = config
            shared_len = max(config.task_len, config.program_len, config.test_len)
            self.shared_encoder = _make_bert(
                _make_bert_config(
                    vocab_size,
                    config.d_model,
                    config.encoder_layers,
                    config.heads,
                    config.ff_mult,
                    config.dropout,
                    shared_len,
                )
            )
            self.register_tokens = nn.Parameter(
                torch.randn(1, config.registers, config.d_model) * 0.02
            )
            self.role_embed = nn.Embedding(config.registers + 2 + config.max_tests, config.d_model)
            self.fusion_encoder = _make_bert(
                _make_bert_config(
                    1,
                    config.d_model,
                    config.fusion_layers,
                    config.heads,
                    config.ff_mult,
                    config.dropout,
                    config.registers + 2 + config.max_tests,
                )
            )
            self.numeric_mlp = nn.Sequential(
                nn.LayerNorm(numeric_dim),
                nn.Linear(numeric_dim, config.numeric_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.numeric_hidden, config.d_model),
                nn.GELU(),
            )
            self.trunk = nn.Sequential(
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.hidden_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())
            self.solved_head = nn.Sequential(nn.Linear(config.d_model, 1), nn.Sigmoid())
            self.rank_head = nn.Linear(config.d_model, 1)

        def forward(
            self,
            task_ids: object,
            task_mask: object,
            program_ids: object,
            program_mask: object,
            test_ids: object,
            test_mask: object,
            numeric: object,
        ) -> tuple[object, object, object]:
            task_vec = self._encode_segment(task_ids, task_mask)
            program_vec = self._encode_segment(program_ids, program_mask)

            batch, max_tests, test_len = test_ids.shape
            flat_test_ids = test_ids.reshape(batch * max_tests, test_len)
            flat_test_mask = test_mask.reshape(batch * max_tests, test_len)
            test_vec = self._encode_segment(flat_test_ids, flat_test_mask)
            test_vec = test_vec.reshape(batch, max_tests, -1)

            registers = self.register_tokens.expand(batch, -1, -1)
            fusion = torch.cat(
                [registers, task_vec.unsqueeze(1), program_vec.unsqueeze(1), test_vec],
                dim=1,
            )
            roles = torch.arange(fusion.shape[1], device=fusion.device).unsqueeze(0)
            fusion = fusion + self.role_embed(roles)
            test_present = (test_mask.sum(dim=-1) > 1).long()
            fusion_mask = torch.cat(
                [
                    torch.ones(
                        batch,
                        self.config.registers + 2,
                        device=fusion.device,
                        dtype=torch.long,
                    ),
                    test_present,
                ],
                dim=1,
            )
            fused = self.fusion_encoder(inputs_embeds=fusion, attention_mask=fusion_mask).last_hidden_state[:, 0]
            numeric_vec = self.numeric_mlp(numeric)
            state = self.trunk(torch.cat([fused, numeric_vec], dim=1))
            return self.hidden_head(state), self.solved_head(state), self.rank_head(state)

        def _encode_segment(self, input_ids: object, attention_mask: object) -> object:
            return self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    return _SharedStructuredReranker


SharedStructuredReranker = _make_shared_structured_reranker_class()


def _encode_rows(
    rows: list[dict[str, Any]],
    token_spec: TokenSpec,
    feature_spec: FeatureSpec,
    config: StructuredConfig,
):
    import torch

    task_name_to_id = {name: index for index, name in enumerate(sorted({row["task"]["name"] for row in rows}))}
    task_ids = []
    task_masks = []
    program_ids = []
    program_masks = []
    test_ids = []
    test_masks = []
    numeric = []
    hidden_targets = []
    solved_targets = []
    task_ids_for_loss = []
    for row in rows:
        structured = row_to_structured_tokens(row)
        ids, mask = encode_tokens(structured["task"], token_spec, config.task_len)
        task_ids.append(ids)
        task_masks.append(mask)

        ids, mask = encode_tokens(structured["program"], token_spec, config.program_len)
        program_ids.append(ids)
        program_masks.append(mask)

        row_test_ids = []
        row_test_masks = []
        for test_tokens in structured["tests"][: config.max_tests]:
            ids, mask = encode_tokens(test_tokens, token_spec, config.test_len)
            row_test_ids.append(ids)
            row_test_masks.append(mask)
        while len(row_test_ids) < config.max_tests:
            row_test_ids.append([0] * config.test_len)
            row_test_masks.append([0] * config.test_len)
        test_ids.append(row_test_ids)
        test_masks.append(row_test_masks)

        numeric.append(extract_features(row, feature_spec))
        hidden_targets.append([target_hidden_pass_rate(row)])
        solved_targets.append([float(bool(row["labels"]["solved"]))])
        task_ids_for_loss.append(task_name_to_id[row["task"]["name"]])

    tensors = (
        torch.tensor(task_ids, dtype=torch.long),
        torch.tensor(task_masks, dtype=torch.long),
        torch.tensor(program_ids, dtype=torch.long),
        torch.tensor(program_masks, dtype=torch.long),
        torch.tensor(test_ids, dtype=torch.long),
        torch.tensor(test_masks, dtype=torch.long),
        torch.tensor(numeric, dtype=torch.float32),
        torch.tensor(hidden_targets, dtype=torch.float32),
        torch.tensor(solved_targets, dtype=torch.float32),
    )
    return tensors, torch.tensor(task_ids_for_loss, dtype=torch.long)


def _predict(model: object, tensors: tuple[object, ...], device: object, batch_size: int):
    import torch

    outputs = []
    size = tensors[-1].shape[0]
    for start in range(0, size, batch_size):
        batch = [tensor[start : start + batch_size].to(device, non_blocking=True) for tensor in tensors[:-2]]
        hidden_pred, _, _ = model(*batch)
        outputs.append(hidden_pred)
    return torch.cat(outputs, dim=0)


def _pairwise_rank_loss(rank_score: object, target: object, task_ids: object):
    import torch
    import torch.nn.functional as F

    scores = rank_score.flatten()
    values = target.flatten()
    task_ids = task_ids.flatten()
    same_task = task_ids[:, None] == task_ids[None, :]
    better = values[:, None] > values[None, :] + 1e-6
    mask = same_task & better
    if not bool(mask.any()):
        return scores.sum() * 0.0
    margins = scores[:, None] - scores[None, :]
    return F.softplus(-margins[mask]).mean()


def _print_metrics(name: str, metrics: object) -> None:
    print(
        f"{name}: "
        f"selected={metrics.selected}/{metrics.tasks} "
        f"solved={metrics.solved}/{metrics.tasks} "
        f"mean_hidden={metrics.mean_hidden_pass_rate:.3f} "
        f"visible_overfit={metrics.visible_overfit} "
        f"mean_candidate={metrics.mean_candidate_index:.1f}"
    )


def _evaluate_epoch(
    model: object,
    eval_rows: list[dict[str, Any]],
    eval_tensors: tuple[object, ...],
    device: object,
    config: StructuredConfig,
    epoch: int,
    train_loss: float,
    mse_loss: object,
    visible_gated: bool,
    started: float,
) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        eval_pred = _predict(model, eval_tensors, device, config.batch_size)
        eval_mse = float(mse_loss(eval_pred, eval_tensors[-2].to(device)).detach().cpu())
    metrics = evaluate_predicted_selection(
        eval_rows,
        [float(value) for value in eval_pred.detach().cpu().flatten().tolist()],
        visible_gated=visible_gated,
    )
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "eval_mse": eval_mse,
        "tasks": metrics.tasks,
        "selected": metrics.selected,
        "selected_solved": metrics.solved,
        "selected_mean_hidden": metrics.mean_hidden_pass_rate,
        "visible_overfit": metrics.visible_overfit,
        "mean_candidate": metrics.mean_candidate_index,
        "elapsed_sec": time.time() - started,
    }
    if torch.cuda.is_available():
        record["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        record["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        record["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return record


def _init_run_dir(
    run_dir: Path,
    config: StructuredConfig,
    train_path: str,
    eval_path: str,
    train_rows: int,
    eval_rows: int,
    params: int,
    vocab: int,
    device: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "train_path": train_path,
        "eval_path": eval_path,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "params": params,
        "vocab": vocab,
        "device": device,
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    history = run_dir / "history.jsonl"
    if history.exists():
        history.unlink()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_summary(
    run_dir: Path,
    result: dict[str, Any],
    config: StructuredConfig,
    eval_rows: list[dict[str, Any]],
    visible_gated: bool,
) -> None:
    raw = evaluate_first_visible_pass(eval_rows)
    learned = evaluate_predicted_selection(
        eval_rows,
        result["eval_predictions"],
        visible_gated=visible_gated,
    )
    oracle = evaluate_oracle_best(eval_rows, visible_gated=visible_gated)
    summary = {
        "config": asdict(config),
        "params": result["params"],
        "device": result["device"],
        "train_mse": result["train_mse"],
        "eval_mse": result["eval_mse"],
        "first_visible": _metrics_to_dict(raw),
        "structured_shared": _metrics_to_dict(learned),
        "oracle_visible_gated": _metrics_to_dict(oracle),
        "history": result["history"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _metrics_to_dict(metrics: object) -> dict[str, Any]:
    return {
        "tasks": metrics.tasks,
        "selected": metrics.selected,
        "solved": metrics.solved,
        "visible_overfit": metrics.visible_overfit,
        "mean_hidden_pass_rate": metrics.mean_hidden_pass_rate,
        "mean_candidate_index": metrics.mean_candidate_index,
    }


def _write_plot(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    eval_mse = [record["eval_mse"] for record in history]
    mean_hidden = [record["selected_mean_hidden"] for record in history]
    solved = [record["selected_solved"] for record in history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, eval_mse, label="eval_mse")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, mean_hidden, label="selected_mean_hidden")
    axes[1].plot(epochs, solved, label="selected_solved")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("selection")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
