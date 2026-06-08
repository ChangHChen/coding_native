from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minileet.features import load_rows, target_hidden_pass_rate
from minileet.rerank import (
    evaluate_first_visible_pass,
    evaluate_oracle_best,
    evaluate_predicted_selection,
)
from minileet.sequence import TokenSpec, encode_row, fit_token_spec


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    max_len: int
    max_vocab: int
    d_model: int
    layers: int
    heads: int
    ff_mult: int
    dropout: float
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(prog="minileet.transformer_rerank")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-vocab", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-visible-fail", action="store_true")
    parser.add_argument("--model-out", default="")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        max_vocab=args.max_vocab,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        seed=args.seed,
    )
    train_rows = load_rows(Path(args.train))
    eval_rows = load_rows(Path(args.eval))
    result = train_and_score(train_rows, eval_rows, config)
    if args.model_out:
        _save_model(Path(args.model_out), result, config)

    raw = evaluate_first_visible_pass(eval_rows)
    learned = evaluate_predicted_selection(
        eval_rows,
        result["eval_predictions"],
        visible_gated=not args.allow_visible_fail,
    )
    oracle = evaluate_oracle_best(eval_rows, visible_gated=not args.allow_visible_fail)

    print(f"device={result['device']}")
    print(f"vocab={len(result['token_spec'].vocab)} max_len={config.max_len}")
    print(f"train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    print(f"train_mse={result['train_mse']:.5f} eval_mse={result['eval_mse']:.5f}")
    for index, loss in enumerate(result["epoch_losses"], start=1):
        print(f"epoch={index} train_loss={loss:.5f}")
    _print_metrics("first_visible", raw)
    _print_metrics("transformer", learned)
    _print_metrics("oracle_visible_gated", oracle)


def train_and_score(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    config: TrainConfig,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_spec = fit_token_spec(train_rows, max_vocab=config.max_vocab)
    x_train, mask_train, y_train = _encode_rows(train_rows, token_spec, config.max_len)
    x_eval, mask_eval, y_eval = _encode_rows(eval_rows, token_spec, config.max_len)

    dataset = TensorDataset(x_train, mask_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = TransformerRegressor(
        vocab_size=len(token_spec.vocab),
        max_len=config.max_len,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    epoch_losses: list[float] = []

    for _ in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_items = 0
        for input_ids, attention_mask, targets in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(input_ids, attention_mask)
            loss = loss_fn(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * input_ids.shape[0]
            total_items += input_ids.shape[0]
        epoch_losses.append(total_loss / total_items)

    model.eval()
    with torch.no_grad():
        train_pred = _predict(model, x_train, mask_train, device, config.batch_size)
        eval_pred = _predict(model, x_eval, mask_eval, device, config.batch_size)
        train_mse = float(loss_fn(train_pred, y_train.to(device)).detach().cpu())
        eval_mse = float(loss_fn(eval_pred, y_eval.to(device)).detach().cpu())

    return {
        "device": str(device),
        "token_spec": token_spec,
        "model": model,
        "train_mse": train_mse,
        "eval_mse": eval_mse,
        "epoch_losses": epoch_losses,
        "eval_predictions": [float(value) for value in eval_pred.detach().cpu().flatten().tolist()],
    }


class TransformerRegressor:
    pass


def _make_transformer_regressor_class():
    import torch
    from torch import nn

    class _TransformerRegressor(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            max_len: int,
            d_model: int,
            layers: int,
            heads: int,
            ff_mult: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_embed = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * ff_mult,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        def forward(self, input_ids: object, attention_mask: object) -> object:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            x = self.token_embed(input_ids) + self.pos_embed(positions)
            padding_mask = attention_mask == 0
            encoded = self.encoder(x, src_key_padding_mask=padding_mask)
            return self.head(encoded[:, 0, :])

    return _TransformerRegressor


TransformerRegressor = _make_transformer_regressor_class()


def _encode_rows(
    rows: list[dict[str, Any]],
    token_spec: TokenSpec,
    max_len: int,
):
    import torch

    encoded = [encode_row(row, token_spec, max_len) for row in rows]
    input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
    attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)
    targets = torch.tensor([[target_hidden_pass_rate(row)] for row in rows], dtype=torch.float32)
    return input_ids, attention_mask, targets


def _predict(model: object, input_ids: object, attention_mask: object, device: object, batch_size: int):
    import torch

    outputs = []
    for start in range(0, input_ids.shape[0], batch_size):
        batch_ids = input_ids[start : start + batch_size].to(device, non_blocking=True)
        batch_mask = attention_mask[start : start + batch_size].to(device, non_blocking=True)
        outputs.append(model(batch_ids, batch_mask))
    return torch.cat(outputs, dim=0)


def _print_metrics(name: str, metrics: object) -> None:
    print(
        f"{name}: "
        f"selected={metrics.selected}/{metrics.tasks} "
        f"solved={metrics.solved}/{metrics.tasks} "
        f"mean_hidden={metrics.mean_hidden_pass_rate:.3f} "
        f"visible_overfit={metrics.visible_overfit} "
        f"mean_candidate={metrics.mean_candidate_index:.1f}"
    )


def _save_model(path: Path, result: dict[str, Any], config: TrainConfig) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "token_spec": result["token_spec"].to_json(),
            "config": config.__dict__,
        },
        path,
    )


if __name__ == "__main__":
    main()

