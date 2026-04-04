from __future__ import annotations

import argparse
import contextlib
import math
import pickle
import pathlib
import time
from dataclasses import asdict
from pathlib import Path

import torch

from dialogue_tokenizer import ByteDialogueTokenizer
from model import GPTConfig, GPTLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small dialogue-oriented GPT on DailyDialog."
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/dailydialog_small"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true",
                        help="Enable linear/layernorm bias terms.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model with torch.compile.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from an existing checkpoint.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_dataset(cache_dir: Path) -> dict[str, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "dailydialog_byte_cache.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc

    tokenizer = ByteDialogueTokenizer()
    dataset = load_dialogue_dataset(load_dataset)

    processed: dict[str, torch.Tensor] = {}
    for split_name in ("train", "validation"):
        token_stream: list[int] = []
        mask_stream: list[int] = []
        for dialog in dataset[split_name]["dialog"]:
            ids, mask = tokenizer.encode_dialogue(dialog)
            if len(ids) < 3:
                continue
            token_stream.extend(ids)
            mask_stream.extend(mask)
        processed[f"{split_name}_ids"] = torch.tensor(
            token_stream, dtype=torch.long)
        processed[f"{split_name}_mask"] = torch.tensor(
            mask_stream, dtype=torch.float32)

    processed["dataset_name"] = "daily_dialog"
    processed["tokenizer_vocab_size"] = tokenizer.vocab_size
    torch.save(processed, cache_path)
    return processed


def load_dialogue_dataset(load_dataset_fn):
    dataset_candidates = [
        "daily_dialog",
        "OpenRL/daily_dialog",
    ]
    last_error: Exception | None = None

    for dataset_name in dataset_candidates:
        try:
            print(f"loading dataset: {dataset_name}")
            return load_dataset_fn(dataset_name)
        except RuntimeError as exc:
            last_error = exc
            if "Dataset scripts are no longer supported" not in str(exc):
                raise
            print(f"dataset loader rejected {dataset_name}: {exc}")
        except Exception as exc:
            last_error = exc
            print(f"failed to load {dataset_name}: {exc}")

    raise RuntimeError(
        "Unable to load a DailyDialog dataset source. "
        "If you are on datasets 4.x, install `datasets<4` or keep using the "
        "built-in fallback dataset mirror."
    ) from last_error


def get_batch(
    token_stream: torch.Tensor,
    mask_stream: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = token_stream.size(0) - block_size - 1
    if max_start <= 0:
        raise RuntimeError("dataset is too small for the selected block size")

    starts = torch.randint(0, max_start, (batch_size,)).tolist()
    x = torch.stack([token_stream[i: i + block_size] for i in starts])
    y = torch.stack([token_stream[i + 1: i + block_size + 1] for i in starts])
    m = torch.stack([mask_stream[i + 1: i + block_size + 1] for i in starts])
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    autocast_ctx,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    for split, ids, mask in (
        ("train", train_ids, train_mask),
        ("val", val_ids, val_mask),
    ):
        losses = []
        for _ in range(eval_iters):
            x, y, m = get_batch(ids, mask, batch_size=batch_size,
                                block_size=block_size, device=device)
            with autocast_ctx():
                _, loss = model(x, y, mask=m)
            losses.append(loss.item())
        metrics[split] = sum(losses) / len(losses)
    model.train()
    return metrics


def build_autocast(device: str, amp_dtype: torch.dtype):
    if device.startswith("cuda"):
        return lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    if hasattr(torch, "autocast"):
        return lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext


def get_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    step: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(config),
        "step": step,
        "best_val_loss": best_val_loss,
        "train_args": serialized_args,
        "tokenizer": {"kind": "byte_dialogue_v1"},
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        with torch.serialization.safe_globals([pathlib.PosixPath, pathlib.WindowsPath]):
            return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = args.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    autocast_ctx = build_autocast(device, amp_dtype)
    use_scaler = device_type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    data = prepare_dataset(args.cache_dir)
    train_ids = data["train_ids"]
    train_mask = data["train_mask"]
    val_ids = data["validation_ids"]
    val_mask = data["validation_mask"]

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=int(data["tokenizer_vocab_size"]),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPTLanguageModel(config).to(device)
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resumed from {args.resume} at step {start_step}")
    raw_model = model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    checkpoint_path = args.output_dir / "ckpt.pt"
    best_path = args.output_dir / "best.pt"

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])

    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"dataset: {data['dataset_name']}")
    print(f"vocab size: {config.vocab_size}")
    print(f"parameters: {total_params / 1e6:.2f}M")
    print(
        "default training settings: "
        f"--batch-size {args.batch_size} --grad-accum-steps {args.grad_accum_steps} --max-steps {args.max_steps}"
    )

    model.train()
    running_mfu_tokens = 0
    window_start = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(
            step,
            base_lr=args.learning_rate,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum_steps):
            x, y, m = get_batch(
                train_ids,
                train_mask,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
            )
            with autocast_ctx():
                _, loss = model(x, y, mask=m)
                loss = loss / args.grad_accum_steps
            loss_accum += loss.item()
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_mfu_tokens += x.numel()

        if args.grad_clip > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), args.grad_clip)

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % 20 == 0:
            elapsed = max(time.time() - window_start, 1e-6)
            toks_per_sec = running_mfu_tokens / elapsed
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.6f} | toks/s {toks_per_sec:,.0f}"
            )
            running_mfu_tokens = 0
            window_start = time.time()

        if step > 0 and step % args.eval_interval == 0:
            metrics = estimate_loss(
                model,
                train_ids,
                train_mask,
                val_ids,
                val_mask,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
                autocast_ctx=autocast_ctx,
            )
            print(
                f"eval step {step:5d} | train {metrics['train']:.4f} | val {metrics['val']:.4f}"
            )
            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                save_checkpoint(
                    best_path,
                    model=raw_model,
                    optimizer=optimizer,
                    config=config,
                    step=step,
                    best_val_loss=best_val_loss,
                    args=args,
                )
                print(f"saved new best checkpoint to {best_path}")
            save_checkpoint(
                checkpoint_path,
                model=raw_model,
                optimizer=optimizer,
                config=config,
                step=step,
                best_val_loss=best_val_loss,
                args=args,
            )

        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(
                checkpoint_path,
                model=raw_model,
                optimizer=optimizer,
                config=config,
                step=step,
                best_val_loss=best_val_loss,
                args=args,
            )

    save_checkpoint(
        checkpoint_path,
        model=raw_model,
        optimizer=optimizer,
        config=config,
        step=max(start_step, args.max_steps - 1),
        best_val_loss=best_val_loss,
        args=args,
    )
    print(f"training complete, final checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
