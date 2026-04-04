from __future__ import annotations

import argparse
import pickle
import pathlib
from pathlib import Path

import torch
from torch.nn import functional as F

from dialogue_tokenizer import ByteDialogueTokenizer, Role
from model import GPTConfig, GPTLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a checkpoint trained by train.py.")
    parser.add_argument("--checkpoint", type=Path, default=Path("out/dailydialog_small/best.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--history-turns", type=int, default=6, help="Keep at most this many user/assistant pairs.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[GPTLanguageModel, ByteDialogueTokenizer]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        with torch.serialization.safe_globals([pathlib.PosixPath, pathlib.WindowsPath]):
            checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, ByteDialogueTokenizer()


def trim_history(
    history: list[tuple[str, str]],
    tokenizer: ByteDialogueTokenizer,
    block_size: int,
    reserve_tokens: int,
) -> list[tuple[str, str]]:
    trimmed = history[:]
    while trimmed:
        prompt_ids = tokenizer.build_prompt(trimmed)
        if len(prompt_ids) + reserve_tokens <= block_size:
            break
        if len(trimmed) >= 2:
            trimmed = trimmed[2:]
        else:
            trimmed = trimmed[1:]
    return trimmed


@torch.no_grad()
def generate_reply(
    model: GPTLanguageModel,
    tokenizer: ByteDialogueTokenizer,
    history: list[tuple[str, str]],
    *,
    device: str,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
) -> str:
    prompt_ids = tokenizer.build_prompt(history)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    disallowed_ids = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.user_id,
        tokenizer.assistant_id,
    }
    generated: list[int] = []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        for token_id in disallowed_ids:
            next_token_logits[:, token_id] = -float("inf")
        if top_k is not None:
            top_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < top_values[:, [-1]]] = -float("inf")
        probs = F.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        if token_id == tokenizer.eot_id:
            break
        generated.append(token_id)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(generated).strip()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    history: list[tuple[str, str]] = []

    print("Commands: /reset, /quit")
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            history.clear()
            print("assistant> context cleared")
            continue

        history.append((Role.USER, user_text))
        history = trim_history(
            history,
            tokenizer,
            model.config.block_size,
            reserve_tokens=args.max_new_tokens,
        )
        reply = generate_reply(
            model,
            tokenizer,
            history,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            max_new_tokens=args.max_new_tokens,
        )
        if not reply:
            reply = "..."
        print(f"assistant> {reply}")
        history.append((Role.ASSISTANT, reply))
        max_messages = max(2, args.history_turns * 2)
        if len(history) > max_messages:
            history = history[-max_messages:]


if __name__ == "__main__":
    main()
