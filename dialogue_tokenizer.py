from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Role:
    USER: str = "user"
    ASSISTANT: str = "assistant"


class ByteDialogueTokenizer:
    """Tiny UTF-8 byte tokenizer with a few chat-specific special tokens."""

    def __init__(self) -> None:
        specials = ["<|pad|>", "<|bos|>", "<|eot|>", "<|user|>", "<|assistant|>"]
        self.special_tokens = specials
        self.token_to_id = {token: 256 + i for i, token in enumerate(specials)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<|pad|>"]
        self.bos_id = self.token_to_id["<|bos|>"]
        self.eot_id = self.token_to_id["<|eot|>"]
        self.user_id = self.token_to_id["<|user|>"]
        self.assistant_id = self.token_to_id["<|assistant|>"]
        self.vocab_size = 256 + len(specials)

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        buf = bytearray()
        pieces: list[str] = []
        for idx in ids:
            if idx < 256:
                buf.append(idx)
                continue
            if buf:
                pieces.append(buf.decode("utf-8", errors="replace"))
                buf.clear()
            if not skip_special_tokens and idx in self.id_to_token:
                pieces.append(self.id_to_token[idx])
        if buf:
            pieces.append(buf.decode("utf-8", errors="replace"))
        return "".join(pieces)

    def encode_message(
        self,
        role: str,
        text: str,
        *,
        predict_text: bool,
        include_eot: bool = True,
    ) -> tuple[list[int], list[int]]:
        if role not in (Role.USER, Role.ASSISTANT):
            raise ValueError(f"unsupported role: {role}")

        role_id = self.user_id if role == Role.USER else self.assistant_id
        text_ids = self.encode_text(text.strip())
        ids = [role_id, *text_ids]
        mask = [0] + [1 if predict_text else 0] * len(text_ids)

        if include_eot:
            ids.append(self.eot_id)
            mask.append(1 if predict_text else 0)

        return ids, mask

    def encode_dialogue(self, turns: Iterable[str]) -> tuple[list[int], list[int]]:
        turn_list = [turn.strip() for turn in turns if turn and turn.strip()]
        ids = [self.bos_id]
        mask = [0]
        pair_count = len(turn_list) // 2
        for pair_idx in range(pair_count):
            user_text = turn_list[2 * pair_idx]
            assistant_text = turn_list[2 * pair_idx + 1]
            user_ids, user_mask = self.encode_message(Role.USER, user_text, predict_text=False)
            assistant_ids, assistant_mask = self.encode_message(
                Role.ASSISTANT,
                assistant_text,
                predict_text=True,
            )
            ids.extend(user_ids)
            ids.extend(assistant_ids)
            mask.extend(user_mask)
            mask.extend(assistant_mask)
        return ids, mask

    def build_prompt(self, history: Sequence[tuple[str, str]]) -> list[int]:
        ids = [self.bos_id]
        for role, text in history:
            message_ids, _ = self.encode_message(role, text, predict_text=False)
            ids.extend(message_ids)
        ids.append(self.assistant_id)
        return ids
