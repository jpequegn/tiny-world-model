"""Character-level vocabulary with a dedicated MASK token."""

from __future__ import annotations

_PRINTABLE = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

MASK_TOKEN = "<MASK>"
PAD_TOKEN = "<PAD>"

_SPECIAL = [PAD_TOKEN, MASK_TOKEN]
CHARS = _SPECIAL + list(_PRINTABLE)

VOCAB_SIZE = len(CHARS)
MASK_ID = CHARS.index(MASK_TOKEN)
PAD_ID = CHARS.index(PAD_TOKEN)

_char2id: dict[str, int] = {c: i for i, c in enumerate(CHARS)}
_id2char: dict[int, str] = {i: c for i, c in enumerate(CHARS)}


def encode(text: str) -> list[int]:
    """Encode a string to a list of token IDs (unknown chars → PAD_ID)."""
    return [_char2id.get(ch, PAD_ID) for ch in text]


def decode(ids: list[int]) -> str:
    """Decode token IDs to a string (skipping PAD and MASK)."""
    return "".join(
        _id2char.get(i, "?")
        for i in ids
        if i not in (PAD_ID, MASK_ID)
    )
