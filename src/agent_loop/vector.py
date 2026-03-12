from __future__ import annotations

import hashlib
import math
import re
from array import array

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
EMBEDDING_DIM = 32


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def pack_embedding(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def unpack_embedding(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    values = array("f")
    values.frombytes(blob)
    return list(values)
