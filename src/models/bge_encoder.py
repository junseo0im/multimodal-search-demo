from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BGEEncoder:
    model_id: str = "BAAI/bge-m3"
    use_fp16: bool = True
    batch_size: int = 64
    max_length: int = 256

    def __post_init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(self.model_id, use_fp16=self.use_fp16)

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        output = self.model.encode(
            items,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        arr = np.asarray(output, dtype=np.float32)
        return arr[0] if single else arr

