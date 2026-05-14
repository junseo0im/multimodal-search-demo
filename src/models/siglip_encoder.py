from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return arr / denom


@dataclass
class SigLIPEncoder:
    base_model_id: str = "google/siglip2-base-patch16-224"
    adapter_path: str | None = None
    device: str | None = None
    dtype: torch.dtype = torch.float16
    normalize: bool = True

    def __post_init__(self) -> None:
        from peft import PeftModel
        from transformers import AutoModel, AutoProcessor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        base = AutoModel.from_pretrained(self.base_model_id, torch_dtype=self.dtype).to(self.device)
        self.model = PeftModel.from_pretrained(base, self.adapter_path).eval() if self.adapter_path else base.eval()

    @torch.no_grad()
    def encode_images(self, paths: Iterable[str | Path], batch_size: int = 16) -> np.ndarray:
        paths = list(paths)
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(path).convert("RGB") for path in paths[i : i + batch_size]]
            batch = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.amp.autocast(self.device_type, dtype=self.dtype, enabled=self.device_type == "cuda"):
                out = self.model.vision_model(pixel_values=batch["pixel_values"])
            all_vecs.append(out.pooler_output.float().cpu().numpy())
        arr = np.vstack(all_vecs).astype(np.float32)
        return l2_normalize(arr) if self.normalize else arr

    @torch.no_grad()
    def encode_texts(self, texts: str | Iterable[str], batch_size: int = 32) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(items), batch_size):
            batch = self.processor(
                text=items[i : i + batch_size],
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            text_inputs = {k: v for k, v in batch.items() if "pixel" not in k}
            with torch.amp.autocast(self.device_type, dtype=self.dtype, enabled=self.device_type == "cuda"):
                out = self.model.text_model(**text_inputs)
            all_vecs.append(out.pooler_output.float().cpu().numpy())
        arr = np.vstack(all_vecs).astype(np.float32)
        arr = l2_normalize(arr) if self.normalize else arr
        return arr[0] if single else arr
