from __future__ import annotations

import json
import re
import urllib.request
import zlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

# CJK Unified Ideographs, Extension A, Hiragana, Katakana, Hangul Syllables
_CJK_CHAR_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_TOKEN_UNIT_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]"
    r"|[^\W\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+"
)


def _tokenize_units(text: str) -> tuple[list[str], bool]:
    """Tokenizes text into linguistic units (CJK characters individually, non-CJK words via \\w+).

    Returns (lowercased_units, is_cjk_dominant).
    """
    units = [m.lower() for m in _TOKEN_UNIT_RE.findall(text)]
    if not units:
        return [], False
    cjk_count = sum(1 for u in units if len(u) == 1 and _CJK_CHAR_RE.match(u))
    return units, (cjk_count / len(units)) >= 0.5


class EmbedderMismatchError(RuntimeError):
    """Raised when an existing store's recorded embedder or vector dimension conflicts with the active embedder."""


class Embedder(Protocol):
    """Protocol for dense or sparse text embedding."""

    def embed_text(self, text: str) -> np.ndarray: ...


def get_embedder_info(embedder: Embedder) -> tuple[str, int | None]:
    """Returns (embedder_id, known_dim) for metadata tracking and mismatch detection."""
    explicit_id = getattr(embedder, "embedder_id", None)
    if explicit_id:
        embedder_id = str(explicit_id)
    else:
        cls_name = type(embedder).__qualname__
        model = getattr(embedder, "model", None)
        dim_attr = getattr(embedder, "dim", None)
        if model is not None:
            embedder_id = f"{cls_name}(model={model})"
        elif dim_attr is not None:
            embedder_id = f"{cls_name}(dim={dim_attr})"
        else:
            embedder_id = cls_name

    raw_dim = getattr(embedder, "dim", None)
    known_dim = int(raw_dim) if isinstance(raw_dim, int) and raw_dim > 0 else None
    return embedder_id, known_dim


def embed_texts_batch(embedder: Embedder, texts: Sequence[str]) -> list[np.ndarray]:
    """Embeds a sequence of texts using embed_batch() when supported, falling back to embed_text()."""
    if not texts:
        return []
    batch_fn = getattr(embedder, "embed_batch", None)
    if callable(batch_fn):
        return [np.asarray(v, dtype=np.float32) for v in batch_fn(texts)]
    return [np.asarray(embedder.embed_text(t), dtype=np.float32) for t in texts]


class OfflineNgramEmbedder:
    """Zero-dependency offline character + word n-gram signed feature hashing embedder.

    Uses Weinberger signed feature hashing with deterministic CRC32 across words,
    individual CJK units, adjacent CJK character bigrams, and character 3-grams
    so unrelated tokens have zero expected collision bias.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        clean = " ".join(text.lower().split())
        if not clean:
            return np.zeros(self.dim, dtype=np.float32)

        vec = np.zeros(self.dim, dtype=np.float32)
        units, _ = _tokenize_units(clean)
        for token in units:
            h = zlib.crc32(token.encode("utf-8"))
            sign = 1.0 if (h & 1) else -1.0
            idx = (h >> 1) % self.dim
            vec[idx] += sign * 1.0

        # Adjacent CJK character bigrams (captures 2-character CJK compounds)
        for i in range(len(clean) - 1):
            c1, c2 = clean[i], clean[i + 1]
            if _CJK_CHAR_RE.match(c1) and _CJK_CHAR_RE.match(c2):
                h = zlib.crc32((c1 + c2).encode("utf-8"))
                sign = 1.0 if (h & 1) else -1.0
                idx = (h >> 1) % self.dim
                vec[idx] += sign * 1.0

        # Character 3-grams
        for i in range(len(clean) - 2):
            tri = clean[i : i + 3]
            h = zlib.crc32(tri.encode("utf-8"))
            sign = 1.0 if (h & 1) else -1.0
            idx = (h >> 1) % self.dim
            vec[idx] += sign * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self.embed_text(t) for t in texts]


class CallableEmbedder:
    """Adapter wrapping a user-supplied embedding function (and optional batch function)."""

    def __init__(
        self,
        fn: Callable[[str], Sequence[float] | np.ndarray],
        *,
        batch_fn: Callable[[Sequence[str]], Sequence[Sequence[float] | np.ndarray]] | None = None,
        embedder_id: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._fn = fn
        self._batch_fn = batch_fn
        self.embedder_id = embedder_id
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        return np.asarray(self._fn(text), dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if self._batch_fn is not None:
            return [np.asarray(v, dtype=np.float32) for v in self._batch_fn(texts)]
        return [self.embed_text(t) for t in texts]


class OpenAICompatibleEmbedder:
    """Zero-dependency HTTP embedder for OpenAI, LiteLLM, vLLM, or compatible /v1/embeddings endpoints."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        dim: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers) if headers else {}
        self.dim = dim

    def _request(self, inputs: list[str]) -> list[np.ndarray]:
        payload = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            req_headers.setdefault("Authorization", f"Bearer {self.api_key}")
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers=req_headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        data = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        return [np.asarray(item["embedding"], dtype=np.float32) for item in data]

    def embed_text(self, text: str) -> np.ndarray:
        vectors = self._request([text])
        return vectors[0] if vectors else np.zeros(0, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return self._request(list(texts))


class OllamaEmbedder:
    """Zero-dependency HTTP embedder for local Ollama /api/embed endpoints."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        dim: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dim = dim

    def _request(self, inputs: list[str]) -> list[np.ndarray]:
        payload = json.dumps({"model": self.model, "input": inputs}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        embeddings = body.get("embeddings")
        if embeddings is None and "embedding" in body:
            embeddings = [body["embedding"]]
        return [np.asarray(vec, dtype=np.float32) for vec in (embeddings or ())]

    def embed_text(self, text: str) -> np.ndarray:
        vectors = self._request([text])
        return vectors[0] if vectors else np.zeros(0, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return self._request(list(texts))
