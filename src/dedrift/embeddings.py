"""Pinned embedders for Tier-2 semantic signatures (SPEC.md §4, principle 5).

The embedding model is pinned FOREVER per project: changing the embedder
invalidates all historical comparisons, so dedrift refuses to compare across
embedder identities rather than silently producing garbage. The pin lives in
``.dedrift/embedder.json`` and is checked on every embedding computation.

Two embedder families ship:

- ``hash`` — a deterministic character-n-gram hashing embedder with zero ML
  dependencies. Weak semantics, but fully reproducible and dependency-free;
  used by the demo and tests, and honest about what it is.
- ``st:<model-name>`` — any sentence-transformers model (requires the
  ``dedrift[embeddings]`` extra). E.g. ``st:all-MiniLM-L6-v2``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from dedrift.schema import InteractionRecord
from dedrift.store import Store, _atomic_private_writer, _harden_permissions

EMBEDDER_FILE = "embedder.json"
CACHE_FILE = "embedding_cache.npz"


class Embedder(Protocol):
    """Anything that maps a list of texts to a (n, d) float array."""

    def __call__(self, texts: list[str]) -> npt.NDArray[np.float64]:
        """Embed texts; rows aligned with input order."""
        ...


class EmbedderMismatchError(RuntimeError):
    """Raised when a comparison would span two different embedders.

    Changing the embedder invalidates all history (principle 5). The remedy
    is a new project (or deliberately re-embedding everything), never a
    silent cross-embedder comparison.
    """


def _hash_embedder(texts: list[str], dim: int = 64, n: int = 3) -> npt.NDArray[np.float64]:
    """Deterministic character-n-gram hashing embedder (zero dependencies).

    Each n-gram is hashed into one of ``dim`` buckets (signed hashing trick);
    vectors are L2-normalized. Reproducible across platforms and versions
    because it depends only on sha256.

    Args:
        texts: Input texts.
        dim: Embedding dimensionality.
        n: Character n-gram size.

    Returns:
        Array of shape (len(texts), dim).
    """
    out = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        padded = f"  {text.lower()}  "
        for j in range(len(padded) - n + 1):
            gram = padded[j : j + n].encode("utf-8")
            digest = hashlib.sha256(gram).digest()
            bucket = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            out[i, bucket] += sign
        norm = np.linalg.norm(out[i])
        if norm > 0:
            out[i] /= norm
    return out


def _load_sentence_transformer(model_name: str) -> Embedder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - extra not installed
        msg = (
            f"embedder 'st:{model_name}' requires the embeddings extra: "
            "pip install 'dedrift[embeddings]'"
        )
        raise RuntimeError(msg) from exc
    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> npt.NDArray[np.float64]:
        return np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float64)

    return encode


def resolve_embedder(name: str) -> Embedder:
    """Resolve an embedder identifier to a callable.

    Args:
        name: ``"hash"`` or ``"st:<sentence-transformers model>"``.

    Returns:
        The embedding callable.

    Raises:
        ValueError: For unknown identifiers.
    """
    if name == "hash":
        return lambda texts: _hash_embedder(texts)
    if name.startswith("st:"):
        return _load_sentence_transformer(name[3:])
    msg = f"unknown embedder {name!r}; expected 'hash' or 'st:<model>'"
    raise ValueError(msg)


# -- pinning -------------------------------------------------------------------


def pin_embedder(store: Store, name: str) -> None:
    """Pin the project's embedder forever (refuses to overwrite a different pin).

    Args:
        store: The project store.
        name: Embedder identifier.

    Raises:
        EmbedderMismatchError: If a different embedder is already pinned.
    """
    conn = store.connect()
    if conn.in_transaction:
        raise RuntimeError("pin_embedder requires ownership of the SQLite transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        current = get_pinned_embedder(store)
        if current is not None and current != name:
            msg = (
                f"project embedder is pinned to {current!r}; refusing to switch to "
                f"{name!r} — changing the embedder invalidates all history. "
                "Start a new project if you truly intend this."
            )
            raise EmbedderMismatchError(msg)
        if current is None:
            path = store.project_dir / EMBEDDER_FILE
            payload = json.dumps(
                {"embedder": name, "pinned_at": datetime.now(timezone.utc).isoformat()},
                indent=2,
            ).encode("utf-8")
            with _atomic_private_writer(path) as stream:
                stream.write(payload)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def get_pinned_embedder(store: Store) -> str | None:
    """Return the pinned embedder identifier, or None if never pinned."""
    path = store.project_dir / EMBEDDER_FILE
    if not path.exists():
        return None
    _harden_permissions(path, 0o600)
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data["embedder"])


# -- cached record embeddings --------------------------------------------------


def embed_records(
    store: Store,
    records: list[InteractionRecord],
    embedder_name: str | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Embed record output texts with the pinned embedder, using the cache.

    The cache is keyed by record ID and stamped with the embedder identity;
    a cache written by a different embedder triggers
    :class:`EmbedderMismatchError` rather than silent mixing.

    Args:
        store: The project store.
        records: Records to embed.
        embedder_name: Override (defaults to the pinned embedder).

    Returns:
        Mapping record_id -> embedding vector.

    Raises:
        EmbedderMismatchError: On any embedder identity conflict.
        ValueError: If no embedder is pinned and none is given.
    """
    name = embedder_name or get_pinned_embedder(store)
    if name is None:
        msg = "no embedder pinned; run `dedrift embedder pin <name>` first"
        raise ValueError(msg)
    pinned = get_pinned_embedder(store)
    if pinned is not None and pinned != name:
        msg = f"requested embedder {name!r} != pinned {pinned!r}; refusing"
        raise EmbedderMismatchError(msg)

    cache_path = store.project_dir / CACHE_FILE
    cached: dict[str, npt.NDArray[np.float64]] = {}
    if cache_path.exists():
        _harden_permissions(cache_path, 0o600)
        with np.load(cache_path, allow_pickle=False) as data:
            cache_embedder = str(data["__embedder__"])
            if cache_embedder != name:
                msg = (
                    f"embedding cache was written by {cache_embedder!r} but the "
                    f"pinned embedder is {name!r}; refusing to mix"
                )
                raise EmbedderMismatchError(msg)
            cached = {k: data[k] for k in data.files if k != "__embedder__"}

    missing = [r for r in records if r.id not in cached]
    if missing:
        encode: Callable[[list[str]], Any] = resolve_embedder(name)
        vectors = encode([r.output.text for r in missing])
        for record, vec in zip(missing, vectors, strict=True):
            cached[record.id] = np.asarray(vec, dtype=np.float64)
        payload: dict[str, Any] = {"__embedder__": np.asarray(name), **cached}
        with _atomic_private_writer(cache_path) as stream:
            np.savez_compressed(stream, **payload)
    return {r.id: cached[r.id] for r in records}
