"""Configuration: defaults from config.yaml, overridden by environment variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # source_code/
DEFAULT_CONFIG_FILE = ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    data_dir: str = "data/documents"
    collection_prefix: str = "math_handbook"

    pdf_parser: str = "pymupdf"
    ocr_enabled: bool = True
    ocr_dpi: int = 300
    ocr_min_chars: int = 50

    chunking: str = "structure"
    chunk_size: int = 1200
    chunk_overlap: int = 150

    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:3b"
    temperature: float = 0.1
    num_ctx: int = 8192

    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant_local"
    retrieval_mode: str = "hybrid"
    top_k: int = 5

    prompt_mode: str = "strict"

    # ---- derived helpers -------------------------------------------------
    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    @property
    def data_path(self) -> Path:
        return self.path(self.data_dir)

    @property
    def collection_name(self) -> str:
        """One collection per (embedding model, parser, chunking) so experiments coexist."""
        raw = f"{self.collection_prefix}__{self.embed_model}__{self.pdf_parser}__{self.chunking}{self.chunk_size}"
        return re.sub(r"[^A-Za-z0-9_\-]", "-", raw)

    def with_overrides(self, **kwargs) -> "Config":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


def _coerce(value: str, target_type):
    if target_type is bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return str(value)


def load_config(config_file: str | os.PathLike | None = None) -> Config:
    """Load config.yaml (if present) and apply UPPER_CASE environment overrides."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT.parent / ".env")
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    file = Path(config_file or os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_FILE))
    values: dict = {}
    if file.exists():
        values = yaml.safe_load(file.read_text()) or {}

    types = {f.name: type(f.default) for f in fields(Config)}
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"Unknown config keys in {file}: {sorted(unknown)}")

    for name, typ in types.items():
        env = os.environ.get(name.upper())
        if env is not None:
            values[name] = env
        if name in values and values[name] is not None:
            values[name] = _coerce(values[name], typ)
    return Config(**values)
