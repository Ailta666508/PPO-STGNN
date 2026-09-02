from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from .config import EnvConfig, PPOConfig, TrainConfig


ConfigType = TypeVar("ConfigType", EnvConfig, PPOConfig)


class ExperimentConfigError(ValueError):
    """Raised when a serialized experiment configuration is invalid."""


def _build_section(cls: type[ConfigType], payload: Any, section: str) -> ConfigType:
    if not isinstance(payload, Mapping):
        raise ExperimentConfigError(f"{section} must be a JSON object")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ExperimentConfigError(f"Unknown {section} fields: {', '.join(unknown)}")
    return cls(**dict(payload))


def train_config_from_dict(payload: Any) -> TrainConfig:
    """Rebuild a training configuration while rejecting silent field typos."""
    if not isinstance(payload, Mapping):
        raise ExperimentConfigError("training config must be a JSON object")

    allowed = {item.name for item in fields(TrainConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ExperimentConfigError(f"Unknown training config fields: {', '.join(unknown)}")

    values = dict(payload)
    values["env"] = _build_section(EnvConfig, values.get("env", {}), "env")
    values["ppo"] = _build_section(PPOConfig, values.get("ppo", {}), "ppo")
    return TrainConfig(**values)


def config_fingerprint(config: TrainConfig) -> str:
    """Return a short stable identifier for the exact resolved configuration."""
    canonical = json.dumps(
        config.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def save_train_config(config: TrainConfig, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_fingerprint(config)


def load_train_config(path: str | Path) -> TrainConfig:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentConfigError(f"Unable to load experiment config: {source}") from error
    return train_config_from_dict(payload)
