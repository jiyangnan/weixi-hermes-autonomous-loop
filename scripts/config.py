"""Configuration for Episodic Memory system."""

import os
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"

DEFAULT_CONFIG = {
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "***",
    },
    "vector_store": {
        "dim": 384,
        "index_type": "flat",
    },
    "storage": {
        "base": str(Path.home() / ".hermes" / "autonomous-loop"),
    },
    # Autonomy gates — defaults mirror config.yaml defaults
    "autonomy": {
        "auto_recall": True,
        "max_recall_episodes": 5,
        "auto_episode_cut": False,
        "auto_reflection": False,
        "recall_min_score": 0.5,
    },
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            user = yaml.safe_load(f) or {}
        # Merge defaults
        config = DEFAULT_CONFIG.copy()
        for section, values in user.items():
            if section in config:
                if isinstance(config[section], dict) and isinstance(values, dict):
                    config[section].update(values)
                else:
                    config[section] = values
            else:
                config[section] = values
        return config
    return DEFAULT_CONFIG


def check_autonomy_gate(gate: str) -> bool:
    """
    Check if a specific autonomy gate is enabled.

    Call this before any autonomous action that carries risk.
    Raises KeyError if gate is unknown (fail-closed on unknown gates).

    Usage:
        if check_autonomy_gate("auto_episode_cut"):
            em.cut_episode(...)
    """
    cfg = load_config()
    gates = cfg.get("autonomy", {})
    if gate not in gates:
        # Unknown gate — fail closed (conservative default: require human)
        raise KeyError(f"Unknown autonomy gate: {gate!r}. Known gates: {list(gates.keys())}")
    return gates[gate]
