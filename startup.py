"""
VyomRaksha — startup.py

HuggingFace Spaces startup script.

Runs before the FastAPI server starts. Checks for trained LoRA adapters on the
HuggingFace Hub and sets the LORA_*_PATH env vars so the R2 environment loads
them instead of falling back to the rule-based policy.

Usage (in Dockerfile or app startup):
    python startup.py && uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

_HF_USERNAME = os.getenv("HF_USERNAME", "D3V1601")

# Map from env-var name (read by r2_environment.py) to hub repo name
_LORA_ENV_TO_HUB: dict[str, str] = {
    "LORA_POWER_PATH":          f"{_HF_USERNAME}/VyomRaksha-power-lora",
    "LORA_FUEL_PATH":           f"{_HF_USERNAME}/VyomRaksha-fuel-lora",
    "LORA_THERMAL_PATH":        f"{_HF_USERNAME}/VyomRaksha-thermal-lora",
    "LORA_COMPUTATIONAL_PATH":  f"{_HF_USERNAME}/VyomRaksha-computational-lora",
    "LORA_STRUCTURAL_PATH":     f"{_HF_USERNAME}/VyomRaksha-structural-lora",
    "LORA_COMMUNICATIONS_PATH": f"{_HF_USERNAME}/VyomRaksha-communications-lora",
    "LORA_PROBE_SYSTEMS_PATH":  f"{_HF_USERNAME}/VyomRaksha-probe_systems-lora",
    "LORA_THREAT_PATH":         f"{_HF_USERNAME}/VyomRaksha-threat-lora",
    "LORA_SARVADRISHI_PATH":    f"{_HF_USERNAME}/VyomRaksha-SarvaDrishti-lora",
}

# Local cache directory for downloaded adapters
_CACHE_DIR = Path(os.getenv("LORA_CACHE_DIR", "/tmp/vyomraksha_loras"))


def _download_adapter(env_var: str, hub_repo: str) -> None:
    """
    Download LoRA adapter from hub to local cache and set the env var.

    Skips silently if the repo doesn't exist — rule-based fallback activates.
    """
    if os.getenv(env_var):
        log.info("%s already set to %s — skipping hub check", env_var, os.getenv(env_var))
        return

    try:
        from huggingface_hub import snapshot_download, repo_exists  # type: ignore[import]

        if not repo_exists(hub_repo, repo_type="model"):
            log.info("%s not on hub yet — using rule-based policy", hub_repo)
            return

        local_path = _CACHE_DIR / hub_repo.replace("/", "--")
        if local_path.exists() and any(local_path.iterdir()):
            log.info("Using cached adapter: %s → %s", hub_repo, local_path)
        else:
            log.info("Downloading adapter: %s → %s", hub_repo, local_path)
            snapshot_download(
                repo_id=hub_repo,
                repo_type="model",
                local_dir=str(local_path),
                ignore_patterns=["*.bin", "*.pt"],  # prefer safetensors
            )

        os.environ[env_var] = str(local_path)
        log.info("Set %s = %s", env_var, local_path)

    except ImportError:
        log.warning("huggingface_hub not installed — cannot check hub for adapters")
    except Exception as exc:
        log.warning("Failed to download %s (%s): %s — using rule-based fallback", hub_repo, env_var, exc)


def main() -> None:
    # Always activate R2 mode in the Space
    os.environ["R2_MODE"] = "true"
    log.info("R2_MODE=true")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for env_var, hub_repo in _LORA_ENV_TO_HUB.items():
        _download_adapter(env_var, hub_repo)

    loaded = [k for k in _LORA_ENV_TO_HUB if os.getenv(k)]
    rule_based = [k.replace("LORA_", "").replace("_PATH", "").lower() for k in _LORA_ENV_TO_HUB if not os.getenv(k)]

    log.info("Startup complete — %d LoRA adapters loaded, %d using rule-based policy",
             len(loaded), len(rule_based))
    if rule_based:
        log.info("Rule-based agents: %s", rule_based)

    # Write a startup summary for the Space logs
    summary = {
        "r2_mode": True,
        "adapters_loaded": len(loaded),
        "rule_based_count": len(rule_based),
        "rule_based_agents": rule_based,
    }
    import json
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
