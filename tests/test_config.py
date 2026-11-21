import os
from pathlib import Path

import pytest

from neuroitburd.config import DEFAULT_PROMPT, ConfigError, Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("NEUROITBURD_"):
            monkeypatch.delenv(key)
    s = Settings.from_env()
    assert s.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert s.device == "auto"
    assert s.lora_path is None
    assert s.lora_weight == 0.8
    assert s.default_prompt == DEFAULT_PROMPT
    assert s.output_dir.is_absolute()
    assert s.fake_pipeline is False


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEUROITBURD_DEVICE", "CUDA")
    monkeypatch.setenv("NEUROITBURD_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("NEUROITBURD_LORA_PATH", str(tmp_path / "x.safetensors"))
    monkeypatch.setenv("NEUROITBURD_LORA_WEIGHT", "0.5")
    monkeypatch.setenv("NEUROITBURD_FAKE_PIPELINE", "true")
    s = Settings.from_env()
    assert s.device == "cuda"
    assert s.output_dir == tmp_path / "out"
    assert s.lora_path == tmp_path / "x.safetensors"
    assert s.lora_weight == 0.5
    assert s.fake_pipeline is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DEVICE", "tpu"),
        ("OFFLOAD", "sequential"),
        ("LORA_WEIGHT", "abc"),
        ("LORA_WEIGHT", "2.5"),
        ("LORA_WEIGHT", "-0.1"),
        ("MAX_UPLOAD_MB", "0"),
        ("MAX_QUEUE_SIZE", "0"),
        ("FAKE_PIPELINE", "x"),
    ],
)
def test_invalid_values(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    monkeypatch.setenv(f"NEUROITBURD_{key}", value)
    with pytest.raises(ConfigError):
        Settings.from_env()
