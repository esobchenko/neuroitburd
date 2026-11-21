import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from neuroitburd import cli
from neuroitburd.config import Settings
from neuroitburd.errors import (
    DeviceUnavailableError,
    GenerationError,
    GpuOutOfMemoryError,
    LoraNotFoundError,
    ModelLoadError,
)
from neuroitburd.image_utils import fit_size
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import FakeStylizer, ProgressCallback, Stylizer


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("NEUROITBURD_"):
            monkeypatch.delenv(key)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "input.jpg"
    Image.linear_gradient("L").resize((640, 480)).convert("RGB").save(path)
    return path


class Recorder:
    """Stylizer factory that records what the CLI passed in."""

    def __init__(self, error: Exception | None = None, gen_error: Exception | None = None):
        self.error = error
        self.gen_error = gen_error
        self.settings: Settings | None = None
        self.params: GenerationParams | None = None

    def __call__(self, settings: Settings) -> Stylizer:
        self.settings = settings
        if self.error is not None:
            raise self.error
        return _RecordingStylizer(self)


class _RecordingStylizer(FakeStylizer):
    def __init__(self, recorder: Recorder) -> None:
        super().__init__()
        self._recorder = recorder

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        on_progress: ProgressCallback | None = None,
    ) -> Image.Image:
        self._recorder.params = params
        if self._recorder.gen_error is not None:
            raise self._recorder.gen_error
        return super().generate(image, params, on_progress)


def run(*argv: Any, factory: Recorder | None = None) -> int:
    return cli.main([str(a) for a in argv], stylizer_factory=factory or Recorder())


# --- success paths ------------------------------------------------------------------


def test_generates_output_in_new_directory(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "nested" / "dir" / "result.png"
    rec = Recorder()
    code = run(
        "--input",
        source,
        "--output",
        out,
        "--strength",
        "0.5",
        "--steps",
        "20",
        "--guidance",
        "6",
        "--seed",
        "42",
        factory=rec,
    )

    assert code == 0
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size == fit_size(640, 480)
    assert rec.params == GenerationParams.create(strength=0.5, steps=20, guidance=6, seed=42)
    assert "seed 42" in capsys.readouterr().out


def test_jpeg_output_by_extension(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.jpg"
    assert run("--input", source, "--output", out) == 0
    with Image.open(out) as img:
        assert img.format == "JPEG"


def test_defaults_and_random_seed_logged(
    source: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = Recorder()
    caplog.set_level("INFO", logger="neuroitburd.cli")
    assert run("--input", source, "--output", tmp_path / "r.png", factory=rec) == 0

    assert rec.params is not None and rec.params.seed is not None
    defaults = GenerationParams()
    assert (rec.params.strength, rec.params.steps, rec.params.guidance) == (
        defaults.strength,
        defaults.steps,
        defaults.guidance,
    )
    assert f"seed={rec.params.seed}" in caplog.text
    assert "target=1176x880" in caplog.text
    assert f"seed {rec.params.seed}" in capsys.readouterr().out


def test_prompts_passed_through(source: Path, tmp_path: Path) -> None:
    rec = Recorder()
    code = run(
        "--input",
        source,
        "--output",
        tmp_path / "r.png",
        "--prompt",
        "  cubist  ",
        "--negative-prompt",
        "   ",
        factory=rec,
    )
    assert code == 0
    assert rec.params is not None
    assert rec.params.prompt == "cubist"
    assert rec.params.negative_prompt is None  # blank means "use the default"


def test_cli_flags_override_env(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROITBURD_DEVICE", "cuda")
    monkeypatch.setenv("NEUROITBURD_LORA_WEIGHT", "0.3")
    lora = tmp_path / "style.safetensors"
    rec = Recorder()
    code = run(
        "--input",
        source,
        "--output",
        tmp_path / "r.png",
        "--device",
        "cpu",
        "--offload",
        "model",
        "--lora",
        lora,
        "--lora-weight",
        "1.2",
        factory=rec,
    )

    assert code == 0
    assert rec.settings is not None
    assert rec.settings.device == "cpu"
    assert rec.settings.offload == "model"
    assert rec.settings.lora_path == lora.resolve()
    assert rec.settings.lora_weight == 1.2


def test_env_settings_used_without_flags(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEUROITBURD_OFFLOAD", "model")
    rec = Recorder()
    assert run("--input", source, "--output", tmp_path / "r.png", factory=rec) == 0
    assert rec.settings is not None and rec.settings.offload == "model"


def test_lora_weight_without_lora_warns(
    source: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    code = run("--input", source, "--output", tmp_path / "r.png", "--lora-weight", "0.5")
    assert code == 0
    assert "--lora-weight is ignored" in caplog.text


# --- exit code 2: bad arguments or input ----------------------------------------------


def test_missing_input_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rec = Recorder()
    code = run("--input", tmp_path / "nope.jpg", "--output", tmp_path / "r.png", factory=rec)
    assert code == 2
    assert "Input file not found" in capsys.readouterr().err
    assert rec.settings is None  # model never loaded
    assert not (tmp_path / "r.png").exists()


def test_not_an_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bogus = tmp_path / "fake.png"
    bogus.write_text("hello")
    assert run("--input", bogus, "--output", tmp_path / "r.png") == 2
    assert "not a supported image" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "fragment"),
    [
        (["--strength", "1.5"], "strength"),
        (["--steps", "0"], "steps"),
        (["--steps", "1", "--strength", "0.5"], "steps * strength"),
        (["--guidance", "-1"], "guidance"),
        (["--seed", "-3"], "seed"),
        (["--lora-weight", "3"], "LoRA weight"),
    ],
)
def test_invalid_params(
    source: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    fragment: str,
) -> None:
    rec = Recorder()
    assert run("--input", source, "--output", tmp_path / "r.png", *flags, factory=rec) == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and fragment in err
    assert rec.settings is None


def test_invalid_env_setting(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("NEUROITBURD_DEVICE", "tpu")
    assert run("--input", source, "--output", tmp_path / "r.png") == 2
    assert "NEUROITBURD_DEVICE" in capsys.readouterr().err


def test_bad_output_extension_checked_before_model_load(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder()
    assert run("--input", source, "--output", tmp_path / "r.gif", factory=rec) == 2
    assert "Unsupported output extension" in capsys.readouterr().err
    assert rec.settings is None


def test_uncreatable_output_dir(source: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    rec = Recorder()
    assert run("--input", source, "--output", blocker / "r.png", factory=rec) == 2
    assert rec.settings is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--output", "r.png"],  # missing --input
        ["--input", "a.jpg", "--output", "r.png", "--steps", "many"],
        ["--input", "a.jpg", "--output", "r.png", "--device", "tpu"],
    ],
)
def test_argparse_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv, stylizer_factory=Recorder())
    assert exc_info.value.code == 2


# --- exit codes 3, 4 and 1 ----------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        LoraNotFoundError("LoRA file not found: x.safetensors"),
        DeviceUnavailableError("CUDA requested but not available"),
        ModelLoadError("Could not load model"),
    ],
)
def test_model_errors_exit_3(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    code = run("--input", source, "--output", tmp_path / "r.png", factory=Recorder(error=error))
    assert code == 3
    assert capsys.readouterr().err == f"Error: {error}\n"
    assert not (tmp_path / "r.png").exists()


@pytest.mark.parametrize(
    "error", [GpuOutOfMemoryError("GPU ran out of memory"), GenerationError("failed")]
)
def test_generation_errors_exit_4(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    rec = Recorder(gen_error=error)
    assert run("--input", source, "--output", tmp_path / "r.png", factory=rec) == 4
    assert capsys.readouterr().err == f"Error: {error}\n"
    assert not (tmp_path / "r.png").exists()


def test_unexpected_error_is_short_and_logged(
    source: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    rec = Recorder(gen_error=RuntimeError("/secret/internal/path exploded"))
    assert run("--input", source, "--output", tmp_path / "r.png", factory=rec) == 1
    err = capsys.readouterr().err
    assert err == "Error: Unexpected error (RuntimeError); see log.\n"
    assert "/secret/internal/path exploded" in caplog.text  # full details go to the log


# --- module entry point -------------------------------------------------------------


def test_python_m_entry_point_with_fake_pipeline(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.webp"
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEUROITBURD_")}
    env["NEUROITBURD_FAKE_PIPELINE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "neuroitburd.cli",
            "--input",
            str(source),
            "--output",
            str(out),
            "--seed",
            "7",
            "--steps",
            "4",
            "--strength",
            "0.5",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"Saved {out}" in proc.stdout
    with Image.open(out) as img:
        assert img.format == "WEBP"
        assert img.size == fit_size(640, 480)
