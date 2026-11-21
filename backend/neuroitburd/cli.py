"""Command-line interface: stylize one image with SDXL img2img.

Example::

    python -m neuroitburd.cli --input examples/input.jpg --output outputs/result.png \\
        --strength 0.65 --steps 30 --guidance 7.0 --seed 42

Exit codes: 0 success, 1 unexpected error, 2 bad arguments or input,
3 model/device/LoRA error, 4 generation or out-of-memory error.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from neuroitburd.config import ConfigError, Settings, setup_logging
from neuroitburd.errors import (
    DeviceUnavailableError,
    GenerationError,
    GpuOutOfMemoryError,
    InvalidImageError,
    InvalidParamsError,
    LoraLoadError,
    LoraNotFoundError,
    ModelLoadError,
    NeuroitburdError,
)
from neuroitburd.image_utils import fit_size, load_image, output_format, save_image
from neuroitburd.params import GenerationParams
from neuroitburd.pipeline import Stylizer, create_stylizer

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_BAD_INPUT = 2
EXIT_MODEL = 3
EXIT_GENERATION = 4

StylizerFactory = Callable[[Settings], Stylizer]

_EXIT_CODES: tuple[tuple[type[NeuroitburdError], int], ...] = (
    (InvalidImageError, EXIT_BAD_INPUT),
    (InvalidParamsError, EXIT_BAD_INPUT),
    (DeviceUnavailableError, EXIT_MODEL),
    (ModelLoadError, EXIT_MODEL),
    (LoraNotFoundError, EXIT_MODEL),
    (LoraLoadError, EXIT_MODEL),
    (GpuOutOfMemoryError, EXIT_GENERATION),
    (GenerationError, EXIT_GENERATION),
)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser. Ranges are checked by ``GenerationParams``, not here."""
    parser = argparse.ArgumentParser(
        prog="neuroitburd",
        description="Stylize an image with SDXL img2img (Neuroitburd).",
        epilog="Exit codes: 2 bad arguments/input, 3 model/device/LoRA error, "
        "4 generation/out-of-memory error.",
    )
    parser.add_argument("--input", required=True, type=Path, help="source image (JPEG/PNG/WebP)")
    parser.add_argument(
        "--output", required=True, type=Path, help="result path (.png, .jpg, .jpeg or .webp)"
    )
    parser.add_argument("--strength", type=float, help="denoising strength, 0-1 (default 0.65)")
    parser.add_argument("--steps", type=int, help="inference steps, 1-100 (default 30)")
    parser.add_argument("--guidance", type=float, help="guidance scale, 0-20 (default 7.0)")
    parser.add_argument("--seed", type=int, help="random seed (random if omitted, then logged)")
    parser.add_argument("--prompt", help="override the default style prompt")
    parser.add_argument("--negative-prompt", help="override the default negative prompt")
    parser.add_argument("--lora", type=Path, help="LoRA adapter .safetensors file")
    parser.add_argument("--lora-weight", type=float, help="LoRA adapter weight, 0-2 (default 0.8)")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), help="compute device (default: auto)"
    )
    parser.add_argument(
        "--offload", choices=("none", "model"), help="CPU offload mode (default: none)"
    )
    return parser


def main(argv: Sequence[str] | None = None, stylizer_factory: StylizerFactory | None = None) -> int:
    """Run the CLI and return the exit code. ``stylizer_factory`` is injectable for tests."""
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        return _run(args, stylizer_factory or create_stylizer)
    except NeuroitburdError as exc:
        return _fail(exc, _exit_code(exc))
    except ConfigError as exc:
        return _fail(exc, EXIT_BAD_INPUT)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.exception("Unexpected error")
        return _fail(exc, EXIT_UNEXPECTED, f"Unexpected error ({type(exc).__name__}); see log.")


def _run(args: argparse.Namespace, stylizer_factory: StylizerFactory) -> int:
    settings = _settings_from_args(args)
    params = GenerationParams.create(
        **_given(
            strength=args.strength,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
        )
    ).with_resolved_seed()

    # Validate everything cheap before the (possibly minutes-long) model load.
    output: Path = args.output.expanduser()
    output_format(output)
    image = load_image(args.input.expanduser())
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidParamsError(f"Cannot create output directory {output.parent}") from exc

    target = fit_size(*image.size)
    logger.info(
        "Params input=%s output=%s source=%dx%d target=%dx%d strength=%s steps=%d "
        "(effective %d) guidance=%s seed=%s prompt=%s negative_prompt=%s lora=%s",
        args.input,
        output,
        image.width,
        image.height,
        target[0],
        target[1],
        params.strength,
        params.steps,
        params.effective_steps,
        params.guidance,
        params.seed,
        _quote(params.prompt or settings.default_prompt),
        _quote(params.negative_prompt or settings.default_negative_prompt),
        f"{settings.lora_path} (weight {settings.lora_weight})" if settings.lora_path else None,
    )

    stylizer = stylizer_factory(settings)
    logger.info("Stylizer %s", stylizer.describe())

    started = time.perf_counter()
    result = stylizer.generate(image, params, on_progress=_progress_printer())
    elapsed = time.perf_counter() - started

    try:
        saved = save_image(result, output)
    except OSError as exc:
        logger.exception("Saving %s failed", output)
        raise GenerationError(f"Could not write output file {output}") from exc

    print(f"Saved {saved} ({result.width}x{result.height}, seed {params.seed}, {elapsed:.1f}s)")
    return EXIT_OK


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Environment settings with CLI flags taking precedence."""
    settings = Settings.from_env()
    overrides: dict[str, Any] = _given(
        device=args.device, offload=args.offload, lora_weight=args.lora_weight
    )
    if args.lora is not None:
        overrides["lora_path"] = args.lora.expanduser().resolve()
    settings = replace(settings, **overrides)  # re-runs Settings validation
    if args.lora_weight is not None and settings.lora_path is None:
        logger.warning("--lora-weight is ignored because no LoRA is set (--lora)")
    return settings


def _given(**values: Any) -> dict[str, Any]:
    """Drop options the user did not pass, so model/settings defaults apply."""
    return {key: value for key, value in values.items() if value is not None}


def _exit_code(exc: NeuroitburdError) -> int:
    for exc_type, code in _EXIT_CODES:
        if isinstance(exc, exc_type):
            return code
    return EXIT_UNEXPECTED


def _fail(exc: Exception, code: int, message: str | None = None) -> int:
    logger.debug("Exiting with code %d", code, exc_info=exc)
    print(f"Error: {message or exc}", file=sys.stderr)
    return code


def _quote(text: str | None) -> str:
    if not text:
        return "''"
    return repr(text if len(text) <= 80 else text[:77] + "...")


def _progress_printer() -> Callable[[int, int], None] | None:
    """Single-line step counter on an interactive terminal; nothing when piped."""
    if not sys.stderr.isatty():
        return None

    def report(step: int, total: int) -> None:
        end = "\n" if step >= total else ""
        print(f"\rStep {step}/{total}", end=end, file=sys.stderr, flush=True)

    return report


if __name__ == "__main__":
    sys.exit(main())
