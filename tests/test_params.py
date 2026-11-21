from typing import Any

import pytest

from neuroitburd.errors import InvalidParamsError
from neuroitburd.params import MAX_PROMPT_CHARS, MAX_SEED, GenerationParams


def test_defaults() -> None:
    p = GenerationParams.create()
    assert (p.strength, p.steps, p.guidance) == (0.65, 30, 7.0)
    assert p.seed is None and p.prompt is None and p.negative_prompt is None
    assert p.effective_steps == 19


@pytest.mark.parametrize(
    "values",
    [
        {"strength": 1.0},
        {"strength": 0.01, "steps": 100},
        {"steps": 1, "strength": 1.0},
        {"steps": 100},
        {"guidance": 0.0},
        {"guidance": 20.0},
        {"seed": 0},
        {"seed": MAX_SEED},
        {"prompt": "x" * MAX_PROMPT_CHARS},
    ],
)
def test_valid_bounds(values: dict[str, Any]) -> None:
    GenerationParams.create(**values)


@pytest.mark.parametrize(
    ("values", "field"),
    [
        ({"strength": 0.0}, "strength"),
        ({"strength": 1.01}, "strength"),
        ({"steps": 0}, "steps"),
        ({"steps": 101}, "steps"),
        ({"guidance": -0.1}, "guidance"),
        ({"guidance": 20.1}, "guidance"),
        ({"seed": -1}, "seed"),
        ({"seed": MAX_SEED + 1}, "seed"),
        ({"prompt": "x" * (MAX_PROMPT_CHARS + 1)}, "prompt"),
        ({"negative_prompt": "x" * (MAX_PROMPT_CHARS + 1)}, "negative_prompt"),
        ({"steps": "many"}, "steps"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_invalid_bounds(values: dict[str, Any], field: str) -> None:
    with pytest.raises(InvalidParamsError, match=field):
        GenerationParams.create(**values)


def test_effective_steps_must_be_positive() -> None:
    with pytest.raises(InvalidParamsError, match=r"steps \* strength"):
        GenerationParams.create(steps=3, strength=0.3)


def test_blank_prompts_become_none() -> None:
    p = GenerationParams.create(prompt="   ", negative_prompt="")
    assert p.prompt is None and p.negative_prompt is None
    assert GenerationParams.create(prompt="  a cat ").prompt == "a cat"


def test_numeric_strings_are_coerced() -> None:
    # The API receives multipart form fields as strings.
    p = GenerationParams.create(strength="0.5", steps="20", seed="7")
    assert (p.strength, p.steps, p.seed) == (0.5, 20, 7)


def test_frozen() -> None:
    p = GenerationParams.create()
    with pytest.raises(ValueError):
        p.steps = 10  # type: ignore[misc]
