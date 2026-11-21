"""Exception hierarchy shared by the CLI, the job worker and the API.

Messages are meant to be shown to users: short, actionable, without tracebacks.
"""

from __future__ import annotations


class NeuroitburdError(Exception):
    """Base class for all expected, user-reportable errors."""


class InvalidImageError(NeuroitburdError):
    """Input image is missing, unreadable, unsupported or has invalid dimensions."""


class InvalidParamsError(NeuroitburdError):
    """Generation parameters are out of range or inconsistent."""
