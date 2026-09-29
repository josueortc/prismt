"""Errors that a user can cause, each with a stable code, a plain-language title and a hint.

Every problem caused by the data, the configuration or the computer is raised as a
:class:`PrismtError`. The command line turns it into one JSON object (for MATLAB) and an
exit code, and MATLAB shows the title, message and hint as they are, so write all three
for someone who has never seen a stack trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USER = 2
EXIT_CANCELLED = 3
EXIT_ENVIRONMENT = 4


class PrismtError(Exception):
    """Base class: something the user can fix."""

    exit_code = EXIT_USER
    default_title = "PRISMT could not continue"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str = "",
        field: str = "",
        title: str = "",
        details: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.field = field
        self.title = title or self.default_title
        self.details = details or []

    def to_dict(self) -> dict:
        out = {
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "hint": self.hint,
            "field": self.field,
        }
        if self.details:
            out["details"] = self.details
        return out

    def __str__(self) -> str:
        return f"{self.message} {self.hint}".strip()


class DatasetError(PrismtError):
    default_title = "There is a problem with the dataset file"


class ConfigError(PrismtError):
    default_title = "There is a problem with the run settings"


class SplitError(PrismtError):
    default_title = "The trials cannot be split safely"


class CheckpointError(PrismtError):
    default_title = "A saved model could not be loaded"


class TrainingDiverged(PrismtError):
    default_title = "Training became unstable"


class Cancelled(PrismtError):
    exit_code = EXIT_CANCELLED
    default_title = "The run was stopped"


class EnvironmentProblem(PrismtError):
    exit_code = EXIT_ENVIRONMENT
    default_title = "There is a problem with the Python environment"


@dataclass
class Issue:
    """A finding from validation or a preflight check. ``level`` is error, warning or info."""

    level: str
    code: str
    message: str
    hint: str = ""
    field: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def raise_if_errors(issues: list[Issue], error_type: type[PrismtError], title: str = "") -> None:
    """Raise ``error_type`` built from the first error, listing all errors in ``details``."""
    errors = [i for i in issues if i.level == "error"]
    if not errors:
        return
    first = errors[0]
    message = first.message
    if len(errors) > 1:
        message += f" ({len(errors) - 1} more problem{'s' if len(errors) > 2 else ''} listed below.)"
    raise error_type(
        first.code,
        message,
        hint=first.hint,
        field=first.field,
        title=title,
        details=[e.to_dict() for e in errors],
    )
