import inspect
import io
import subprocess
from collections.abc import Mapping
from typing import TypedDict

from openai import OpenAI
from openai.types.chat import ChatCompletion


def _cm(input: float, output: float) -> tuple[float, float]:
    """Converts costs per 1M tokens to costs per token."""
    return (input / 1e6, output / 1e6)


_MODEL_COSTS: Mapping[str, tuple[float, float]] = {
    "gpt-3.5-turbo": _cm(1, 2),
    "gpt-4": _cm(30, 60),
    "gpt-4o-2024-05-13": _cm(5, 15),
    "gpt-3.5-turbo-0125": _cm(0.5, 1.5),
}
"""Costs per token."""

MODELS_ALLOWED = list(_MODEL_COSTS.keys())
"""Allowed models."""


def calculate_cost(model: str, response: ChatCompletion) -> float:
    """Cost of a completion response combining input and output tokens.

    If usage information is not available, returns NaN.
    """
    usage = response.usage
    if usage is None:
        return float("nan")

    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    input_cost, output_cost = _MODEL_COSTS[model]
    return input_tokens * input_cost + output_tokens * output_cost


class ClientConfig(TypedDict):
    api_type: str
    key: str


def init_client(api_type: str, config_from_type: dict[str, ClientConfig]) -> OpenAI:
    """Create client for OpenAI API from the config file."""
    config = config_from_type[api_type]

    if not api_type.startswith("openai"):
        raise ValueError(f"Unknown API type: {config["api_type"]}")

    print("Using OpenAI API")
    return OpenAI(api_key=config["key"])


def get_args_() -> dict[str, str]:
    """Get the arguments of the caller function.

    It does this by getting the frame of the caller and extracting the arguments.
    Values of type `io.TextIOWrapper` are converted to their `name` attribute.

    Note: impure function - reads the stack frame.

    Returns:
        Mapping of argument names to their string values.
    """
    current_frame = inspect.currentframe()
    if current_frame is None:
        return {}

    frame = current_frame.f_back
    if frame is None:
        return {}

    args, _, _, local_vars = inspect.getargvalues(frame)

    result: dict[str, str] = {}
    for arg in args:
        value = local_vars[arg]
        result[arg] = value.name if isinstance(value, io.TextIOWrapper) else str(value)

    return result


def get_current_commit_() -> str:
    """Get the current commit hash, with "(dirty)" if the repo is dirty.

    Note: impure function - runs shell commands.
    Returns "unknown" if the git commands fail.
    """
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        # Output is non-empty if there are changes (dirty)
        is_dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"
    else:
        if is_dirty:
            git_hash += " (dirty)"
        return git_hash


def render_args(args: dict[str, str]) -> str:
    """Render the arguments as a string."""
    return (
        ">>> CONFIG\n"
        + "\n".join(f"  {key}: {value}" for key, value in args.items())
        + "\n"
    )
