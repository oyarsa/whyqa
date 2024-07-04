from collections.abc import Mapping

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
