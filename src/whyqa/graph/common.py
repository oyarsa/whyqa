# pyright: basic
import argparse
import copy
import json
import logging
import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetItem:
    id: str
    query: str
    texts: Sequence[str]
    answer: str


@dataclass(frozen=True)
class OutputItem:
    id: str
    query: str
    texts: Sequence[str]
    expected_answer: str
    generated_answer: str


@dataclass(frozen=True)
class Metrics:
    cosine_similarity: float
    em: float
    f1: float
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float

    def __str__(self) -> str:
        output = ["Dataset metrics:"]
        output.extend(f"  {name}: {value}" for name, value in asdict(self).items())
        return "\n".join(output)


@dataclass(frozen=True)
class ResultItem:
    output: OutputItem
    metrics: Metrics


@dataclass
class APIInteraction:
    role: str
    data: str


def load_dataset(file_path: Path) -> list[DatasetItem]:
    """Load the dataset from a JSON file."""
    data: list[dict[str, Any]] = json.loads(file_path.read_text())
    return [
        DatasetItem(
            id=item["id"],
            query=item["query"],
            texts=item["texts"],
            answer=item["answer"],
        )
        for item in data
    ]


class GPTClient:
    def __init__(self, api_key: str, model: str, seed: int) -> None:
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._log: dict[str, list[APIInteraction]] = defaultdict(list)
        self._seed = seed
        self._input_tokens = 0
        self._output_tokens = 0

    def run(self, item_id: str, prompt: str) -> str:
        """Call the OpenAI API with the given prompt. Returns the response text.

        If the request is successful, returns the response string with leading and
        trailing whitespace removed. If there was an error calling the API, logs the
        error and returns an empty string.

        Logs the interaction per item id (both the user prompt and assistant result)
        on both successful and failed API calls. The log can be obtained from the `log`
        property.
        """
        try:
            self._log[item_id].append(APIInteraction(role="user", data=prompt))
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                seed=self._seed,
            )
            result = response.choices[0].message.content or "<empty>"

            if response.usage:
                self._input_tokens += response.usage.prompt_tokens
                self._output_tokens += response.usage.completion_tokens
        except (openai.OpenAIError, IndexError) as e:
            log.exception("Error calling OpenAI API")
            self._log[item_id].append(
                APIInteraction(role="assistant", data=f"Error: {e}")
            )
            return ""
        else:
            self._log[item_id].append(APIInteraction(role="assistant", data=result))
            return result.strip()

    @property
    def log(self) -> Mapping[str, Sequence[APIInteraction]]:
        return copy.deepcopy(self._log)

    def calc_cost(self) -> float:
        """Calculate the cost spent based on the number of tokens used."""
        model_pricing = {
            "gpt-3.5-turbo": (1, 2),
            "gpt-3.5-turbo-0125": (0.5, 1.5),
            "gpt-4": (30, 60),
            "gpt-4-0613": (30, 60),
            "gpt-4-turbo": (10, 30),
            "gpt-4-turbo-2024-04-09": (10, 30),
            "gpt-4o": (5, 15),
            "gpt-4o-2024-05-13": (5, 15),
            "gpt-4o-2024-08-06": (5, 15),
            "gpt-4o-mini-2024-07-18": (0.15, 0.60),
            "gpt-4o-mini": (0.15, 0.60),
        }

        if self._model not in model_pricing:
            log.warning(f"Pricing for model {self._model} is not available")
            return float("nan")

        input_, output = model_pricing[self._model]
        input_cost = (self._input_tokens / 1e6) * input_
        output_cost = (self._output_tokens / 1e6) * output

        return input_cost + output_cost


def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate the cosine similarity between two texts using sentence embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return (similarity + 1) / 2  # Normalise to [0, 1]


def setup_logging(log: logging.Logger, log_level: str) -> None:
    """Set up logging for the application."""
    if log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid log level: {log_level}")

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(log_level.upper())

    # Suppress useless warnings from transformers
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
    )


def show_elapsed_time(seconds: float) -> str:
    """Render the time as (eg) "4m 25s"."""
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:.0f}m {seconds:.0f}s"


def parse_args(doc: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the JSON file containing the dataset"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Defaults to the OPENAI_API_KEY env var (including .env)",
    )
    parser.add_argument(
        "--senttf-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: %(default)s)",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4o-mini",
        help="OpenAI GPT model name (default: %(default)s)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name to store outputs (default: {GPT model}-{ISO timestamp})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="output",
        help="Path to output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--max-texts",
        "-k",
        type=int,
        default=None,
        help="Maximum number of texts to process per item (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the OpenAI API (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=None, help="Maximum number of samples"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()
