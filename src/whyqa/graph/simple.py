# pyright: basic
"""Answer a WhyQA dataset using OpenAI API.

This script loads a dataset from a JSON file, generates answers using the OpenAI API
by providing the question and supporting texts directly in the prompt, and evaluates
the answers using sentence embeddings.

The dataset is a JSON file containing a list of items, each with the following keys:
- id (str): A unique identifier for the item.
- query (str): The question to answer.
- texts (list[str]): A list of texts to use as context for answering the question.
- answer (str): The expected answer to the question.

The output directory contains a subdirectory for each run, with the following files:
- result.json: A JSON file containing the results.
- config.json: A JSON file containing the configuration used for the run.
- log.json: A JSON file containing logs of all interactions with the OpenAI API.
- metrics.json: A JSON file containing the overall metrics for the dataset.

The `result.json` file contains a list of items, each with the following keys:
- id (str): The item's unique identifier.
- query (str): The question.
- texts (list[str]): The list of texts used as context.
- expected_answer (str): The expected answer.
- generated_answer (str): The generated answer.
- metrics (dict): Metrics for this specific item.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dotenv
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from whyqa import metrics as metrics_

log = logging.getLogger(__file__)


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


@dataclass
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
        self._log: dict[str, list[APIInteraction]] = {}
        self._seed = seed
        self._input_tokens = 0
        self._output_tokens = 0

    def run(self, item_id: str, prompt: str) -> str:
        """Call the OpenAI API with the given prompt. Returns the response text."""
        try:
            self._log.setdefault(item_id, []).append(
                APIInteraction(role="user", data=prompt)
            )
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
        except Exception as e:
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
        return self._log.copy()

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


def answer_question(
    client: GPTClient, item_id: str, query: str, texts: Sequence[str]
) -> str:
    """Generate an answer to the question using the provided texts."""
    context = "\n\n".join(texts)
    prompt = f"""Given the following context, answer the question.
Make sure to consider the information in the context when generating the answer.
The answer must be as concise as possible and should not contain any additional text.
Ensure that the answer only contains the relevant information and no additional context.

Context:
{context}

Question:
{query}

Answer:"""
    response = client.run(item_id, prompt)
    answer = response.lstrip("Answer:").strip()

    if not answer:
        log.warning(f"Failed to generate answer for question: {query}")
        answer = "Unable to generate answer"

    return answer


def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Calculate the cosine similarity between two texts using sentence embeddings."""
    embeddings = model.encode([text1, text2])
    similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    return (similarity + 1) / 2  # Normalise to [0, 1]


def main(
    dataset_path: Path,
    api_key: str | None,
    senttf_model_name: str,
    gpt_model_name: str,
    output_path: Path,
    run_name: str | None,
    max_texts: int | None,
    seed: int,
    max_samples: int | None,
    log_level: str,
) -> None:
    if log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid log level: {log_level}")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(log_level.upper())

    # Suppress useless warnings from transformers
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
    )

    dotenv.load_dotenv()
    if api_key is None:
        api_key = os.environ["OPENAI_API_KEY"]
    if not run_name:
        run_name = f"{gpt_model_name}-{datetime.now(UTC).isoformat()}"

    client = GPTClient(api_key, gpt_model_name, seed)
    dataset = load_dataset(dataset_path)
    senttf_model = SentenceTransformer(senttf_model_name)

    # Save the configuration for reproducibility
    config = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "senttf_model": senttf_model_name,
        "gpt_model": gpt_model_name,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "max_texts": max_texts,
        "max_samples": max_samples,
        "seed": seed,
    }
    log.info(
        "Configuration:\n" + "\n".join(f"  {k}: {v}" for k, v in config.items()) + "\n"
    )

    dataset = dataset[:max_samples]
    output_items: list[OutputItem] = []

    for i, item in enumerate(tqdm(dataset), 1):
        log.debug(f"Item {i}/{len(dataset)}:")

        texts = item.texts[:max_texts]
        log.debug(f"  Generating answer using {len(texts)} texts.")
        predicted_answer = answer_question(client, item.id, item.query, texts)

        output_items.append(
            OutputItem(
                id=item.id,
                query=item.query,
                texts=texts,
                expected_answer=item.answer,
                generated_answer=predicted_answer,
            )
        )

        log.debug(f"Query: {item.query}")
        log.debug(f"Predicted Answer: {predicted_answer}")
        log.debug(f"Expected Answer: {item.answer}")

    results: list[ResultItem] = []
    for output in output_items:
        cosine = calculate_similarity(
            output.expected_answer, output.generated_answer, senttf_model
        )
        metrics = metrics_.calculate_sentence(
            output.expected_answer, output.generated_answer
        )
        results.append(
            ResultItem(
                output,
                Metrics(
                    cosine_similarity=cosine,
                    em=metrics.em,
                    f1=metrics.f1,
                    rouge_l_precision=metrics.rouge_l_precision,
                    rouge_l_recall=metrics.rouge_l_recall,
                    rouge_l_f1=metrics.rouge_l_f1,
                ),
            )
        )

    dataset_metrics = metrics_.calculate_dataset([
        metrics_.Instance(result.output.expected_answer, result.output.generated_answer)
        for result in results
    ])
    avg_similarity = sum(result.metrics.cosine_similarity for result in results) / len(
        results
    )
    final_metrics = Metrics(
        cosine_similarity=avg_similarity,
        em=dataset_metrics.em,
        f1=dataset_metrics.f1,
        rouge_l_precision=dataset_metrics.rouge_l_precision,
        rouge_l_recall=dataset_metrics.rouge_l_recall,
        rouge_l_f1=dataset_metrics.rouge_l_f1,
    )
    log.info(final_metrics)

    log.info(f"Total Cost: ${client.calc_cost()}")

    output_dir = output_path / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "result.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2)
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(final_metrics), indent=2)
    )

    log_serialisable = {
        item_id: [asdict(interaction) for interaction in interactions]
        for item_id, interactions in client.log.items()
    }
    (output_dir / "log.json").write_text(json.dumps(log_serialisable, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    args = parser.parse_args()

    main(
        args.dataset_path,
        args.api_key,
        args.senttf_model,
        args.gpt_model,
        args.output_dir,
        args.run_name,
        args.max_texts,
        args.seed,
        args.max_samples,
        args.log_level,
    )
