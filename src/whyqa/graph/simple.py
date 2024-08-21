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
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from whyqa import metrics as metrics_
from whyqa.graph.common import (
    GPTClient,
    Metrics,
    OutputItem,
    ResultItem,
    calculate_similarity,
    load_dataset,
    parse_args,
    show_elapsed_time,
    setup_logging,
)

log = logging.getLogger(__file__)


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
    setup_logging(log, log_level)

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

    start_time = time.perf_counter()

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

    elapsed_time_s = time.perf_counter() - start_time
    config["elapsed_time"] = elapsed_time_s

    log.info(f"Elapsed time: {show_elapsed_time(elapsed_time_s)}")

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
    args = parse_args(__doc__)
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
