import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import typer
from openai import OpenAI
from tqdm import tqdm

from whyqa import metric
from whyqa.gpt.eval import calculate_cost


@dataclass(frozen=True)
class Result:
    """Output instance from the GPT answer."""

    narrative: str
    question: str
    answer: str
    pred: str
    model_used: str
    timestamp: str
    cost: float


def run_answer(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    dataset: list[dict[str, Any]],
    print_messages: bool,
) -> list[Result]:
    results: list[Result] = []

    for item in tqdm(dataset):
        message = "\n\n".join(
            [
                user_prompt,
                f"Narrative: {item["narrative"]}",
                f"Question: {item["question"]}",
            ]
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=0,
        )

        result = response.choices[0].message.content
        results.append(
            Result(
                narrative=item["narrative"],
                question=item["question"],
                answer=item["answer"],
                pred=result or "<empty>",
                model_used=model,
                timestamp=datetime.now(UTC).isoformat(),
                cost=calculate_cost(model, response),
            )
        )

        if print_messages:
            answerable = item["is_ques_answerable_annotator"] == "Answerable"
            print(message)
            print(f"\nAnswerable: {answerable}")
            print(f"\nAnswer: {item["answer"]}")
            print(f"\nGPT: '{result}'")
            print()
            print("-" * 80)
            print()

    return results


class ClientConfig(TypedDict):
    api_type: str
    key: str


def init_client(api_type: str, config_from_type: dict[str, ClientConfig]) -> OpenAI:
    """Create client for OpenAI API."""
    config = config_from_type[api_type]

    if not api_type.startswith("openai"):
        raise ValueError(f"Unknown API type: {config["api_type"]}")

    print("Using OpenAI API")
    return OpenAI(api_key=config["key"])


def render_metrics(results: dict[str, float]) -> str:
    """Render metrics as a string."""
    return ">>> METRICS\n" + "\n".join(
        f"  {key}: {value:.4f}" for key, value in results.items()
    )


SYSTEM_PROMPTS = {
    "simple": """You are a helpful assistant that can answer questions about why \
things happen."""
}
USER_PROMPTS = {
    "simple": """Based on the story, answer the question. The answer might be implicit \
in the text. The response should be just the answer, nothing else.""",
    "instructions": """\
""",
}


def main(
    file: typer.FileText = typer.Argument(..., help="Input JSON file"),
    output: Path = typer.Argument(..., help="Path to output JSON file"),
    model: str = typer.Argument(..., help="Model to use"),
    key_file: typer.FileText = typer.Argument(..., help="Path to API key file"),
    key_name: str = typer.Argument(..., help="API key name"),
    n: int = typer.Option(10, help="Number of samples to process"),
    rand: bool = typer.Option(False, help="Randomise the dataset"),
    system_prompt: str = typer.Option("simple", help="System prompt to use"),
    user_prompt: str = typer.Option("simple", help="User prompt to use"),
    print_messages: bool = typer.Option(False, help="Print messages sent to API"),
    include_unanswerable: bool = typer.Option(
        False, help="Include unanswerable questions"
    ),
) -> None:
    dataset = json.load(file)
    if not include_unanswerable:
        dataset = [
            d for d in dataset if d["is_ques_answerable_annotator"] == "Answerable"
        ]
    if rand:
        random.shuffle(dataset)

    if n <= 0:
        raise ValueError("Number of samples must be greater than 0")
    dataset = dataset[:n]

    client = init_client(key_name, json.load(key_file))
    data_answered = run_answer(
        client,
        model,
        SYSTEM_PROMPTS[system_prompt],
        USER_PROMPTS[user_prompt],
        dataset,
        print_messages,
    )

    total_cost = sum(r.cost for r in data_answered)
    print("Total cost:", total_cost)

    metric_result = metric.calculate(
        [metric.Instance(gold=d.answer, pred=d.pred) for d in data_answered]
    )
    print(render_metrics(metric_result))

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps([asdict(d) for d in data_answered], indent=2))
    output.with_stem(f"{output.stem}_metrics").write_text(
        json.dumps(metric_result, indent=2)
    )

    with (output.parent / "cost.csv").open("a") as f:
        ts = datetime.now(UTC).isoformat()
        f.write(f"{ts},{total_cost}\n")


if __name__ == "__main__":
    app = typer.Typer(
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=False,
    )
    app.command()(main)
    app()
