import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import typer
from openai import OpenAI
from tqdm import tqdm

from whyqa.gpt.gpt_eval import calculate_cost


def process_dataset(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    dataset: list[dict[str, Any]],
    print_messages: bool,
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    total_cost = 0

    for item in tqdm(dataset):
        message = "\n\n".join(
            [
                user_prompt,
                f'Narrative: {item["narrative"]}',
                f'Question: {item["question"]}',
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
            {
                "narrative": item["narrative"],
                "question": item["question"],
                "answer": item["answer"],
                "pred": result,
                "model_used": model,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        total_cost += calculate_cost(model, response)

        if print_messages:
            answerable = item["is_ques_answerable_annotator"] == "Answerable"
            print(message)
            print(f"\nAnswerable: {answerable}")
            print(f"\nAnswer: {item['answer']}")
            print(f"\nGPT: '{result}'")
            print()
            print("-" * 80)
            print()

    return results, total_cost


class ClientConfig(TypedDict):
    api_type: str
    key: str


def init_client(api_type: str, config_from_type: dict[str, ClientConfig]) -> OpenAI:
    """Create client for OpenAI API."""
    config = config_from_type[api_type]

    if not api_type.startswith("openai"):
        raise ValueError(f"Unknown API type: {config['api_type']}")

    print("Using OpenAI API")
    return OpenAI(api_key=config["key"])


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
    key_file: Path = typer.Argument(..., help="Path to API key file"),
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

    n = n if n > 0 else len(dataset)
    dataset = dataset[:n]

    client = OpenAI(api_key=key_file.read_text().strip())

    processed_data, total_cost = process_dataset(
        client,
        model,
        SYSTEM_PROMPTS[system_prompt],
        USER_PROMPTS[user_prompt],
        dataset,
        print_messages,
    )
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(processed_data, indent=4))
    print("Total cost:", total_cost)

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
