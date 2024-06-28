"""Use GPT to answer why-questions from the TellMeWhy dataset.

The input is a JSON file with the following format:
- answer (str): The answer to the question
- narrative (str): The story
- question (str): The question asked
- is_ques_answerable_annotator (str): Whether the question is answerable. Can be
  either 'Answerable' or 'Unanswerable'.

See the `dataset` directory for examples.

The key file is a JSON file where the keys are names of keys, and the values are
objects with the following format:
- api_type (str): The type of API to use. Currently only 'openai' is supported.
- key (str): The API key.

Two files are created in the output directory:
- output.json: The results of the GPT model
- metrics.json: The metrics calculated for the results
"""

import inspect
import io
import json
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, TypedDict

import typer
from openai import OpenAI
from tqdm import tqdm

from whyqa import metrics
from whyqa.gpt.eval import calculate_cost


@dataclass(frozen=True)
class Entry:
    """Input entry from the TellMeWhy dataset."""

    answer: str
    narrative: str
    question: str
    answerable: bool


@dataclass(frozen=True)
class Prediction:
    """Single GPT prediction."""

    pred: str
    metrics: metrics.Result


@dataclass(frozen=True)
class Result:
    """Output instance from with all GPT predictions."""

    narrative: str
    question: str
    answer: str
    preds: list[Prediction]
    model_used: str
    timestamp: str
    cost: float

    @property
    def best_pred(self) -> Prediction:
        """Get best prediction based on F1 score."""
        return max(self.preds, key=lambda p: p.metrics.f1)


def render_message(item: Entry, message: str, predictions: list[Prediction]) -> str:
    return "\n".join(
        [
            message,
            f"\nAnswerable: {item.answerable}",
            f"\nAnswer: {item.answer}",
            "\nGPT:",
            *(f"  {i}) {p.pred}" for i, p in enumerate(predictions, start=1)),
            "",
            "-" * 80,
            "",
        ]
    )


def run_answer(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    dataset: list[Entry],
    num_outputs: int,
    temperature: float,
    print_messages: bool,
) -> list[Result]:
    """Run GPT on the dataset.

    Args:
        client: Initialise OpenAI client.
        model: OpenAI model to use (e.g. "gpt-3.5-turbo-0125").
        system_prompt: System prompt to use.
        user_prompt: User prompt to use (filled in, no templates).
        dataset: List of entries to process.
        num_outputs: Number of outputs that the model should generate.
        temperature: Temperature for GPT sampling. If num_outputs > 0, this cannot be 0.
        print_messages: Whether to print messages sent to the API.

    Returns:
        Results for each entry in the dataset.
    """
    assert not (
        num_outputs > 0 and temperature == 0
    ), "Temperature must be > 0 when num_outputs > 0"

    results: list[Result] = []

    for item in tqdm(dataset):
        message = "\n\n".join(
            [
                user_prompt,
                f"Narrative: {item.narrative}",
                f"Question: {item.question}",
            ]
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=0,
            n=num_outputs,
        )

        predictions: list[Prediction] = []
        for output in response.choices:
            result = output.message.content or "<empty>"
            predictions.append(
                Prediction(
                    pred=result,
                    metrics=metrics.calculate_sentence(gold=item.answer, pred=result),
                )
            )

        results.append(
            Result(
                narrative=item.narrative,
                question=item.question,
                answer=item.answer,
                preds=predictions,
                model_used=response.model,
                timestamp=datetime.now(UTC).isoformat(),
                cost=calculate_cost(model, response),
            )
        )

        if print_messages:
            print(render_message(item, message, predictions))

    return results


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


def render_metrics(results: metrics.Result) -> str:
    """Render metrics as a string."""
    return ">>> METRICS\n" + "\n".join(
        f"  {key}: {value:.4f}" for key, value in asdict(results).items()
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


def load_dataset(file: TextIO) -> list[Entry]:
    """Load dataset from (opened) JSON file.

    See module docstring for the JSON format.
    """
    return [
        Entry(
            answer=d["answer"],
            narrative=d["narrative"],
            question=d["question"],
            answerable=d["is_ques_answerable_annotator"] == "Answerable",
        )
        for d in json.load(file)
    ]


def get_args_() -> str | None:
    """Render arguments of the current function.

    It does this by getting the frame of the caller and extracting the arguments.
    Arguments of type `io.TextIOWrapper` are converted to their `name` attribute.

    Returns:
        A rendered string of the arguments, ready to be printed. If there is
        trouble getting the frame, returns None.
    """
    current_frame = inspect.currentframe()
    if current_frame is None:
        return None

    frame = current_frame.f_back
    if frame is None:
        return None

    args, _, _, local_vars = inspect.getargvalues(frame)

    out = [">>> ARGS:"]
    for arg in args:
        value = local_vars[arg]
        if isinstance(value, io.TextIOWrapper):
            value_str = value.name
        else:
            value_str = str(value)
        out.append(f"  {arg}: {value_str}")

    return "\n".join(out) + "\n"


def main(
    file: typer.FileText = typer.Argument(..., help="Input JSON file"),
    output_dir: Path = typer.Argument(..., help="Path to output directory"),
    model: str = typer.Argument(..., help="OpenAI model to use"),
    key_file: typer.FileText = typer.Argument(..., help="Path to API key file"),
    key_name: str = typer.Argument(..., help="API key name"),
    n: int = typer.Option(10, min=1, help="Number of samples to process"),
    rand: bool = typer.Option(False, help="Randomise the dataset"),
    seed: int = typer.Option(0, help="Seed for randomisation"),
    system_prompt: str = typer.Option("simple", help="System prompt to use"),
    user_prompt: str = typer.Option("simple", help="User prompt to use"),
    print_messages: bool = typer.Option(False, help="Print messages sent to API"),
    answerable_only: bool = typer.Option(True, help="Include only aswerable questions"),
    num_outputs: int = typer.Option(1, min=1, help="Number of outputs to generate"),
    temperature: float = typer.Option(0, min=0, max=1, help="Temperature for GPT"),
) -> None:
    print(get_args_())

    dataset = load_dataset(file)
    if answerable_only:
        dataset = [d for d in dataset if d.answerable]
    if rand:
        random.seed(seed)
        random.shuffle(dataset)
    dataset = dataset[:n]

    client = init_client(key_name, json.load(key_file))

    if num_outputs > 0 and temperature == 0:
        raise ValueError("Temperature must be greater than 0 when num_outputs > 0")

    data_answered = run_answer(
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPTS[system_prompt],
        user_prompt=USER_PROMPTS[user_prompt],
        dataset=dataset,
        num_outputs=num_outputs,
        temperature=temperature,
        print_messages=print_messages,
    )

    print("Model used:", data_answered[0].model_used)
    print("Total cost:", sum(r.cost for r in data_answered))

    metric_result = metrics.calculate_dataset(
        [metrics.Instance(gold=d.answer, pred=d.best_pred.pred) for d in data_answered]
    )
    print(render_metrics(metric_result))

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "output.json").write_text(
        json.dumps([asdict(d) for d in data_answered], indent=2)
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(metric_result), indent=2)
    )
    print("\nFiles saved to:", output_dir)

    with (output_dir / "cost.csv").open("a") as f:
        ts = datetime.now(UTC).isoformat()
        f.write(f"{ts},{sum(r.cost for r in data_answered)}\n")


if __name__ == "__main__":
    app = typer.Typer(
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )
    main.__doc__ = __doc__
    app.command()(main)
    app()
