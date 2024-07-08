#!/usr/bin/env python3
"""Run a GPT model on the given data and evaluate the results."""

import abc
import json
import random
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Optional, cast, no_type_check, override

import pandas as pd
import typer
from openai import OpenAI
from scipy.stats import spearmanr  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,  # type: ignore
    cohen_kappa_score,  # type: ignore
    precision_recall_fscore_support,  # type: ignore
)
from tqdm import tqdm

from whyqa.gpt.common import (
    MODELS_ALLOWED,
    calculate_cost,
    get_args_,
    get_current_commit_,
    init_client,
    render_args,
)


def run_gpt_(
    client: OpenAI, model: str, system_prompt: str, message: str
) -> tuple[str, float, str]:
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
    cost = calculate_cost(model, response)
    model = response.model
    return result or "<empty>", cost, model


@dataclass(frozen=True)
class Entry:
    narrative: str
    question: str
    answer: str
    pred: str
    human_eval: bool


@dataclass(frozen=True)
class Result(Entry):
    model_eval: bool


class ResultMode(abc.ABC):
    @abc.abstractmethod
    def parse_line(self, line: str) -> int: ...
    @abc.abstractmethod
    def to_binary(self, value: int) -> bool: ...


@dataclass(frozen=True)
class ScoreMode(ResultMode):
    threshold: int

    @override
    def parse_line(self, line: str) -> int:
        return int(line.lower().replace("score:", "").strip())

    @override
    def to_binary(self, value: int) -> bool:
        return value >= self.threshold


class BinaryMode(ResultMode):
    @override
    def parse_line(self, line: str) -> int:
        return int(line.lower().replace("result:", "").strip())

    @override
    def to_binary(self, value: int) -> bool:
        if value not in (0, 1):
            raise ValueError(f"Invalid binary value: {value}")
        return bool(value)


class ResultModeType(StrEnum):
    SCORE = "score"
    BINARY = "binary"

    def new(self, score_threshold: int | None) -> ResultMode:
        if self is self.BINARY:
            return BinaryMode()

        if score_threshold is None:
            raise ValueError("Threshold required for score mode")
        return ScoreMode(score_threshold)


SYSTEM_PROMPTS = {
    "simple": """You are a helpful assistant that can evaluate whether an answer is \
correct given a question.""",
}
USER_PROMPTS = {
    "simple": (
        "Based on the story, question and answer, consider the answer is correct \
for the question. Explain your decision.",
        ResultModeType.SCORE,
    ),
    "instructions_score": (
        """\
Based on the story, question and answer, determine if the answer is correct for the \
question. Explain your decision.

Evaluate the answer according to the following criteria:

1. Read the story, the question and the answer. Check if the answer correctly answers \
the question.
2. Make sure that the answer only contains the necessary information.
3. Assign a score for validity on a scale from 1 to 5, where 1 is the lowest and 5 is \
the highest based on the Evaluation Criteria.
4. The story always provides the reason for the question, but it might be implicit. \
Reason about the text instead of merely extracting spans.

Response format:
Explanation: ...
Score: number from 1 to 5
""",
        ResultModeType.SCORE,
    ),
    "instructions_binary": (
        """\
Based on the story, question and answer, determine if the answer is correct for the \
question. Explain your decision.

Evaluate the answer according to the following criteria:

1. Read the story, the question and the answer. Check if the answer correctly answers \
the question.
2. The story always provides the reason for the question, but it might be implicit.
Reason about the text instead of merely extracting spans.
3. Answer with the result of the evaluation: 1 if the answer is correct, 0 \
otherwise.

Response format:

Explanation: ...
Result: 1 or 0
""",
        ResultModeType.BINARY,
    ),
}


@no_type_check
def calc_frequencies(results: list[Result]) -> pd.DataFrame:
    """Calculate the frequencies of the results (human_eval vs. model_eval).

    The operations here make the type-checker go crazy, so we disable them.
    """
    result_counts = Counter((i.human_eval, i.model_eval) for i in results)

    df = pd.DataFrame(list(result_counts.items()), columns=["Combination", "Count"])
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    return df


def safe_div(a: float, b: float) -> float:
    """Try to divide two numbers, return 0 if the denominator is 0."""
    return a / b if b != 0 else 0


def calc_metrics(results: list[Result]) -> dict[str, float]:
    """Calculate classification metrics, including agreement and correlation.

    For classification, use accuracy and macro-averaged F1, precision and recall. For
    correlation, use Spearman's R, and for agreement, Cohen's Kappa.

    Since the evaluations are majority "valid", a high accuracy doesn't mean much (a
    naive 100% classifier has 82% accuracy). Using a macro-averaged score means both
    classes get equal weight, which is more representative of the result.
    """
    y_true = [r.human_eval for r in results]
    y_pred = [r.model_eval for r in results]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(  # type: ignore
            y_true,
            y_pred,
            average="macro",
            zero_division=0,  # type: ignore
        )
        kappa = cohen_kappa_score(y_true, y_pred)
        spearman = cast(float, spearmanr(y_true, y_pred)[0])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "spearman": spearman,
    }


def main(
    file: Path = typer.Argument(
        ...,
        help="Path to the json file containing the data (list of objects with keys"
        " 'input', 'output', 'gold', 'valid').",
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Path to output directory.",
    ),
    key_file: typer.FileText = typer.Argument(..., help="Path to API key file"),
    key_name: str = typer.Argument(..., help="API key name"),
    n: int = typer.Option(
        10,
        help="Number of examples to run. Use 0 to run all.",
    ),
    rand: bool = typer.Option(
        True,
        help="Whether to shuffle the data before selecting n examples.",
    ),
    model: str = typer.Option(
        "gpt-3.5-turbo",
        help="Which GPT model to use (gpt-3.5-turbo or gpt-4).",
    ),
    system_prompt: str = typer.Option(
        "simple",
        help="Which system prompt to use (only 'simple' for now).",
    ),
    user_prompt_key: str = typer.Option(
        "simple",
        "--user-prompt",
        help="Which user prompt to use ('simple', 'instructions_score',"
        "'instructions_binary').",
    ),
    print_messages: bool = typer.Option(
        False,
        help="Whether to print the prompt, context, gold and prediction. If false, only"
        " the progress bar and evaluation results are printed.",
    ),
    result_threshold: Optional[int] = typer.Option(
        None,
        help="Threshold for the score mode. The model result is valid is the score is"
        " equal or greater than this.",
    ),
) -> None:
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model. Options: {MODELS_ALLOWED}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {list(SYSTEM_PROMPTS)}")
    if user_prompt_key not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {list(USER_PROMPTS)}")

    args = get_args_() | {"commit": get_current_commit_()}
    print(render_args(args))

    client = init_client(key_name, json.load(key_file))
    user_prompt, result_mode_type = USER_PROMPTS[user_prompt_key]
    result_mode = result_mode_type.new(result_threshold)

    data = [
        Entry(
            narrative=d["narrative"],
            question=d["question"],
            answer=d["answer"],
            pred=d["pred"],
            human_eval=d["valid"],
        )
        for d in json.loads(file.read_text())
    ]
    if rand:
        random.shuffle(data)

    n = n or len(data)
    if n % 2 != 0:
        n += 1

    sampled_data = data[:n]

    messages: list[tuple[Entry, str, str]] = []
    for item in sampled_data:
        story = f"Story: {item.narrative}"
        question = f"Question: {item.question}"
        pred = f"Answer: {item.pred}"
        gold = f"Gold: {item.answer}"
        valid_msg = f"Human eval: {item.human_eval}"

        display_msg = "\n\n".join([story, question, pred, gold, valid_msg]).strip()
        gpt_msg = "\n\n".join([
            user_prompt,
            story,
            question,
            pred,
        ]).strip()
        messages.append((item, display_msg, gpt_msg))

    result_data: list[Result] = []
    total_cost = 0
    model_used = None

    for item, display_msg, gpt_msg in tqdm(messages):
        result_s, cost, model = run_gpt_(
            client, model, SYSTEM_PROMPTS[system_prompt], gpt_msg
        )
        total_cost += cost
        model_used = model

        result = result_mode.parse_line(result_s.splitlines()[-1])

        result_data.append(
            Result(**asdict(item), model_eval=result_mode.to_binary(result))
        )

        if print_messages:
            print(display_msg)
            print(f"\nGPT: '{result_s}'")
            print("-" * 80)
            print()

    print(f"\nModel used: {model_used}")
    print(calc_frequencies(result_data))

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "results.json").write_text(
        json.dumps(
            [asdict(d) for d in result_data],
            indent=2,
        )
    )
    (output_dir / "config.json").write_text(json.dumps(args, indent=2))

    classification_metrics = calc_metrics(result_data)
    print("\nClassification metrics:")
    print(json.dumps(classification_metrics, indent=2))
    (output_dir / "classification_metrics.json").write_text(
        json.dumps(classification_metrics, indent=2)
    )

    print(f"\nTotal cost: ${total_cost}")


if __name__ == "__main__":
    app = typer.Typer(
        context_settings={"help_option_names": ["-h", "--help"]},
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )
    app.command(help=__doc__)(main)
    app()
