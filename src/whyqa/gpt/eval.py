#!/usr/bin/env python3
"""Run a GPT model on the given data and evaluate the results."""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import no_type_check

import pandas as pd
import typer
from openai import OpenAI
from tqdm import tqdm

from whyqa.gpt.common import (
    MODELS_ALLOWED,
    calculate_cost,
    get_args_,
    get_current_commit_,
    init_client,
    render_args,
)


def run_gpt(
    client: OpenAI, model: str, system_prompt: str, message: str
) -> tuple[str, float]:
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
    return result or "<empty>", cost


SYSTEM_PROMPTS = {
    "simple": """You are a helpful assistant that can evaluate whether an answer is \
correct given a question.""",
}
USER_PROMPTS = {
    "simple": "Based on the story, question and answer, consider the answer is correct \
for the question. Explain your decision.",
    "instructions_score": """\
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
    "instructions_binary": """\
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
}


@no_type_check
def calc_frequencies(results: defaultdict[tuple[bool, int], int]) -> pd.DataFrame:
    """Calculate the frequencies of the results.

    The operations here make the type-checker go crazy, so we disable them.
    """
    df = pd.DataFrame(list(results.items()), columns=["Combination", "Count"])
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)
    return df


@dataclass(frozen=True)
class Entry:
    narrative: str
    question: str
    answer: str
    pred: str
    valid: bool


def main(
    file: Path = typer.Argument(
        ...,
        help="Path to the json file containing the data (list of objects with keys"
        " 'input', 'output', 'gold', 'valid').",
    ),
    output_dir: Path = typer.Option(
        Path("out"),
        help="Path to output directory.",
    ),
    n: int = typer.Option(
        10,
        help="Number of examples to run. Use 0 to run all.",
    ),
    rand: bool = typer.Option(
        True,
        help="Whether to shuffle the data before selecting n examples.",
    ),
    key_file: typer.FileText = typer.Argument(..., help="Path to API key file"),
    key_name: str = typer.Argument(..., help="API key name"),
    model: str = typer.Option(
        "gpt-4",
        help="Which GPT model to use (gpt-3.5-turbo or gpt-4).",
    ),
    system_prompt: str = typer.Option(
        "simple",
        help="Which system prompt to use (only 'simple' for now).",
    ),
    user_prompt: str = typer.Option(
        "instructions",
        help="Which user prompt to use ('simple', 'instructions_score',"
        "'instructions_binary').",
    ),
    print_messages: bool = typer.Option(
        False,
        help="Whether to print the prompt, context, gold and prediction. If false, only"
        " the progress bar and evaluation results are printed.",
    ),
) -> None:
    if model not in MODELS_ALLOWED:
        raise ValueError(f"Invalid model. Options: {MODELS_ALLOWED}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {list(SYSTEM_PROMPTS)}")
    if user_prompt not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {list(USER_PROMPTS)}")

    args = get_args_() | {"commit": get_current_commit_()}
    print(render_args(args))

    client = init_client(key_name, json.load(key_file))

    data = [
        Entry(
            narrative=d["narrative"],
            question=d["question"],
            answer=d["answer"],
            pred=d["pred"],
            valid=d["valid"],
        )
        for d in json.loads(file.read_text())
    ]
    if rand:
        random.shuffle(data)

    n = n or len(data)
    if n % 2 != 0:
        n += 1

    sampled_data = data[:n]

    messages: list[tuple[str, str, bool]] = []
    for item in sampled_data:
        story = f"Story: {item.narrative}"
        question = f"Question: {item.question}"
        pred = f"Answer: {item.pred}"
        gold = f"Gold: {item.answer}"
        valid = item.valid

        display_msg = "\n\n".join([story, question, pred, gold]).strip()
        gpt_msg = "\n\n".join([
            USER_PROMPTS[user_prompt],
            story,
            question,
            pred,
        ]).strip()
        messages.append((display_msg, gpt_msg, valid))

    results: defaultdict[tuple[bool, int], int] = defaultdict(int)
    total_cost = 0

    for display_msg, gpt_msg, valid in tqdm(messages):
        result_s, cost = run_gpt(client, model, SYSTEM_PROMPTS[system_prompt], gpt_msg)
        total_cost += cost

        last_line = result_s.splitlines()[-1].replace("Score:", "").strip()
        result = int(last_line) if last_line.isdigit() else 0
        results[(valid, result)] += 1

        if print_messages:
            print(display_msg)
            print(f"\nGPT: '{result_s}'")
            print("-" * 80)
            print()

    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "config.json").write_text(json.dumps(args, indent=2))

    df = calc_frequencies(results)
    print(df)
    print(f"\nTotal cost: ${total_cost}")


if __name__ == "__main__":
    typer.run(main)
