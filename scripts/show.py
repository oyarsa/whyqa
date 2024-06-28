#!/usr/bin/env python3
# pyright: basic
import json
import random
from typing import Optional

import typer

DEFAULT_PROMPT = "Based on the story, answer the question."


def main(
    infile: typer.FileText,
    n: Optional[int] = None,
    rand: bool = False,
    seed: int = 0,
    prompt: str = DEFAULT_PROMPT,
) -> None:
    data = json.load(infile)
    if rand:
        random.seed(seed)
        random.shuffle(data)

    for item in data[:n]:
        story = f"Story: {item['narrative']}"
        question = f"Question: {item['question']}"
        gold = f"Gold: {item['answer']}"
        answerable = f'Answerable: {item["is_ques_answerable_annotator"]}'
        full = "\n\n".join([prompt, story, question, gold, answerable])
        print(full)
        print("-" * 80)
        print()


if __name__ == "__main__":
    app = typer.Typer(
        context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False
    )
    app.command()(main)
    app()
