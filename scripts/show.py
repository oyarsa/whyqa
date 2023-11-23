# pyright: basic
import json
import random
from pathlib import Path
from typing import Optional

import typer


def main(infile: Path, n: Optional[int] = None, rand: bool = False) -> None:
    prompt = "Based on the story, answer the question."

    data = json.loads(infile.read_text())
    if rand:
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
    typer.run(main)
