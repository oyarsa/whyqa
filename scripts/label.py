# pyright: basic
import json
from pathlib import Path

import typer
from readchar import readkey


def label_entry() -> bool | None:
    while True:
        print("Valid extraction? (y|space)/n/q: ", end="", flush=True)

        answer = readkey().lower()
        if answer == "q":
            return None
        if answer not in ["y", "n", " "]:
            print("Invalid answer")
            continue

        valid = answer in ["y", " "]
        print("valid" if valid else "invalid")
        return valid


def main(path: Path) -> None:
    data = json.loads(path.read_text())

    for i, item in enumerate(data):
        if item.get("valid") is not None:
            continue

        print(f"[{i + 1}/{len(data)}]")
        print(f'\nStory: {item["narrative"]}')
        print(f'\nQuestion: {item["question"]}')
        print(f'\nAnswer: {item["answer"]}')
        print(f'\nPred: {item["pred"]}')
        print()

        valid = label_entry()
        if valid is None:
            break

        item["valid"] = valid

        print()
        print("-" * 80)
        print()

    path.with_suffix(".labelled.json").write_text(json.dumps(data))


if __name__ == "__main__":
    typer.run(main)
