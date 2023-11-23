# pyright: basic
import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.metrics import confusion_matrix


def main(path: Path) -> None:
    data = json.loads(path.read_text())

    annotator_answerables: list[str] = [
        item["is_ques_answerable_annotator"] for item in data
    ]
    majority_answerables: list[str] = [item["is_ques_answerable"] for item in data]

    labels = sorted(set(annotator_answerables + majority_answerables))

    conf_matrix = confusion_matrix(
        annotator_answerables, majority_answerables, labels=labels
    )

    index_labels = [f"Annotator: {label}" for label in labels]
    column_labels = [f"Majority: {label}" for label in labels]
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=index_labels, columns=column_labels
    )
    normalized_matrix_df = (conf_matrix_df / len(data) * 100).round(2)

    print(conf_matrix_df)
    print()
    print(normalized_matrix_df)


if __name__ == "__main__":
    typer.run(main)
