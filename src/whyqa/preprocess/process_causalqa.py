# pyright: basic
"""Transform CSV files to JSON by splitting context strings into lists.

This script takes a CSV file path as a commandline argument, processes the file
by splitting the 'context' field of each row into a list of strings and saves the result
to a new JSON file.

The 'context' field in the CSV file contains multiple search results in a single string.
This is broken down into individual search results using a regex pattern.

The CSV file is expected to have the following columns:
- id: An integer ID for each row.
- question: A string containing a search query.
- context: A string containing multiple search results.
- answer: A string containing the correct answer.

The dataset has to be manually downloaded from https://zenodo.org/records/7186761
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def process_context(context: str) -> list[str]:
    """Split a context string into a list of individual search results.

    This function is designed to handle multiple search results in a single string,
    where each result starts with a title in parentheses.

    Args:
        context: A string containing multiple search results.

    Returns:
        A list of individual search result strings.
    """
    # Pattern: (Title) followed by content up to the next title or end of string
    # This isn't perfect (for example, nested parens screw this up), but it's the best
    # that I could do with the unstructed data the datasets have. Blame the CausalQA
    # dataset curators that decided to do this.
    pattern = r"\([^)]+\).*?(?=\([^)]+\)|$)"
    documents = re.findall(pattern, context, re.DOTALL)
    return [doc.strip() for doc in documents]


def main(input_path: Path, output_file: Path) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        output: list[dict[str, Any]] = [
            {
                "id": int(row["id"]),
                "query": row["question"],
                "texts": process_context(row["context"]),
                "answer": row["answer"],
            }
            for row in reader
        ]

    output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CSV files by splitting context strings."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input CSV file")
    parser.add_argument(
        "output_file", type=Path, help="Path to the transformed JSON file"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file)
