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
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def process_context(context: str) -> list[str]:
    """Split a context string into a list of individual search results.

    This uses a regex pattern to identify and split the context string into separate
    search results. Each result is expected to have the form:
    '(... title ...) mon dd, yyyy [... content ...]'

    Args:
        context: A string containing multiple search results.

    Returns:
        A list of individual search result strings.
    """
    pattern = r"\([^)]+\)\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}"

    def split_at_matches(text: str, matches: Iterable[re.Match[str]]) -> Iterable[str]:
        last_end = 0
        for match in matches:
            yield text[last_end : match.start()].strip()
            last_end = match.start()
        yield text[last_end:].strip()

    matches = re.finditer(pattern, context)
    return [result for result in split_at_matches(context, matches) if result]
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
