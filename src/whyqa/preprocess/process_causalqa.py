"""Process JSON files by splitting context strings into lists and adding a count.

This script takes a JSON file path as a command-line argument, processes the file
by splitting the 'context' field of each object into a list of strings based on
a specific pattern, adds a 'num_contexts' field with the count of contexts, and
saves the result to a new JSON file with 'processed' appended to the filename.

The dataset has to be manually downloaded from https://zenodo.org/records/7186761
"""

import argparse
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


def main(input_path: Path) -> None:
    data: list[dict[str, Any]] = json.loads(input_path.read_text())
    output = [
        {
            "id": int(item["id"]),
            "query": item["question"],
            "texts": process_context(item["context"]),
            "answer": item["answer"],
        }
        for item in data
    ]

    output_path = input_path.with_stem(f"{input_path.stem}_processed")
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSON files by splitting context strings."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file")
    args = parser.parse_args()

    main(args.input_file)
