"""Prints the log the graph baseline script.

Takes a JSON file with an object where the keys are the item IDs and the values are
arrays of object. These objects have the "role" and "data" keys, both strings.
"""

import argparse
import json
import textwrap
from pathlib import Path


def main(file: Path, max_samples: int, max_entries: int) -> None:
    logs: dict[str, list[dict[str, str]]] = json.loads(file.read_text())

    for i, (item_id, log) in enumerate(logs.items(), 1):
        if i > max_samples:
            break

        print("Item:", item_id)
        print("-" * 50)
        print()

        for j, entry in enumerate(log, 1):
            if j > max_entries:
                break

            print(f"Entry {j}:")
            print(f"Role: {entry["role"]}")
            print(f"Data:\n{textwrap.indent(entry["data"], prefix=" " * 4)}")
            print()

        print("-" * 50)
        print()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", type=Path, help="Path to JSON log file")
    parser.add_argument(
        "--max_samples",
        "-n",
        type=int,
        default=int(1e9),
        help="Maximum number of items from the log to print (default: all)",
    )
    parser.add_argument(
        "--max_entries",
        "-k",
        type=int,
        default=int(1e9),
        help="Maximum number of entries from each log to print (default: all)",
    )

    args = parser.parse_args()
    main(args.file, args.max_samples, args.max_entries)
