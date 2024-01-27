# pyright: basic
import argparse
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "destination",
    type=Path,
    nargs="?",
    default="dataset",
    help="Destination directory for data files",
)
args = parser.parse_args()

base_url = "https://huggingface.co/datasets/StonyBrookNLP/tellmewhy/resolve/main/data"
files = ["train.json", "validation.json", "test.json"]

args.destination.mkdir(parents=True, exist_ok=True)

print(f"Downloading data files to '{args.destination.resolve()}'...")
for file in files:
    url = f"{base_url}/{file}"
    print(f"Downloading {url}")
    data = [json.loads(line) for line in urllib.request.urlopen(url)]
    (args.destination / file).write_text(json.dumps(data, indent=2))
