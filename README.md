# WhyQA

This repository tackles the problem of WhyQA using the TellMeWhy dataset. You can find
TellMeWhy:

- GitHub: https://github.com/StonyBrookNLP/tellmewhy
- JSON files: https://drive.google.com/file/d/13eZfq0PuvQug7A25OnyRbfjf9eYuGW5s/view
- Hugging Face Datasets: https://huggingface.co/datasets/StonyBrookNLP/tellmewhy

Either way, you want these three files (based on the Google Drive link):
- `test_full.json`: test set (`test.json` on Hugging Face Datasets)
- `train.json`: train set
- `val.json`: validation set (`validation.json` on Hugging Face Datasets)

## Getting started

This repository is tested with Python 3.11. We use [Poetry](https://python-poetry.org/)
to manage dependencies. To set up the project, run:

```sh
poetry install  # set up virtualenv and install dependencies
poetry shell  # activate virtualenv
```

