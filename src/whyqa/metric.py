"""Calculate metrics for WhyQA"""

import re
import string
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    gold: str
    pred: str


def _get_tokens(s: str) -> list[str]:
    """Lower text, remove punctuation, articles and split by whitespace."""
    s = s.casefold()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove common articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def calculate(instances: list[Instance]) -> dict[str, float]:
    """Calculate token-level precision, recall and F1 scores, and Exact Match."""
    gold_len = 0
    pred_len = 0
    common_tokens = 0
    equal_count = 0

    for instance in instances:
        pred_toks = _get_tokens(instance.pred)
        gold_toks = _get_tokens(instance.gold)

        gold_len += len(gold_toks)
        pred_len += len(pred_toks)

        common = Counter(gold_toks) & Counter(pred_toks)
        common_tokens += sum(common.values())

        equal_count += int(gold_toks == pred_toks)

    precision = common_tokens / pred_len if pred_len != 0 else 0
    recall = common_tokens / gold_len if gold_len != 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    em = equal_count / len(instances)

    return {"precision": precision, "recall": recall, "f1": f1, "em": em}
