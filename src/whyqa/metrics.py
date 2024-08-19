"""Calculate metrics for WhyQA"""

import copy
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from contextlib import contextmanager
@contextmanager
def disable_logging():
    """Temporarily disable logging.

    This is useful when using external libraries that log to the root logger (*cough*
    rouge_score *cough*).

    Any log messages that are generated during the context will be discarded, including
    our own, unfortunately.
    """
    root = logging.getLogger()
    original_level = root.level
    original_handlers = copy.deepcopy(root.handlers)

    try:
        yield
    finally:
        # Remove all handlers
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        # Restore original handlers
        for handler in original_handlers:
            root.addHandler(handler)
        # Restore original level
        root.setLevel(original_level)


@dataclass(frozen=True)
class Instance:
    gold: str
    pred: str


@dataclass(frozen=True)
class Result:
    precision: float
    recall: float
    f1: float
    em: float


def _get_tokens(s: str) -> list[str]:
    """Lower text, remove punctuation, articles and split by whitespace."""
    s = s.casefold()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove common articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def calculate_dataset(instances: list[Instance]) -> Result:
    """Calculate dataset token-level precision, recall and F1 scores, and Exact Match."""
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

    return Result(precision=precision, recall=recall, f1=f1, em=em)


def calculate_sentence(gold: str, pred: str) -> Result:
    """Calculate sentence token precision, recall and F1 scores, and Exact Match."""
    gold_toks = _get_tokens(gold)
    pred_toks = _get_tokens(pred)

    pred_len = len(pred_toks)
    gold_len = len(gold_toks)

    common = Counter(gold_toks) & Counter(pred_toks)
    common_tokens = sum(common.values())

    precision = common_tokens / pred_len if pred_len != 0 else 0
    recall = common_tokens / gold_len if gold_len != 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    em = int(gold_toks == pred_toks)

    return Result(precision=precision, recall=recall, f1=f1, em=em)
