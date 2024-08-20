"""Calcuate metrics for WhyQA.

Mostly copied from CausalQA's evaluation script:
https://github.com/andreaschandra/CausalQA/blob/c25bc80f7c68709e7ae87739195168da016a243e/finetuning/measures.py#L45

With some refactoring and modifications to fit the other parts of the project.
"""

import logging
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import no_type_check

from rouge_score import rouge_scorer, scoring  # type: ignore


@contextmanager
def _disable_logging() -> Iterator[None]:
    """Temporarily disable logging.

    This is useful when using external libraries that log to the root logger (*cough*
    rouge_score *cough*).

    Any log messages that are generated during the context will be discarded, including
    our own, unfortunately.
    """
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers.copy()

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
    f1: float
    em: float
    rouge_l_precision: float
    rouge_l_recall: float
    rouge_l_f1: float


@dataclass(frozen=True)
class RougeResult:
    precision: float
    recall: float
    f1: float


def _preprocess(s: str) -> str:
    """Preprocess the input string, lowercasing, removing punctuation and articles."""
    s = s.casefold()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth.

    Follows a standard bag-of-words F1 score calculation with the text being tokenized
    by whitespace.
    """
    prediction_tokens = pred.split()
    ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)

    equal = sum(common.values())
    if equal == 0:
        return 0

    precision = equal / len(prediction_tokens)
    recall = equal / len(ground_truth_tokens)

    return (2 * precision * recall) / (precision + recall)


def _em(pred: str, ground_truth: str) -> int:
    """Calculate exact match between prediction and ground truth."""
    return int(pred == ground_truth)


def _calculate_measures(
    measure: Callable[[str, str], float],
    predictions: Iterable[str],
    ground_truths: Iterable[Iterable[str]],
) -> float:
    """Calculate measures for predictions and ground truths using a measure function."""
    result = 0

    for pred, ground_truth in zip(predictions, ground_truths):
        value = max(measure(pred, answer) for answer in ground_truth)
        result += value

    return result / len(list(predictions))


@no_type_check
def _rouge_l(
    predictions: Iterable[str], ground_truths: Iterable[Iterable[str]]
) -> RougeResult:
    """Calculate ROUGE-L scores for predictions and ground truths."""

    with _disable_logging():
        scorer = rouge_scorer.RougeScorer(["rougeL"])
        aggregator = scoring.BootstrapAggregator()

        for pred, gts in zip(predictions, ground_truths):
            score = scorer.score_multi(gts, pred)
            aggregator.add_scores(score)

        results = aggregator.aggregate()

    return RougeResult(
        precision=results["rougeL"].mid.precision,
        recall=results["rougeL"].mid.recall,
        f1=results["rougeL"].mid.fmeasure,
    )


def calculate_dataset(instances: list[Instance]) -> Result:
    """Calculate dataset-level metrics: token F1, exact match and ROUGE-L."""
    predictions = [_preprocess(instance.pred) for instance in instances]
    ground_truths = [[_preprocess(instance.gold)] for instance in instances]

    rougel = _rouge_l(predictions, ground_truths)
    f1 = _calculate_measures(_f1, predictions, ground_truths)
    em = _calculate_measures(_em, predictions, ground_truths)

    return Result(
        f1=f1,
        em=em,
        rouge_l_precision=rougel.precision,
        rouge_l_recall=rougel.recall,
        rouge_l_f1=rougel.f1,
    )


def calculate_sentence(gold: str, pred: str) -> Result:
    """Calculate sentence-level metrics: token F1, exact match and ROUGE-L."""
    return calculate_dataset([Instance(gold=gold, pred=pred)])
