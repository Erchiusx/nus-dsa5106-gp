import re
import string
from typing import Iterable


def compute_accuracy(predictions: Iterable[object], references: Iterable[object]) -> dict[str, float]:
    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"accuracy": 0.0}
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    return {"accuracy": correct / len(predictions)}


def compute_exact_match(
    predictions: Iterable[str],
    references: Iterable[str],
    ignore_case: bool = True,
    ignore_punctuation: bool = False,
) -> dict[str, float]:
    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"exact_match": 0.0}

    normalized_predictions = [_normalize_text(pred, ignore_case, ignore_punctuation) for pred in predictions]
    normalized_references = [_normalize_text(ref, ignore_case, ignore_punctuation) for ref in references]
    correct = sum(
        int(pred == ref)
        for pred, ref in zip(normalized_predictions, normalized_references)
    )
    return {"exact_match": correct / len(predictions)}


def _normalize_text(text: object, ignore_case: bool, ignore_punctuation: bool) -> str:
    normalized = "" if text is None else str(text)
    if ignore_case:
        normalized = normalized.lower()
    if ignore_punctuation:
        normalized = re.sub(f"[{re.escape(string.punctuation)}]", "", normalized)
    return normalized.strip()
