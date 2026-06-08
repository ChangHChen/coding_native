from __future__ import annotations

from typing import Any


LEGACY_UNSCORED_SCORE = -1.0e9
LEGACY_UNSCORED_THRESHOLD = -1.0e8


def score_or_none(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if score <= LEGACY_UNSCORED_THRESHOLD:
        return None
    return score


def prediction_score(prediction: dict[str, Any] | None, score_mode: str) -> float | None:
    if prediction is None:
        return None
    scores = prediction.get("scores", {})
    if not isinstance(scores, dict):
        return None
    return score_or_none(scores.get(score_mode))
