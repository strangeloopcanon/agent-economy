from __future__ import annotations

from institution_service.judges import JudgeDecision, aggregate_judge_votes
from institution_service.schemas import VerifyStatus


def test_aggregate_judge_votes_empty_is_infra() -> None:
    assert aggregate_judge_votes(decisions=[], min_passes=None) == VerifyStatus.INFRA


def test_aggregate_judge_votes_majority_pass() -> None:
    decisions = [
        JudgeDecision(verdict="PASS", confidence=0.9),
        JudgeDecision(verdict="PASS", confidence=0.7),
        JudgeDecision(verdict="FAIL", confidence=0.6),
    ]
    assert aggregate_judge_votes(decisions=decisions, min_passes=None) == VerifyStatus.PASS


def test_aggregate_judge_votes_majority_fail() -> None:
    decisions = [
        JudgeDecision(verdict="FAIL", confidence=0.9),
        JudgeDecision(verdict="FAIL", confidence=0.7),
        JudgeDecision(verdict="PASS", confidence=0.6),
    ]
    assert aggregate_judge_votes(decisions=decisions, min_passes=None) == VerifyStatus.FAIL


def test_aggregate_judge_votes_tie_is_flake() -> None:
    decisions = [
        JudgeDecision(verdict="PASS", confidence=0.9),
        JudgeDecision(verdict="FAIL", confidence=0.9),
    ]
    assert aggregate_judge_votes(decisions=decisions, min_passes=None) == VerifyStatus.FAIL
