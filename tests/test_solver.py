"""Unit tests for the CLadder estimand solver (causefusion.counterfactual)."""
import pytest

from causefusion.counterfactual import (
    SolveRequest, answer_from_estimand, estimand, solve, solve_backadj,
)


def test_unconfounded_ate_is_difference_of_conditionals():
    # P(Y=1|X=1) - P(Y=1|X=0) = 0.8 - 0.3
    assert estimand("ate", {"p(Y | X)": [0.3, 0.8]}) == pytest.approx(0.5)


def test_backdoor_ate_averages_over_confounder():
    gi = {"p(V1)": 0.5, "p(Y | V1, X)": [[0.2, 0.4], [0.6, 0.8]]}
    # 0.5 * (0.4-0.2) + 0.5 * (0.8-0.6) = 0.2
    assert estimand("ate", gi) == pytest.approx(0.2)


def test_solve_answers_from_question_direction():
    req = SolveRequest(
        query_type="ate",
        question="Will X increase the chance of Y?",
        given_info={"p(Y | X)": [0.3, 0.8]},
    )
    v = solve(req)
    assert (v.answer, v.abstained) == ("yes", False)
    assert v.value == pytest.approx(0.5)


def test_ett_direction_is_flipped():
    # ETT asks about Y had X NOT occurred, so the asserted direction is
    # inverted relative to the sign of the effect (unlike ATE).
    assert answer_from_estimand("ett", "Would Y decrease?", 0.5) == "yes"
    assert answer_from_estimand("ett", "Would Y increase?", 0.5) == "no"
    assert answer_from_estimand("ate", "Would Y increase?", 0.5) == "yes"


def test_abstains_on_unregistered_shape():
    # Confounded ETT is a deliberate abstention: the items lack P(V1 | X=1).
    gi = {"p(V1)": 0.5, "p(Y | V1, X)": [[0.2, 0.4], [0.6, 0.8]]}
    v = solve(SolveRequest("ett", "Will X increase Y?", gi))
    assert v.abstained and v.answer is None
    assert "no registered formula" in v.reason


def test_abstains_on_ambiguous_direction():
    v = solve(SolveRequest("ate", "Is X less likely to increase Y?", {"p(Y | X)": [0.3, 0.8]}))
    assert v.abstained


def test_backadj_picks_the_valid_adjustment_set():
    # In a confounding graph, V1 is the valid set; the empty set is not.
    assert solve_backadj("confounding", [["V1"], []]) == "yes"
    assert solve_backadj("confounding", [[], ["V1"]]) == "no"
    assert solve_backadj("confounding", [["V1"], ["V1"]]) is None  # tie -> abstain
