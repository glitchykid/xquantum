import pytest

from xquantum.evaluation import evaluate


def test_repeated_views_count_once_in_primary_patient_metrics():
    result = evaluate(
        ["a", "a", "b"], [[0.9, 0.1], [0.7, 0.3], [0.4, 0.6]], ["p1", "p1", "p2"], ["a", "b"]
    )
    assert result["samples"] == 3
    assert result["patients"] == 2
    assert result["patient_level"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["patient_level"]["per_class"]["a"]["sensitivity"] == 1
    assert result["patient_level"]["per_class"]["a"]["specificity"] == 1
    assert result["patient_level"]["confidence_intervals"] is None
    assert result["warnings"]


def test_missing_class_does_not_become_a_perfect_balanced_accuracy():
    result = evaluate(["a", "a"], [[0.9, 0.1], [0.8, 0.2]], ["p1", "p2"], ["a", "b"])
    assert result["balanced_accuracy"] is None
    assert result["per_class"]["b"]["sensitivity"] is None
    assert result["per_class"]["a"]["specificity"] is None


def test_seeded_patient_intervals_are_reproducible_and_json_safe():
    labels = ["a"] * 4 + ["b"] * 4
    scores = [
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.1, 0.9],
        [0.4, 0.6],
    ]
    patients = [f"p{i}" for i in range(8)]
    a = evaluate(labels, scores, patients, ["a", "b"], seed=17)
    b = evaluate(labels, scores, patients, ["a", "b"], seed=17)
    assert a == b
    ci = a["patient_level"]["confidence_intervals"]
    assert ci["method"] == "stratified_patient_bootstrap_percentile"
    assert ci["balanced_accuracy"]["lower"] <= 0.75 <= ci["balanced_accuracy"]["upper"]
    import json

    json.dumps(a, allow_nan=False)


@pytest.mark.parametrize(
    "labels,scores,patients,classes",
    [
        ([], [], [], ["a", "b"]),
        (["a"], [[float("nan"), 0.1]], ["p"], ["a", "b"]),
        (["a"], [[0.2, 0.1]], ["p"], ["a", "b"]),
        (["a"], [[1.1, -0.1]], ["p"], ["a", "b"]),
        (["a"], [[0.9, 0.1]], [""], ["a", "b"]),
        (["c"], [[0.9, 0.1]], ["p"], ["a", "b"]),
        (["a", "b"], [[0.9, 0.1], [0.1, 0.9]], ["p", "p"], ["a", "b"]),
        (["a"], [[0.9, 0.1]], ["p"], ["a", "a"]),
    ],
)
def test_invalid_or_conflicting_patient_data_is_rejected(labels, scores, patients, classes):
    with pytest.raises(ValueError):
        evaluate(labels, scores, patients, classes)


def test_each_patient_has_equal_weight_regardless_of_view_count():
    a = evaluate(
        ["a", "a", "b", "b"],
        [[0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        ["p1", "p2", "p3", "p4"],
        ["a", "b"],
    )
    b = evaluate(
        ["a", "a", "a", "b", "b"],
        [[0.6, 0.4], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        ["p1", "p1", "p2", "p3", "p4"],
        ["a", "b"],
    )
    assert a["patient_level"] == b["patient_level"]


def test_known_confusion_matrix_has_expected_sensitivity_and_specificity():
    result = evaluate(
        ["a", "a", "a", "b", "b"],
        [[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1], [0.2, 0.8]],
        ["p1", "p2", "p3", "p4", "p5"],
        ["a", "b"],
    )
    assert result["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["per_class"]["a"]["sensitivity"] == pytest.approx(2 / 3)
    assert result["per_class"]["a"]["specificity"] == 0.5
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
