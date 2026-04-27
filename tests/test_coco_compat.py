import numpy as np
import pytest

from detmet import DetectionMetricsManager
from tests.fixtures.coco_fixtures import (  # noqa: F401
    multiclass_mixed_fixture,
    perfect_detection_fixture,
    single_class_basic_fixture,
)
from tests.utils.coco_oracle import compute_coco_api_reference_metrics


@pytest.mark.coco_path
class TestComputeMapContract:
    """Validate COCO-backed mAP contract and deterministic fixture behavior."""

    def test_compute_map_returns_valid_metrics(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()
        exp_map, exp_map50, exp_map75, _ = compute_coco_api_reference_metrics(gt_path, pred_path)

        assert "global" in result.metrics
        assert "mAP" in result.metrics["global"]
        assert "mAP50" in result.metrics["global"]
        assert "mAP75" in result.metrics["global"]

        mAP = result.metrics["global"]["mAP"]
        mAP50 = result.metrics["global"]["mAP50"]
        mAP75 = result.metrics["global"]["mAP75"]

        assert float(mAP) == pytest.approx(exp_map, rel=1e-6)
        assert float(mAP50) == pytest.approx(exp_map50, rel=1e-6)
        assert float(mAP75) == pytest.approx(exp_map75, rel=1e-6)

    def test_perfect_detection_fixture_map_is_one(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        assert result.metrics["global"]["mAP"] == pytest.approx(1.0, rel=1e-2)
        assert result.metrics["global"]["mAP50"] == pytest.approx(1.0, rel=1e-2)
        assert result.metrics["global"]["mAP75"] == pytest.approx(1.0, rel=1e-2)

    def test_single_class_basic_fixture_ap_in_range(self, single_class_basic_fixture):
        gt_path, pred_path = single_class_basic_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()
        exp_map, exp_map50, exp_map75, exp_per_class = compute_coco_api_reference_metrics(
            gt_path, pred_path
        )

        assert float(result.metrics["global"]["mAP"]) == pytest.approx(exp_map, rel=1e-6)
        assert float(result.metrics["global"]["mAP50"]) == pytest.approx(exp_map50, rel=1e-6)
        assert float(result.metrics["global"]["mAP75"]) == pytest.approx(exp_map75, rel=1e-6)
        assert float(result.metrics["object"]["ap"]) == pytest.approx(exp_per_class[1], rel=1e-6)

    def test_multiclass_mixed_fixture_map_is_one(self, multiclass_mixed_fixture):
        gt_path, pred_path = multiclass_mixed_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        assert result.metrics["global"]["mAP"] == pytest.approx(1.0, rel=1e-2)

    def test_map50_is_not_less_than_map(self, single_class_basic_fixture):
        gt_path, pred_path = single_class_basic_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        assert result.metrics["global"]["mAP50"] >= result.metrics["global"]["mAP"]

    def test_per_class_ap_exists(self, multiclass_mixed_fixture):
        gt_path, pred_path = multiclass_mixed_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()
        _, _, _, exp_per_class = compute_coco_api_reference_metrics(gt_path, pred_path)

        assert "cat" in result.metrics
        assert "dog" in result.metrics
        assert "ap" in result.metrics["cat"]
        assert "ap" in result.metrics["dog"]
        assert float(result.metrics["cat"]["ap"]) == pytest.approx(exp_per_class[1], rel=1e-6)
        assert float(result.metrics["dog"]["ap"]) == pytest.approx(exp_per_class[2], rel=1e-6)


@pytest.mark.coco_path
class TestPRCurvesFromCOCOeval:
    """Validate PR curves extracted from COCOeval-backed metrics."""

    def test_pr_curves_exist_in_metrics(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        assert "pr_curves" in result.metrics
        assert "global" in result.metrics["pr_curves"]

    def test_pr_curve_has_recall_precision_ap(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        pr_global = result.metrics["pr_curves"]["global"]
        assert "recall" in pr_global
        assert "precision" in pr_global
        assert "ap" in pr_global
        assert len(pr_global["recall"]) > 0
        assert len(pr_global["precision"]) > 0
        assert isinstance(pr_global["ap"], (float, np.floating))

    def test_pr_curve_perfect_detection_high_precision(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        pr_global = result.metrics["pr_curves"]["global"]
        assert np.all(np.asarray(pr_global["precision"]) >= 0.99)

    def test_pr_curve_ap_matches_global_map_approximately(self, perfect_detection_fixture):
        gt_path, pred_path = perfect_detection_fixture
        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()
        exp_map, _, _, _ = compute_coco_api_reference_metrics(gt_path, pred_path)

        pr_ap = float(result.metrics["pr_curves"]["global"]["ap"])
        mAP = float(result.metrics["global"]["mAP"])
        assert pr_ap == pytest.approx(mAP, rel=1e-6)
        assert pr_ap == pytest.approx(exp_map, rel=1e-6)
