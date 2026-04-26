import numpy as np
import pytest

from detmet import DetectionMetricsManager
from tests.fixtures.coco_realistic_fixtures import (  # noqa: F401
    fixture_1TP1FP,
    fixture_duplicates,
    fixture_multiimage,
)
from tests.utils.coco_oracle import compute_coco_api_reference_metrics


@pytest.mark.coco_numerical
class TestNumericalEquivalenceWithCOCOAPI:
    """Numerically compare project metrics against official COCO API output."""

    def test_map_1tp1fp_matches_coco_api(self, fixture_1TP1FP):
        (
            gt_path,
            pred_path,
            exp_mAP,
            exp_mAP50,
            exp_mAP75,
            exp_per_class,
        ) = fixture_1TP1FP

        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        project_mAP = float(result.metrics["global"]["mAP"])
        project_mAP50 = float(result.metrics["global"]["mAP50"])
        project_mAP75 = float(result.metrics["global"]["mAP75"])

        coco_mAP, coco_mAP50, coco_mAP75, coco_per_class = compute_coco_api_reference_metrics(
            gt_path, pred_path
        )

        assert project_mAP == pytest.approx(exp_mAP, rel=1e-3)
        assert project_mAP == pytest.approx(coco_mAP, rel=1e-6)
        assert project_mAP50 == pytest.approx(exp_mAP50, rel=1e-3)
        assert project_mAP50 == pytest.approx(coco_mAP50, rel=1e-6)
        assert project_mAP75 == pytest.approx(exp_mAP75, rel=1e-3)
        assert project_mAP75 == pytest.approx(coco_mAP75, rel=1e-6)

        # Class key is converted to class name in manager output
        project_ap = float(result.metrics["object"].get("ap", 0.0))
        assert project_ap == pytest.approx(exp_per_class[1], rel=1e-3)
        assert project_ap == pytest.approx(coco_per_class[1], rel=1e-6)

    def test_map_multiimage_matches_coco_api(self, fixture_multiimage):
        (
            gt_path,
            pred_path,
            exp_mAP,
            exp_mAP50,
            exp_mAP75,
            _,
        ) = fixture_multiimage

        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        project_mAP = float(result.metrics["global"]["mAP"])
        project_mAP50 = float(result.metrics["global"]["mAP50"])
        project_mAP75 = float(result.metrics["global"]["mAP75"])

        coco_mAP, coco_mAP50, coco_mAP75, _ = compute_coco_api_reference_metrics(
            gt_path, pred_path
        )

        assert project_mAP == pytest.approx(exp_mAP, rel=1e-3)
        assert project_mAP == pytest.approx(coco_mAP, rel=1e-6)
        assert project_mAP50 == pytest.approx(exp_mAP50, rel=1e-3)
        assert project_mAP50 == pytest.approx(coco_mAP50, rel=1e-6)
        assert project_mAP75 == pytest.approx(exp_mAP75, rel=1e-3)
        assert project_mAP75 == pytest.approx(coco_mAP75, rel=1e-6)

    def test_map_duplicates_matches_coco_api(self, fixture_duplicates):
        (
            gt_path,
            pred_path,
            exp_mAP,
            exp_mAP50,
            exp_mAP75,
            _,
        ) = fixture_duplicates

        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        project_mAP = float(result.metrics["global"]["mAP"])
        project_mAP50 = float(result.metrics["global"]["mAP50"])
        project_mAP75 = float(result.metrics["global"]["mAP75"])

        coco_mAP, coco_mAP50, coco_mAP75, _ = compute_coco_api_reference_metrics(
            gt_path, pred_path
        )

        assert project_mAP == pytest.approx(exp_mAP, rel=1e-3)
        assert project_mAP == pytest.approx(coco_mAP, rel=1e-6)
        assert project_mAP50 == pytest.approx(exp_mAP50, rel=1e-3)
        assert project_mAP50 == pytest.approx(coco_mAP50, rel=1e-6)
        assert project_mAP75 == pytest.approx(exp_mAP75, rel=1e-3)
        assert project_mAP75 == pytest.approx(coco_mAP75, rel=1e-6)


@pytest.mark.coco_numerical
class TestPRCurveNumericalValidation:
    """Validate PR curve structure and AP consistency for COCO-backed curves."""

    def test_pr_curve_1tp1fp_has_101_points_and_monotonic_precision(self, fixture_1TP1FP):
        gt_path, pred_path, exp_mAP, _, _, _ = fixture_1TP1FP

        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        pr_global = result.metrics["pr_curves"]["global"]
        recall = np.asarray(pr_global["recall"])
        precision = np.asarray(pr_global["precision"])

        assert len(recall) == 101
        assert len(precision) == 101
        assert recall[0] == pytest.approx(0.0)
        assert recall[-1] == pytest.approx(1.0)

        # Precision in COCOeval is non-increasing across recall thresholds
        assert np.all(np.diff(precision) <= 1e-8)

        assert float(pr_global["ap"]) == pytest.approx(exp_mAP, rel=1e-3)

    def test_pr_curve_ap_matches_global_map(self, fixture_multiimage):
        gt_path, pred_path, _, _, _, _ = fixture_multiimage

        manager = DetectionMetricsManager(
            groundtruth_json_path=gt_path,
            prediction_json_path=pred_path,
        )
        result = manager.calculate_metrics()

        pr_ap = float(result.metrics["pr_curves"]["global"]["ap"])
        mAP = float(result.metrics["global"]["mAP"])
        assert pr_ap == pytest.approx(mAP, rel=5e-2)
