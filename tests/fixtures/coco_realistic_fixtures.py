import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

import pytest
from tests.utils.coco_oracle import compute_coco_api_reference_metrics


def _compute_expected_from_coco_api(
    coco_data: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Tuple[float, float, float, Dict[int, float]]:
    """Compute expected metrics at runtime using official COCO API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = os.path.join(tmpdir, "gt.json")
        pred_path = os.path.join(tmpdir, "predictions.json")

        with open(gt_path, "w") as f:
            json.dump(coco_data, f)
        with open(pred_path, "w") as f:
            json.dump(predictions, f)

        return compute_coco_api_reference_metrics(gt_path, pred_path)


def create_fixture_1TP1FP() -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, float, float, Dict[int, float]]:
    """
    1 image, 2 GTs, 2 predictions for one class: 1 likely TP + 1 FP.
    Expected values are calibrated with COCOeval.
    """
    coco_data = {
        "info": {
            "description": "Realistic 1TP+1FP fixture",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "image_1.jpg",
                "height": 640,
                "width": 640,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 40000.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [400.0, 400.0, 200.0, 200.0],
                "area": 40000.0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {
                "id": 1,
                "name": "object",
                "supercategory": "thing",
            }
        ],
    }

    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [110.0, 110.0, 180.0, 180.0],
            "score": 0.95,
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [50.0, 50.0, 80.0, 80.0],
            "score": 0.85,
        },
    ]

    expected_mAP, expected_mAP50, expected_mAP75, per_class_ap = _compute_expected_from_coco_api(
        coco_data, predictions
    )

    return coco_data, predictions, expected_mAP, expected_mAP50, expected_mAP75, per_class_ap


def create_fixture_multiimage() -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, float, float, Dict[int, float]]:
    """
    5 images with varied TP/FP/FN behavior and duplicate detections.
    Expected values are calibrated with COCOeval.
    """
    coco_data = {
        "info": {
            "description": "Multiimage realistic fixture",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": i,
                "file_name": f"image_{i}.jpg",
                "height": 640,
                "width": 640,
            }
            for i in range(1, 6)
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 100.0, 100.0], "area": 10000.0, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200.0, 200.0, 100.0, 100.0], "area": 10000.0, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [100.0, 100.0, 80.0, 80.0], "area": 6400.0, "iscrowd": 0},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [300.0, 300.0, 80.0, 80.0], "area": 6400.0, "iscrowd": 0},
            {"id": 5, "image_id": 3, "category_id": 1, "bbox": [150.0, 150.0, 120.0, 120.0], "area": 14400.0, "iscrowd": 0},
            {"id": 6, "image_id": 4, "category_id": 1, "bbox": [50.0, 50.0, 90.0, 90.0], "area": 8100.0, "iscrowd": 0},
            {"id": 7, "image_id": 4, "category_id": 1, "bbox": [200.0, 200.0, 90.0, 90.0], "area": 8100.0, "iscrowd": 0},
            {"id": 8, "image_id": 4, "category_id": 1, "bbox": [350.0, 350.0, 90.0, 90.0], "area": 8100.0, "iscrowd": 0},
            {"id": 9, "image_id": 5, "category_id": 1, "bbox": [100.0, 100.0, 110.0, 110.0], "area": 12100.0, "iscrowd": 0},
            {"id": 10, "image_id": 5, "category_id": 1, "bbox": [300.0, 300.0, 110.0, 110.0], "area": 12100.0, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
    }

    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [52.0, 52.0, 96.0, 96.0], "score": 0.95},
        {"image_id": 1, "category_id": 1, "bbox": [202.0, 202.0, 96.0, 96.0], "score": 0.92},
        {"image_id": 2, "category_id": 1, "bbox": [102.0, 102.0, 76.0, 76.0], "score": 0.88},
        {"image_id": 3, "category_id": 1, "bbox": [152.0, 152.0, 116.0, 116.0], "score": 0.85},
        {"image_id": 3, "category_id": 1, "bbox": [50.0, 50.0, 80.0, 80.0], "score": 0.80},
        {"image_id": 3, "category_id": 1, "bbox": [400.0, 400.0, 60.0, 60.0], "score": 0.75},
        {"image_id": 4, "category_id": 1, "bbox": [52.0, 52.0, 86.0, 86.0], "score": 0.90},
        {"image_id": 4, "category_id": 1, "bbox": [202.0, 202.0, 86.0, 86.0], "score": 0.87},
        {"image_id": 5, "category_id": 1, "bbox": [102.0, 102.0, 106.0, 106.0], "score": 0.93},
        {"image_id": 5, "category_id": 1, "bbox": [302.0, 302.0, 106.0, 106.0], "score": 0.91},
        {"image_id": 5, "category_id": 1, "bbox": [103.0, 103.0, 104.0, 104.0], "score": 0.89},
        {"image_id": 5, "category_id": 1, "bbox": [500.0, 500.0, 50.0, 50.0], "score": 0.70},
    ]

    expected_mAP, expected_mAP50, expected_mAP75, per_class_ap = _compute_expected_from_coco_api(
        coco_data, predictions
    )

    return coco_data, predictions, expected_mAP, expected_mAP50, expected_mAP75, per_class_ap


def create_fixture_duplicates() -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, float, float, Dict[int, float]]:
    """
    1 image, 2 GT, 4 preds with duplicates: 2 TP + duplicate FP + extra FP.
    """
    coco_data = {
        "info": {
            "description": "Duplicates fixture",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "image_dup.jpg",
                "height": 640,
                "width": 640,
            }
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100.0, 100.0, 150.0, 150.0], "area": 22500.0, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300.0, 300.0, 150.0, 150.0], "area": 22500.0, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
    }

    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [102.0, 102.0, 146.0, 146.0], "score": 0.96},
        {"image_id": 1, "category_id": 1, "bbox": [302.0, 302.0, 146.0, 146.0], "score": 0.94},
        {"image_id": 1, "category_id": 1, "bbox": [103.0, 103.0, 144.0, 144.0], "score": 0.92},
        {"image_id": 1, "category_id": 1, "bbox": [500.0, 500.0, 80.0, 80.0], "score": 0.80},
    ]

    expected_mAP, expected_mAP50, expected_mAP75, per_class_ap = _compute_expected_from_coco_api(
        coco_data, predictions
    )

    return coco_data, predictions, expected_mAP, expected_mAP50, expected_mAP75, per_class_ap


@pytest.fixture
def fixture_1TP1FP(tmp_path):
    """Pytest fixture for the 1TP+1FP realistic scenario."""
    coco_data, predictions, exp_mAP, exp_mAP50, exp_mAP75, exp_per_class = create_fixture_1TP1FP()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file), exp_mAP, exp_mAP50, exp_mAP75, exp_per_class


@pytest.fixture
def fixture_multiimage(tmp_path):
    """Pytest fixture for the multi-image realistic scenario."""
    coco_data, predictions, exp_mAP, exp_mAP50, exp_mAP75, exp_per_class = create_fixture_multiimage()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file), exp_mAP, exp_mAP50, exp_mAP75, exp_per_class


@pytest.fixture
def fixture_duplicates(tmp_path):
    """Pytest fixture for the duplicate-detection scenario."""
    coco_data, predictions, exp_mAP, exp_mAP50, exp_mAP75, exp_per_class = create_fixture_duplicates()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file), exp_mAP, exp_mAP50, exp_mAP75, exp_per_class


# Backward-compatible aliases used by ad-hoc debug scripts.
# These wrappers return only (gt_data, pred_data).
def create_1tp1fp_fixture() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gt_data, pred_data, *_ = create_fixture_1TP1FP()
    return gt_data, pred_data


def create_multiimage_fixture() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gt_data, pred_data, *_ = create_fixture_multiimage()
    return gt_data, pred_data


def create_duplicates_fixture() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gt_data, pred_data, *_ = create_fixture_duplicates()
    return gt_data, pred_data
