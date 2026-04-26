import json
import pytest


def create_perfect_detection_fixture() -> tuple:
    """
    Single image, single GT, single perfect prediction (IoU=1.0).
    """
    coco_data = {
        "info": {
            "description": "Perfect detection fixture for COCO validation",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "perfect.jpg",
                "height": 480,
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
            }
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
            "bbox": [100.0, 100.0, 200.0, 200.0],
            "score": 0.99,
        }
    ]

    return coco_data, predictions


def create_single_class_basic_fixture() -> tuple:
    """
    Single image, 2 GTs (same class), 2 preds (same class): 1 TP + 1 FP.
    """
    coco_data = {
        "info": {
            "description": "Basic single-class fixture for AP validation",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "basic.jpg",
                "height": 480,
                "width": 640,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 100.0, 100.0],
                "area": 10000.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [200.0, 200.0, 100.0, 100.0],
                "area": 10000.0,
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
            "bbox": [12.0, 12.0, 95.0, 95.0],
            "score": 0.95,
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [400.0, 400.0, 50.0, 50.0],
            "score": 0.85,
        },
    ]

    return coco_data, predictions


def create_multiclass_mixed_fixture() -> tuple:
    """
    Single image, 2 GTs (different classes), 2 perfect preds.
    """
    coco_data = {
        "info": {
            "description": "Multiclass fixture for mAP validation",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "multiclass.jpg",
                "height": 480,
                "width": 640,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 100.0, 100.0],
                "area": 10000.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [200.0, 200.0, 100.0, 100.0],
                "area": 10000.0,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "thing"},
            {"id": 2, "name": "dog", "supercategory": "thing"},
        ],
    }

    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 100.0, 100.0],
            "score": 0.99,
        },
        {
            "image_id": 1,
            "category_id": 2,
            "bbox": [200.0, 200.0, 100.0, 100.0],
            "score": 0.99,
        },
    ]

    return coco_data, predictions


@pytest.fixture
def perfect_detection_fixture(tmp_path):
    """Save perfect-detection fixture to temporary JSON files."""
    coco_data, predictions = create_perfect_detection_fixture()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)

    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file)


@pytest.fixture
def single_class_basic_fixture(tmp_path):
    """Save single-class basic fixture to temporary JSON files."""
    coco_data, predictions = create_single_class_basic_fixture()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)

    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file)


@pytest.fixture
def multiclass_mixed_fixture(tmp_path):
    """Save multiclass mixed fixture to temporary JSON files."""
    coco_data, predictions = create_multiclass_mixed_fixture()

    gt_file = tmp_path / "gt.json"
    pred_file = tmp_path / "predictions.json"

    with open(gt_file, "w") as f:
        json.dump(coco_data, f)

    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    return str(gt_file), str(pred_file)
