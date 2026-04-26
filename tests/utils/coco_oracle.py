from typing import Dict, Tuple
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_coco_api_reference_metrics(
    gt_json_path: str,
    pred_json_path: str,
) -> Tuple[float, float, float, Dict[int, float]]:
    """Compute COCO reference metrics using official COCOeval as test oracle."""
    gt_coco = COCO(gt_json_path)
    pred_coco = gt_coco.loadRes(pred_json_path)

    evaluator = COCOeval(gt_coco, pred_coco, "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    mAP = float(evaluator.stats[0])
    mAP50 = float(evaluator.stats[1])
    mAP75 = float(evaluator.stats[2])

    per_class_ap: Dict[int, float] = {}
    if hasattr(evaluator, "eval") and "precision" in evaluator.eval:
        try:
            aind = next(
                i for i, label in enumerate(evaluator.params.areaRngLbl) if label == "all"
            )
        except StopIteration:
            aind = 0
        try:
            mind = next(
                i for i, max_det in enumerate(evaluator.params.maxDets) if max_det == 100
            )
        except StopIteration:
            mind = len(evaluator.params.maxDets) - 1

        precision = evaluator.eval["precision"]  # [T, R, K, A, M]
        for k, cat_id in enumerate(evaluator.params.catIds):
            pr_array = precision[:, :, k, aind, mind]
            valid_pr = pr_array[pr_array > -1]
            ap = float(valid_pr.mean()) if valid_pr.size > 0 else 0.0
            per_class_ap[int(cat_id)] = ap

    return mAP, mAP50, mAP75, per_class_ap


def validate_fixture_expected_values(
    gt_json_path: str,
    pred_json_path: str,
    fixture_name: str,
) -> Dict[str, object]:
    """Utility helper to print oracle values for fixture calibration."""
    mAP, mAP50, mAP75, per_class_ap = compute_coco_api_reference_metrics(
        gt_json_path, pred_json_path
    )
    return {
        "fixture": fixture_name,
        "mAP": mAP,
        "mAP50": mAP50,
        "mAP75": mAP75,
        "per_class_ap": per_class_ap,
    }
