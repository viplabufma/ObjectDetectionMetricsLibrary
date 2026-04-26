import json
import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tests.fixtures.coco_realistic_fixtures import (
    create_fixture_1TP1FP,
    create_fixture_duplicates,
    create_fixture_multiimage,
)


def run_debug_check() -> int:
    fixtures = [
        ("1TP1FP", create_fixture_1TP1FP),
        ("multiimage", create_fixture_multiimage),
        ("duplicates", create_fixture_duplicates),
    ]

    results = []
    for name, create_func in fixtures:
        gt_data, pred_data, exp_mAP, exp_mAP50, exp_mAP75, _ = create_func()
        expected = {"mAP": exp_mAP, "mAP50": exp_mAP50, "mAP75": exp_mAP75}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as gt_f:
            json.dump(gt_data, gt_f)
            gt_path = gt_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as pred_f:
            json.dump(pred_data, pred_f)
            pred_path = pred_f.name

        try:
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(pred_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

            with redirect_stdout(StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

            stats = coco_eval.stats
            actual = {
                "mAP": float(stats[0]),
                "mAP50": float(stats[1]),
                "mAP75": float(stats[2]),
            }

            match = all(abs(actual[k] - expected[k]) < 1e-6 for k in expected)
            status = "PASS" if match else "FAIL"
            results.append({"name": name, "expected": expected, "actual": actual, "status": status})
        finally:
            os.unlink(gt_path)
            os.unlink(pred_path)

    print("Comparison Table")
    print(f"{'Fixture':<12} {'mAP exp/act':<18} {'mAP50 exp/act':<18} {'mAP75 exp/act':<18} {'Status':<8}")
    print("-" * 80)
    for result in results:
        map_pair = f"{result['expected']['mAP']:.3f}/{result['actual']['mAP']:.3f}"
        map50_pair = f"{result['expected']['mAP50']:.3f}/{result['actual']['mAP50']:.3f}"
        map75_pair = f"{result['expected']['mAP75']:.3f}/{result['actual']['mAP75']:.3f}"
        print(
            f"{result['name']:<12} {map_pair:<18} {map50_pair:<18} {map75_pair:<18} {result['status']:<8}"
        )

    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(run_debug_check())
