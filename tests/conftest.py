import pytest


def pytest_configure(config):
    """Register custom pytest markers used by this project."""
    config.addinivalue_line(
        "markers",
        "local_path: Tests that validate the local confusion matrix path only",
    )
    config.addinivalue_line(
        "markers",
        "coco_path: Tests that validate COCO-backed mAP/AP integration",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests that use real datasets (slower, may be brittle)",
    )
    config.addinivalue_line(
        "markers",
        "coco_numerical: Tests that validate numerical equivalence vs official COCO API",
    )
