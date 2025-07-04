import pytest
import math
import numpy as np
from detmet.validations import (
    ValidationError,
    PositiveInteger,
    ProbabilityFloat,
    BoundingBox,
    JSONData,
    Boolean,
    GroundTruthAnnotation,
    PredictionAnnotation,
    AnnotationList,
    ImageAnnotations,
    PrecisionRecallResult,
    GroundTruthAnnotations,
    PredictionAnnotations,
    RecallList,
    PrecisionList,
    validated
)

# PositiveInteger Tests
def test_positive_integer_valid():
    """Test valid positive integers of various types"""
    assert PositiveInteger.validate(5) == 5
    assert PositiveInteger.validate(0) == 0
    assert PositiveInteger.validate(np.int32(10)) == 10
    assert PositiveInteger.validate(5.0) == 5  # Integer float

def test_positive_integer_invalid():
    """Test invalid inputs for PositiveInteger"""
    with pytest.raises(ValidationError):
        PositiveInteger.validate(-5)  # Negative
    
    with pytest.raises(ValidationError):
        PositiveInteger.validate(5.5)  # Non-integer float
    
    with pytest.raises(ValidationError):
        PositiveInteger.validate("abc")  # Non-numeric string
    
    with pytest.raises(ValidationError):
        PositiveInteger.validate(math.inf)  # Infinity

# ProbabilityFloat Tests
def test_probability_float_valid():
    """Test valid probability values"""
    assert ProbabilityFloat.validate(0.5) == 0.5
    assert ProbabilityFloat.validate(0.0) == 0.0
    assert ProbabilityFloat.validate(1.0) == 1.0
    assert ProbabilityFloat.validate("0.7") == 0.7  # String conversion

def test_probability_float_invalid():
    """Test invalid probability values"""
    with pytest.raises(ValidationError):
        ProbabilityFloat.validate(-0.1)  # Below 0
    
    with pytest.raises(ValidationError):
        ProbabilityFloat.validate(1.1)  # Above 1
    
    with pytest.raises(ValidationError):
        ProbabilityFloat.validate(math.nan)  # NaN
    
    with pytest.raises(ValidationError):
        ProbabilityFloat.validate("abc")  # Non-numeric string

# BoundingBox Tests
def test_bounding_box_valid():
    """Test valid bounding boxes"""
    # Standard types
    assert BoundingBox.validate([10, 20, 30, 40]) == (10, 20, 30, 40)
    assert BoundingBox.validate((5.5, 10.5, 15.0, 20.0)) == (5.5, 10.5, 15.0, 20.0)
    
    # NumPy arrays
    assert BoundingBox.validate(np.array([1, 2, 3, 4])) == (1, 2, 3, 4)
    assert BoundingBox.validate(np.array([1.5, 2.5, 3.5, 4.5])) == (1.5, 2.5, 3.5, 4.5)
    
    # Edge cases
    assert BoundingBox.validate([0, 0, 0, 0]) == (0, 0, 0, 0)  # Zero size
    assert BoundingBox.validate(["10", "20", "30.5", "40"]) == (10, 20, 30.5, 40)  # String numbers

def test_bounding_box_invalid():
    """Test invalid bounding boxes"""
    # Wrong types
    with pytest.raises(ValidationError):
        BoundingBox.validate("invalid")
    
    with pytest.raises(ValidationError):
        BoundingBox.validate({"key": "value"})
    
    # Wrong sizes
    with pytest.raises(ValidationError):
        BoundingBox.validate([10, 20])  # Too few
        
    with pytest.raises(ValidationError):
        BoundingBox.validate(np.array([1, 2, 3]))  # Too few in array
        
    with pytest.raises(ValidationError):
        BoundingBox.validate([10, 20, 30, 40, 50])  # Too many
    
    # Invalid values
    with pytest.raises(ValidationError):
        BoundingBox.validate([10, 20, -5, 30])  # Negative width
        
    with pytest.raises(ValidationError):
        BoundingBox.validate([10, 20, 30, "abc"])  # Non-numeric
        
    with pytest.raises(ValidationError):
        BoundingBox.validate([math.inf, 0, 10, 10])  # Infinite value
        
    with pytest.raises(ValidationError):
        BoundingBox.validate([math.nan, 0, 10, 10])  # NaN value


# JSONData Tests
def test_json_data_valid(tmp_path):
    """Test valid JSON inputs"""
    # Valid JSON string
    assert JSONData.validate('{"key": "value"}') == {"key": "value"}
    
    # Valid dictionary
    assert JSONData.validate({"key": "value"}) == {"key": "value"}
    
    # Valid JSON file
    file_path = tmp_path / "test.json"
    file_path.write_text('{"file": "content"}')
    assert JSONData.validate(str(file_path)) == {"file": "content"}

def test_json_data_invalid(tmp_path):
    """Test invalid JSON inputs"""
    with pytest.raises(ValidationError):
        JSONData.validate("invalid json")  # Malformed JSON string
    
    with pytest.raises(ValidationError):
        JSONData.validate(123)  # Non-string/non-dict
    
    # Invalid file path
    with pytest.raises(ValidationError):
        JSONData.validate(str(tmp_path / "nonexistent.json"))

# Boolean Tests
def test_boolean_valid():
    """Test valid boolean representations"""
    assert Boolean.validate(True) is True
    assert Boolean.validate("true") is True
    assert Boolean.validate("1") is True
    assert Boolean.validate("no") is False
    assert Boolean.validate(0) is False

def test_boolean_invalid():
    """Test invalid boolean representations"""
    with pytest.raises(ValidationError):
        Boolean.validate("maybe")
    
    with pytest.raises(ValidationError):
        Boolean.validate(2)

# AnnotationValidator Tests
def test_ground_truth_annotation_valid():
    """Test valid ground truth annotations"""
    ann = {"bbox": [10, 20, 30, 40]}
    validated = GroundTruthAnnotation.validate(ann)
    assert validated["bbox"] == (10, 20, 30, 40)

def test_prediction_annotation_valid():
    """Test valid prediction annotations"""
    ann = {"bbox": [10, 20, 30, 40], "score": 0.9}
    validated = PredictionAnnotation.validate(ann)
    assert validated["bbox"] == (10, 20, 30, 40)
    assert validated["score"] == 0.9

def test_annotation_missing_keys():
    """Test annotations missing required keys"""
    with pytest.raises(ValueError):
        GroundTruthAnnotation.validate({})  # Missing bbox
    
    with pytest.raises(ValueError):
        PredictionAnnotation.validate({"bbox": [1,2,3,4]})  # Missing score

# AnnotationList Tests
def test_annotation_list_valid():
    """Test valid annotation lists"""
    anns = [{"bbox": [1,2,3,4]}, {"bbox": [5,6,7,8]}]
    validated_list = AnnotationList.validate(GroundTruthAnnotation, anns)
    assert len(validated_list) == 2
    assert validated_list[0]["bbox"] == (1,2,3,4)

def test_annotation_list_invalid():
    """Test invalid annotation lists"""
    with pytest.raises(TypeError):
        AnnotationList.validate(GroundTruthAnnotation, "not a list")
    
    # Invalid item in list
    with pytest.raises(ValueError):
        AnnotationList.validate(GroundTruthAnnotation, [{"invalid": "data"}])

# ImageAnnotations Tests
def test_image_annotations_valid():
    """Test valid image annotations"""
    data = [[{"bbox": [1,2,3,4]}], [{"bbox": [5,6,7,8]}]]
    validated = ImageAnnotations.validate(
        GroundTruthAnnotation, 
        data, 
        min_length=2
    )
    assert len(validated) == 2

def test_image_annotations_invalid():
    """Test invalid image annotations"""
    # Not a list
    with pytest.raises(ValidationError):
        ImageAnnotations.validate(GroundTruthAnnotation, "not list", min_length=1)
    
    # Too short (outer list)
    with pytest.raises(ValidationError) as exc_info:
        ImageAnnotations.validate(
            GroundTruthAnnotation, 
            [[{"bbox": [1,2,3,4]}], []],  # Valid inner lists
            min_length=3  # But outer list too short
        )
    assert "at least 3 elements" in str(exc_info.value)
    
    # Invalid item in outer list
    with pytest.raises(TypeError):
        ImageAnnotations.validate(
            GroundTruthAnnotation, 
            [{"invalid": "data"}],  # Not a list of annotations
            min_length=1
        )
    
    # Valid empty inner lists
    result = ImageAnnotations.validate(
        GroundTruthAnnotation, 
        [[], [{"bbox": [5,6,7,8]}]],
        min_length=2
    )
    assert len(result) == 2
    assert len(result[0]) == 0  # Empty image is valid

# PrecisionRecallResult Tests
def test_precision_recall_result_valid():
    """Test valid precision-recall results"""
    data = {
        "precision": np.array([0.9, 0.8]),
        "recall": np.array([0.7, 0.6]),
        "thresholds": np.array([0.5, 0.4]),
        "ap": 0.85,
        "per_class": {1: 0.9, 2: 0.8}
    }
    validated = PrecisionRecallResult.validate(data)
    assert validated == data

def test_precision_recall_result_invalid():
    """Test invalid precision-recall results"""
    # Missing key
    with pytest.raises(ValueError):
        PrecisionRecallResult.validate({"precision": [], "recall": []})
    
    # Wrong type
    with pytest.raises(TypeError):
        PrecisionRecallResult.validate({
            "precision": [0.9],  # Should be np.array
            "recall": np.array([0.7]),
            "thresholds": np.array([0.5]),
            "ap": 0.8,
            "per_class": {}
        })

# RecallList Tests
def test_recall_list_valid():
    """Test valid recall lists"""
    assert RecallList.validate([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]
    assert RecallList.validate([0.3, 0.3, 0.4]) == [0.3, 0.3, 0.4]  # Non-decreasing

def test_recall_list_invalid():
    """Test invalid recall lists"""
    with pytest.raises(ValidationError):
        RecallList.validate([])  # Empty list
    
    with pytest.raises(ValidationError):
        RecallList.validate([0.5, 0.4, 0.6])  # Not non-decreasing
    
    with pytest.raises(ValidationError):
        RecallList.validate([-0.1, 0.5])  # Invalid probability

# PrecisionList Tests
def test_precision_list_valid():
    """Test valid precision lists"""
    assert PrecisionList.validate([0.9, 0.8, 0.7]) == [0.9, 0.8, 0.7]

def test_precision_list_invalid():
    """Test invalid precision lists"""
    with pytest.raises(ValidationError):
        PrecisionList.validate([])  # Empty list
    
    with pytest.raises(ValidationError):
        PrecisionList.validate([1.1, 0.5])  # Invalid probability

def test_validated_decorator_success():
    """Test successful validation with decorator"""
    @validated
    def sample_func(a: PositiveInteger, b: ProbabilityFloat) -> ProbabilityFloat:
        # Return a value within the valid probability range
        return b * 0.5  # 0.4 * 0.5 = 0.2
    
    # Test both input validation and output validation
    assert sample_func(5, 0.4) == 0.2

def test_validated_decorator_failure():
    """Test validation failure with decorator"""
    # Input validation failure
    @validated
    def sample_func(a: PositiveInteger) -> PositiveInteger:
        return a
    
    with pytest.raises(TypeError):
        sample_func(-5)  # Invalid input
    
    # Output validation failure
    @validated
    def invalid_return() -> ProbabilityFloat:
        return 1.5  # Out of range
    
    with pytest.raises(ValidationError):
        invalid_return()

# GroundTruthAnnotations/PredictionAnnotations Tests
def test_gt_annotations_valid():
    """Test valid ground truth annotations"""
    data = [[{"bbox": [1,2,3,4]}], [{"bbox": [5,6,7,8]}]]
    validated = GroundTruthAnnotations.validate(data)
    assert len(validated) == 2

def test_annotations_valid():
    """Test valid annotations structure"""
    # Valid GroundTruthAnnotations
    data = [[{"bbox": [1,2,3,4]}], [{"bbox": [5,6,7,8]}]]
    validated = GroundTruthAnnotations.validate(data)
    assert len(validated) == 2
    
    # Valid PredictionAnnotations
    data = [[{"bbox": [1,2,3,4], "score": 0.9}]]
    validated = PredictionAnnotations.validate(data)
    assert len(validated) == 1
    
    # Valid empty images
    data = [[], [{"bbox": [5,6,7,8], "score": 0.9}]]
    validated = PredictionAnnotations.validate(data)
    assert len(validated) == 2
    assert len(validated[0]) == 0

def test_annotations_invalid():
    """Test invalid annotations structure"""
    # GroundTruthAnnotations - empty outer list
    with pytest.raises(ValidationError) as exc_info:
        GroundTruthAnnotations.validate([])
    assert "at least 1 elements, got 0" in str(exc_info.value)
    
    # GroundTruthAnnotations - invalid type
    with pytest.raises(ValidationError):
        GroundTruthAnnotations.validate("invalid")
    
    # GroundTruthAnnotations - invalid inner item
    with pytest.raises(TypeError):
        GroundTruthAnnotations.validate([{"invalid": "data"}])
    
    # PredictionAnnotations - empty outer list
    with pytest.raises(ValidationError) as exc_info:
        PredictionAnnotations.validate([])
    assert "at least 1 elements, got 0" in str(exc_info.value)
    
    # PredictionAnnotations - missing score
    with pytest.raises(ValueError):
        PredictionAnnotations.validate([[{"bbox": [1,2,3,4]}]])