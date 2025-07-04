from .utils import safe_convert_to_float
import functools
import inspect
import math
import json
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import numpy as np

class ValidationError(ValueError):
    """Custom exception for validation errors"""
    pass

class PositiveInteger:
    """Validates that value is a positive finite integer using static methods"""
    @staticmethod
    def validate(value: Any) -> int:
        try:
            if math.isinf(value):
                raise ValueError(f"Value cannot be infinite, got {value}")
            if math.isnan(value):
                raise ValueError(f"Value cannot be NaN, got {value}")
            if value < 0:
                raise ValueError(f"Value must be >= 0, got {value}")
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"Value must be exact integer, got {value}")
            if not isinstance(value, (int, np.integer)) and not (
                isinstance(value, float) and value.is_integer()):
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    raise TypeError(f"Value must be integer-like, got {type(value).__name__}")
            else:
                ivalue = int(value)    

            return ivalue
        except (TypeError, ValueError) as e:
            raise ValidationError(f"PositiveInteger validation failed: {e}") from e
        
class ProbabilityFloat:
    """Validates that value is a probability between 0.0 and 1.0 inclusive"""
    @staticmethod
    def validate(value: Any) -> float:
        try:
            fvalue = safe_convert_to_float(value)
            if math.isinf(fvalue):
                raise ValueError(f"Value cannot be infinite, got {fvalue}")
            if math.isnan(fvalue):
                raise ValueError(f"Value cannot be NaN, got {fvalue}")
            if not (0.0 <= fvalue <= 1.0):
                raise ValueError(f"Value must be between 0.0 and 1.0, got {fvalue}")
            return fvalue
        except (TypeError, ValueError) as e:
            raise ValidationError(f"ProbabilityFloat validation failed: {e}") from e

class BoundingBox:
    """Validates and converts bounding box coordinates"""
    @staticmethod
    def validate(value: Any) -> Tuple[float, float, float, float]:
        try:
            # Convert to tuple if it's a NumPy array
            if isinstance(value, np.ndarray):
                value = tuple(value)
                
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Expected list, tuple or numpy array, got {type(value).__name__}")
            
            if len(value) != 4:
                raise ValueError(f"Expected 4 elements, got {len(value)}")
            
            try:
                coords = tuple(map(float, value))
            except (TypeError, ValueError) as e:
                raise ValueError("All elements must be numeric") from e
            
            for i, coord in enumerate(coords):
                if math.isnan(coord):
                    raise ValueError(f"Coordinate {i+1} is NaN")
                if math.isinf(coord):
                    raise ValueError(f"Coordinate {i+1} is infinite")
            
            x, y, w, h = coords
            
            if w < 0:
                raise ValueError(f"Width must be non-negative, got {w}")
            if h < 0:
                raise ValueError(f"Height must be non-negative, got {h}")
            
            return (x, y, w, h)
        except (TypeError, ValueError) as e:
            raise ValidationError(f"BoundingBox validation error: {e}") from e

class JSONData:
    """Validates and loads JSON data from various sources"""
    @staticmethod
    def validate(value: Any) -> dict:
        try:
            if isinstance(value, str):
                try:
                    with open(value, 'r') as f:
                        return json.load(f)
                except FileNotFoundError:
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        raise ValueError("Invalid JSON string")
                except OSError:
                    raise ValueError("Invalid file path")
            elif isinstance(value, dict):
                return value
            else:
                raise TypeError(f"Expected str, dict or file path, got {type(value).__name__}")
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise ValidationError(f"JSON validation failed: {e}") from e

class Boolean:
    """Validates that value is a boolean"""
    @staticmethod
    def validate(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 't', 'y'):
                return True
            if value.lower() in ('false', '0', 'no', 'f', 'n'):
                return False
        if isinstance(value, int):
            if value == 0:
                return False
            if value == 1:
                return True
        raise ValidationError(f"Value must be boolean, got {type(value).__name__}")

class AnnotationValidator:
    """Base class for annotation validators with static methods"""
    REQUIRED_KEYS: List[str] = []
    BBOX_REQUIRED: bool = True

    @classmethod
    def validate(cls, value: Any) -> dict:
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")
            
        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in value]
        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
            
        if cls.BBOX_REQUIRED and 'bbox' in value:
            value['bbox'] = BoundingBox.validate(value['bbox'])
            
        if 'score' in value:
            try:
                score = float(value['score'])
                if math.isnan(score) or math.isinf(score):
                    raise ValueError("Score cannot be NaN or infinite")
                value['score'] = score
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid score: {e}") from e
                
        if 'category_id' in value:
            try:
                value['category_id'] = int(value['category_id'])
            except (TypeError, ValueError):
                raise ValueError("category_id must be integer-like")
                    
        return value

class GroundTruthAnnotation(AnnotationValidator):
    """Validates ground truth annotations"""
    REQUIRED_KEYS = ['bbox']
    BBOX_REQUIRED = True

class PredictionAnnotation(AnnotationValidator):
    """Validates prediction annotations"""
    REQUIRED_KEYS = ['bbox', 'score']
    BBOX_REQUIRED = True

class AnnotationList:
    """Validates a list of annotations"""
    @staticmethod
    def validate(annotation_type: Type, value: Any) -> List[Any]:
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        return [annotation_type.validate(item) for item in value] if value else value

class ImageAnnotations:
    """Validates per-image annotations (list of annotation lists)"""
    @staticmethod
    def validate(annotation_type: Type, value: Any, min_length: int = 0) -> List[List[dict]]:
        if not isinstance(value, list):
            raise ValidationError(f"Expected list, got {type(value).__name__}")
            
        if len(value) < min_length:
            raise ValidationError(
                f"List must contain at least {min_length} elements, got {len(value)}"
            )            
        # Validate each image's annotations (allow empty inner lists)
        return [AnnotationList.validate(annotation_type, img_ann) for img_ann in value]
    
class PrecisionRecallResult:
    """Validates precision-recall curve result structure"""
    @staticmethod
    def validate(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")
            
        required_keys = ['precision', 'recall', 'thresholds', 'ap', 'per_class']
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")
            
        for key in ['precision', 'recall', 'thresholds']:
            if not isinstance(value[key], np.ndarray):
                raise TypeError(f"{key} must be numpy array, got {type(value[key]).__name__}")
                
        if not isinstance(value['ap'], (float, int)):
            raise TypeError(f"ap must be numeric, got {type(value['ap']).__name__}")
            
        if not isinstance(value['per_class'], dict):
            raise TypeError(f"per_class must be dict, got {type(value['per_class']).__name__}")
            
        for class_id, ap_val in value['per_class'].items():
            if not isinstance(ap_val, (float, int)):
                raise TypeError(f"per_class value for class {class_id} must be numeric")
                
        return value

# Specialized validators for function parameters
class GroundTruthAnnotations:
    """Validates all_gts parameter"""
    @staticmethod
    def validate(value: Any) -> Any:
        return ImageAnnotations.validate(GroundTruthAnnotation, value, min_length=1)

class PredictionAnnotations:
    """Validates all_preds parameter"""
    @staticmethod
    def validate(value: Any) -> Any:
        return ImageAnnotations.validate(PredictionAnnotation, value, min_length=1)

class RecallList:
    """Validates recall list for AP calculation"""
    @staticmethod
    def validate(value: Any) -> List[float]:
        if not isinstance(value, list):
            raise ValidationError(f"recall must be a list, got {type(value).__name__}")
            
        if len(value) == 0:
            raise ValidationError("recall cannot be an empty list")
            
        validated_list = []
        for i, r in enumerate(value):
            try:
                validated_r = ProbabilityFloat.validate(r)
            except ValidationError as e:
                raise ValidationError(f"recall[{i}]: {e}") from e
            validated_list.append(validated_r)
            
        # Check non-decreasing order
        for i in range(1, len(validated_list)):
            if validated_list[i] < validated_list[i-1]:
                raise ValidationError(
                    f"recall must be non-decreasing. "
                    f"recall[{i-1}]={validated_list[i-1]} > recall[{i}]={validated_list[i]}"
                )
                    
        return validated_list

class PrecisionList:
    """Validates precision list for AP calculation"""
    @staticmethod
    def validate(value: Any) -> List[float]:
        if not isinstance(value, list):
            raise ValidationError(f"precision must be a list, got {type(value).__name__}")
            
        if len(value) == 0:
            raise ValidationError("precision cannot be an empty list")
            
        validated_list = []
        for i, p in enumerate(value):
            try:
                validated_p = ProbabilityFloat.validate(p)
            except ValidationError as e:
                raise ValidationError(f"precision[{i}]: {e}") from e
            validated_list.append(validated_p)
            
        return validated_list
    
def validated(func: Callable) -> Callable:
    """Decorator to apply validators based on type annotations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        errors = []
        
        for name, value in bound.arguments.items():
            validator_class = sig.parameters[name].annotation
            
            if hasattr(validator_class, 'validate') and callable(validator_class.validate):
                try:
                    bound.arguments[name] = validator_class.validate(value)
                except ValidationError as e:
                    errors.append(f"  {name}: {e}")
        
        if errors:
            raise TypeError(f"Invalid arguments:\n" + "\n".join(errors))
        
        result = func(*bound.args, **bound.kwargs)
        
        return_validator = sig.return_annotation
        if hasattr(return_validator, 'validate') and callable(return_validator.validate):
            try:
                result = return_validator.validate(result)
            except ValidationError as e:
                raise ValidationError(f"Invalid return value: {e}") from e
                
        return result
    return wrapper