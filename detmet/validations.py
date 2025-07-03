from typing import Any, Callable, Dict, List, Tuple, Type, Union

def validate_normalized(value: float, param_name: str = "value") -> None:
    """
    Verify if a value is normalized (between 0.0 and 1.0 inclusive).
    
    Parameters
    ----------
    value : float
        The value to be checked for normalization.
    param_name : str, optional
        Name of the parameter for error messaging, by default "value".
    
    Raises
    ------
    ValueError
        If the value is not within the range [0.0, 1.0].
    
    Examples
    --------
    >>> check_normalized(0.5)  # Valid
    >>> check_normalized(1.0)  # Valid
    
    >>> check_normalized(-0.1)
    Traceback (most recent call last):
    ...
    ValueError: 'threshold' must be between 0.0 and 1.0 (got: -0.1)
    
    >>> check_normalized(1.5, "alpha")
    Traceback (most recent call last):
    ...
    ValueError: 'alpha' must be between 0.0 and 1.0 (got: 1.5)
    
    >>> parameters = {'threshold': 0.7, 'ratio': 1.2}
    >>> for name, val in parameters.items():
    ...     try:
    ...         check_normalized(val, name)
    ...     except ValueError as e:
    ...         print(f"Validation failed: {e}")
    Validation failed: 'ratio' must be between 0.0 and 1.0 (got: 1.2)
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"'{param_name}' must be between 0.0 and 1.0 (got: {value})"
        )

def validate_json_format(json_path: str):
    import json
    try:
        with open(json_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in path {json_path}: {str(e)}")


def validate_bbox_type(box: Union[list, tuple], box_name: str) -> None:
    """
    Validate the type of a bounding box.
    
    Parameters
    ----------
    box : list or tuple
        Bounding box coordinates
    box_name : str
        Identifier for the box (used in error messages)
    
    Raises
    ------
    TypeError
        If the input is not a list or tuple
    """
    if not isinstance(box, (list, tuple)):
        raise TypeError(f"{box_name} must be a list or tuple, got {type(box)}")

def validate_bbox_size(box: Union[list, tuple], box_name: str) -> None:
    """
    Validate the size of a bounding box.
    
    Parameters
    ----------
    box : list or tuple
        Bounding box coordinates
    box_name : str
        Identifier for the box (used in error messages)
    
    Raises
    ------
    ValueError
        If the box doesn't have exactly 4 elements
    """
    if len(box) != 4:
        raise ValueError(f"{box_name} must have 4 elements, got {len(box)}")

def convert_bbox_coordinates(box: Union[list, tuple], box_name: str) -> Tuple[float, float, float, float]:
    """
    Convert and validate bounding box coordinates to numeric values.
    
    Parameters
    ----------
    box : list or tuple
        Bounding box coordinates
    box_name : str
        Identifier for the box (used in error messages)
    
    Returns
    -------
    Tuple[float, float, float, float]
        Converted coordinates (x, y, width, height)
    
    Raises
    ------
    TypeError
        If coordinates can't be converted to float
    """
    try:
        x, y, w, h = map(float, box)
        return x, y, w, h
    except (TypeError, ValueError) as e:
        raise TypeError(f"All coordinates in {box_name} must be numeric") from e

def validate_coordinate_values(coords: List[float], box_name: str) -> None:
    """
    Validate coordinate values for NaN and infinity.
    
    Parameters
    ----------
    coords : list of float
        Bounding box coordinates to validate
    box_name : str
        Identifier for the box (used in error messages)
    
    Raises
    ------
    ValueError
        If any coordinate is NaN or infinity
    """
    if any(coord != coord or coord == float('inf') or coord == float('-inf') for coord in coords):
        raise ValueError(f"{box_name} contains NaN or infinite values: {coords}")

def validate_bbox_dimensions(w: float, h: float, box_name: str) -> None:
    """
    Validate bounding box dimensions.
    
    Parameters
    ----------
    w : float
        Width of the bounding box
    h : float
        Height of the bounding box
    box_name : str
        Identifier for the box (used in error messages)
    
    Raises
    ------
    ValueError
        If width or height are negative
    """
    if w < 0 or h < 0:
        raise ValueError(
            f"{box_name} has invalid dimensions: width={w}, height={h} "
            "(both must be >= 0)"
        )

def validate_bounding_boxes(box1: Union[list, tuple], box2: Union[list, tuple]) -> None:
    """
    Validate two bounding boxes for object detection operations.
    
    Performs comprehensive validation including:
    - Type checking (must be list or tuple)
    - Size validation (must have exactly 4 elements)
    - Numeric conversion and validation
    - NaN and infinity checks
    - Dimension validation (non-negative width/height)
    
    Parameters
    ----------
    box1 : list or tuple
        First bounding box in [x, y, width, height] format
    box2 : list or tuple
        Second bounding box in [x, y, width, height] format
    
    Raises
    ------
    TypeError
        If boxes are wrong type or contain non-numeric values
    ValueError
        For size issues, negative dimensions, or invalid numeric values
    """
    # Validate types
    validate_bbox_type(box1, "box1")
    validate_bbox_type(box2, "box2")
    
    # Validate sizes
    validate_bbox_size(box1, "box1")
    validate_bbox_size(box2, "box2")
    
    # Convert and validate coordinates
    x1, y1, w1, h1 = convert_bbox_coordinates(box1, "box1")
    x2, y2, w2, h2 = convert_bbox_coordinates(box2, "box2")
    
    # Validate coordinate values
    validate_coordinate_values([x1, y1, w1, h1], "box1")
    validate_coordinate_values([x2, y2, w2, h2], "box2")
    
    # Validate dimensions
    validate_bbox_dimensions(w1, h1, "box1")
    validate_bbox_dimensions(w2, h2, "box2")

def validate_int_and_int_like(value: int, name: str = 'Value'):
    import numpy as np
    if not isinstance(value, (int, np.integer)) if 'numpy' in globals() else not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be integer, got {type(value).__name__}")

def validate_positive(value: Union[int, float], name: str = 'Value'):
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")

def validate_float_and_float_like(value: float, name: str = 'Value'):
    try:
        float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be float, got {type(value).__name__}")

def validate_number_is_not_NaN(value: Any, name: str = "Value"):
    import math
    if not isinstance(value, (float, int, complex)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be NaN")

def validate_number_is_not_infinite(value: Union[int, float], name: str = "Value"):
    if not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if value == float('inf'):
        raise ValueError(f"{value} cannot be infinite")