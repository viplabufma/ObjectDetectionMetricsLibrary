from typing import Any, List, Union
import numpy as np

def map_class_keys_recursive(
    obj: Union[dict, list, Any], 
    class_list: List[str]
) -> Union[dict, list, Any]:
    """
    Recursively maps numeric keys to class names throughout all dictionary levels.
    
    Args:
        obj: Dictionary or object to process
        class_list: List of class names
    
    Returns:
        Object with numeric keys replaced by class names
    """
    if class_list and class_list[-1] == 'background':
        class_list = class_list[:-1]
    
    if isinstance(obj, dict):
        # Process dictionaries
        new_dict = {}
        
        # Separate numeric and non-numeric keys
        numeric_keys = []
        non_numeric_items = {}
        
        for key, value in obj.items():
            try:
                # Try to convert key to int
                num_key = int(key)
                numeric_keys.append((key, num_key, value))
            except (ValueError, TypeError):
                # Non-numeric key - process value recursively
                non_numeric_items[key] = map_class_keys_recursive(value, class_list)
        
        # Sort numeric keys
        numeric_keys.sort(key=lambda x: x[1])
        
        # Map numeric keys to class names
        for idx, (orig_key, num_key, value) in enumerate(numeric_keys):
            if idx < len(class_list):
                new_key = class_list[idx]
            else:
                new_key = orig_key  # Keep original if no matching name
            
            new_dict[new_key] = map_class_keys_recursive(value, class_list)
        
        # Add non-numeric items
        new_dict.update(non_numeric_items)
        return new_dict
    
    elif isinstance(obj, list):
        # Process lists
        return [map_class_keys_recursive(item, class_list) for item in obj]
    
    else:
        # Keep other types unchanged
        return obj
    
def convert_numpy(
    obj: Any, 
    convert_keys_to_str: bool = False
) -> Any:
    """
    Convert NumPy objects to native Python types recursively.
    
    Args:
        obj: Object to convert (any type)
        convert_keys_to_str: Convert non-string keys to strings (useful for JSON)
    
    Returns:
        Object converted to native Python types
    """
    # Handle arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle NumPy scalars
    elif isinstance(obj, np.generic):
        if isinstance(obj, (np.integer, np.unsignedinteger)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.character):
            return str(obj)
        return obj.item()  # Fallback for other types
    
    # Handle dictionaries (keys and values)
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            k = convert_numpy(k, convert_keys_to_str)
            v = convert_numpy(v, convert_keys_to_str)
            if convert_keys_to_str and not isinstance(k, str):
                k = str(k)
            new_dict[k] = v
        return new_dict
    
    # Handle lists and tuples
    elif isinstance(obj, list):
        return [convert_numpy(item, convert_keys_to_str) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(item, convert_keys_to_str) for item in obj)
    
    # Other types don't need conversion
    return obj


def check_normalized(value: float, param_name: str = "value") -> None:
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
    >>> check_normalized(0.5)
    >>> check_normalized(1.0)
    
    >>> check_normalized(-0.1)
    Traceback (most recent call last):
    ...
    ValueError: 'threshold' must be between 0.0 and 1.0 (got: -0.1)
    
    >>> check_normalized(1.5, "alpha")
    Traceback (most recent call last):
    ...
    ValueError: 'alpha' must be between 0.0 and 1.0 (got: 1.5)
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"'{param_name}' must be between 0.0 and 1.0 (got: {value})"
        )