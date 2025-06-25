from typing import Any, List, Union
import numpy as np

def map_class_keys_recursive(
    obj: Union[dict, list, Any], 
    class_list: List[str]
) -> Union[dict, list, Any]:
    """
    Recursively maps numeric keys to class names throughout all dictionary levels.
    
    This function processes nested data structures, converting integer keys to corresponding
    class names based on the provided class list. Useful for making metrics dictionaries
    more readable by replacing numeric class IDs with human-readable names.
    
    Parameters
    ----------
    obj : Union[dict, list, Any]
        The object to process. Can be a dictionary, list, or any other type.
    class_list : List[str]
        List of class names in index order. The last element is removed if it's 'background'.
    
    Returns
    -------
    Union[dict, list, Any]
        Processed object with numeric keys replaced by class names where possible.
    
    Notes
    -----
    - The function removes the last element of class_list if it's 'background'
    - Numeric keys are mapped to class names by index (0 -> class_list[0])
    - Non-numeric keys and values are preserved
    - Non-mappable numeric keys (index >= len(class_list)) are kept as original
    - Recursively processes nested dictionaries and lists
    
    Examples
    --------
    >>> class_list = ['cat', 'dog']
    >>> input_dict = {'0': {'precision': 0.9}, '1': {'recall': 0.8}, 'other': 'value'}
    >>> map_class_keys_recursive(input_dict, class_list)
    {'cat': {'precision': 0.9}, 'dog': {'recall': 0.8}, 'other': 'value'}
    
    >>> input_list = [{'0': 1}, {'1': 2}]
    >>> map_class_keys_recursive(input_list, class_list)
    [{'cat': 1}, {'dog': 2}]
    """
    if class_list and class_list[-1] == 'background':
        class_list = class_list[:-1]
    
    if isinstance(obj, dict):
        new_dict = {}
        numeric_keys = []
        non_numeric_items = {}
        
        for key, value in obj.items():
            try:
                num_key = int(key)
                numeric_keys.append((key, num_key, value))
            except (ValueError, TypeError):
                non_numeric_items[key] = map_class_keys_recursive(value, class_list)
        
        numeric_keys.sort(key=lambda x: x[1])
        
        for idx, (orig_key, num_key, value) in enumerate(numeric_keys):
            new_key = class_list[idx] if idx < len(class_list) else orig_key
            new_dict[new_key] = map_class_keys_recursive(value, class_list)
        
        new_dict.update(non_numeric_items)
        return new_dict
    
    elif isinstance(obj, list):
        return [map_class_keys_recursive(item, class_list) for item in obj]
    
    else:
        return obj
    
def convert_numpy(
    obj: Any, 
    convert_keys_to_str: bool = False
) -> Any:
    """
    Convert NumPy objects to native Python types recursively.
    
    This function is particularly useful when preparing data for JSON serialization,
    as NumPy types are not JSON-serializable by default.
    
    Parameters
    ----------
    obj : Any
        The object to convert. Can be any Python object.
    convert_keys_to_str : bool, optional
        Whether to convert non-string dictionary keys to strings, by default False.
        Useful for JSON serialization which requires string keys.
    
    Returns
    -------
    Any
        Object with all NumPy types converted to native Python types.
    
    Notes
    -----
    Handles the following conversions:
    - NumPy arrays -> Python lists
    - NumPy integers -> Python int
    - NumPy floats -> Python float
    - NumPy booleans -> Python bool
    - NumPy strings -> Python str
    - Other NumPy types -> Python equivalent using item()
    - Recursively processes dictionaries, lists, and tuples
    
    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 3])
    >>> convert_numpy(arr)
    [1, 2, 3]
    
    >>> d = {'key': np.float32(0.5), 'nested': [np.bool_(True)]}
    >>> convert_numpy(d)
    {'key': 0.5, 'nested': [True]}
    
    >>> d = {0: np.int64(5), 1: np.array([1.0])}
    >>> convert_numpy(d, convert_keys_to_str=True)
    {'0': 5, '1': [1.0]}
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