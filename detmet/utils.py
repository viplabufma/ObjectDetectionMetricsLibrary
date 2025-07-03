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

def vectorized_bbox_iou(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between multiple ground truth and predicted bounding boxes.

    Parameters
    ----------
    gt_boxes : np.ndarray
        [N, 4] array of ground truth boxes as [x1, y1, x2, y2]
    pred_boxes : np.ndarray
        [M, 4] array of predicted boxes as [x1, y1, x2, y2]

    Returns
    -------
    np.ndarray
        [N, M] matrix of IoU values

    Notes
    -----
    - Handles empty input arrays by returning a zero matrix
    - Uses broadcasting for efficient computation
    - Returns 0.0 for boxes with no overlap

    Examples
    --------
    >>> gt = np.array([[10, 10, 20, 20]])
    >>> pred = np.array([[15, 15, 25, 25]])
    >>> iou_matrix = vectorized_bbox_iou(gt, pred)
    """
    # Ensure arrays are 2D: (num_boxes, 4)
    gt_boxes = gt_boxes.reshape(-1, 4)
    pred_boxes = pred_boxes.reshape(-1, 4)

    N = gt_boxes.shape[0]
    M = pred_boxes.shape[0]

    # Return zero matrix if no boxes
    if N == 0 or M == 0:
        return np.zeros((N, M))

    # Expand dimensions for broadcasting
    gt_boxes = gt_boxes[:, None, :]  # shape (N, 1, 4)
    pred_boxes = pred_boxes[None, :, :]  # shape (1, M, 4)

    # Calculate intersection coordinates
    x1_inter = np.maximum(gt_boxes[..., 0], pred_boxes[..., 0])
    y1_inter = np.maximum(gt_boxes[..., 1], pred_boxes[..., 1])
    x2_inter = np.minimum(gt_boxes[..., 2], pred_boxes[..., 2])
    y2_inter = np.minimum(gt_boxes[..., 3], pred_boxes[..., 3])

    # Calculate intersection area
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Calculate union area
    gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = gt_area + pred_area - inter_area

    return inter_area / np.maximum(union_area, 1e-7)

def compute_iou_matrix(gt_anns: List[dict], pred_anns: List[dict], convert_to_xyxy: bool = True) -> np.ndarray:
    """
    Compute IoU matrix between ground truth and predicted annotations.

    Parameters
    ----------
    gt_anns : List[dict]
        List of ground truth annotations, each containing 'bbox' in [x, y, w, h] or [x1, y1, x2, y2]
    pred_anns : List[dict]
        List of predicted annotations, each containing 'bbox' in [x, y, w, h] or [x1, y1, x2, y2]
    convert_to_xyxy : bool, optional
        If True, convert bboxes from [x, y, w, h] to [x1, y1, x2, y2]. If False, assume bboxes
        are already in [x1, y1, x2, y2], by default True

    Returns
    -------
    np.ndarray
        IoU matrix of shape (num_gt, num_pred)

    Examples
    --------
    >>> gt = [{'bbox': [10, 10, 20, 20]}]
    >>> pred = [{'bbox': [12, 12, 18, 18]}]
    >>> iou_matrix = compute_iou_matrix(gt, pred)
    """
    if not gt_anns and not pred_anns:
        return np.zeros((0, 0))

    gt_boxes = np.array([g['bbox'] for g in gt_anns], dtype=np.float32)
    pred_boxes = np.array([p['bbox'] for p in pred_anns], dtype=np.float32)

    if convert_to_xyxy and gt_boxes.size > 0:
        gt_boxes[:, 2] += gt_boxes[:, 0]  # x2 = x + w
        gt_boxes[:, 3] += gt_boxes[:, 1]  # y2 = y + h
    if convert_to_xyxy and pred_boxes.size > 0:
        pred_boxes[:, 2] += pred_boxes[:, 0]  # x2 = x + w
        pred_boxes[:, 3] += pred_boxes[:, 1]  # y2 = y + h

    return vectorized_bbox_iou(gt_boxes, pred_boxes)