import pytest
import numpy as np
from detmet.utils import map_class_keys_recursive, convert_numpy

class TestMapClassKeysRecursive:
    """Test suite for map_class_keys_recursive function."""
    
    def test_flat_dict_mapping(self):
        """Test mapping of numeric keys in flat dictionary."""
        input_dict = {'0': 'cat_value', '1': 'dog_value'}
        class_list = ['cat', 'dog']
        result = map_class_keys_recursive(input_dict, class_list)
        assert result == {'cat': 'cat_value', 'dog': 'dog_value'}

    def test_background_removal(self):
        """Test automatic removal of 'background' class from class_list."""
        input_dict = {'0': 'class0', '1': 'background_value'}
        class_list = ['person', 'car', 'background']
        result = map_class_keys_recursive(input_dict, class_list)
        assert result == {'person': 'class0', 'car': 'background_value'}

    def test_nested_structure_handling(self):
        """Test processing of nested dictionaries and lists."""
        input_obj = {
            'metrics': {
                '0': {'precision': 0.9},
                '1': [{'recall': 0.8}, {'2': 'extra'}]
            }
        }
        class_list = ['cat', 'dog']
        result = map_class_keys_recursive(input_obj, class_list)
        expected = {
            'metrics': {
                'cat': {'precision': 0.9},
                'dog': [{'recall': 0.8}, {'cat': 'extra'}]
            }
        }
        assert result == expected

    def test_non_numeric_key_preservation(self):
        """Test preservation of non-numeric keys."""
        input_dict = {'summary': {'0': 5}, 'other_key': 10}
        class_list = ['class_a']
        result = map_class_keys_recursive(input_dict, class_list)
        assert result == {'summary': {'class_a': 5}, 'other_key': 10}

    def test_insufficient_classes(self):
        """Test handling when class list is shorter than numeric keys."""
        input_dict = {'0': 'A', '1': 'B', '2': 'C'}
        class_list = ['apple', 'banana']
        result = map_class_keys_recursive(input_dict, class_list)
        assert result == {'apple': 'A', 'banana': 'B', '2': 'C'}

    def test_non_ordered_keys(self):
        """Test mapping with non-sequential numeric keys."""
        input_dict = {'2': 'C', '0': 'A', '1': 'B'}
        class_list = ['first', 'second']
        result = map_class_keys_recursive(input_dict, class_list)
        assert result == {'first': 'A', 'second': 'B', '2': 'C'}


class TestConvertNumpy:
    """Test suite for convert_numpy function."""
    
    def test_numpy_array_conversion(self):
        """Test conversion of numpy array to Python list."""
        arr = np.array([1, 2, 3])
        result = convert_numpy(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_scalar_conversion(self):
        """Test conversion of numpy scalars to native Python types."""
        inputs = [
            (np.int64(5), int),
            (np.float32(0.5), float),
            (np.bool_(True), bool),
            (np.bytes_('test'), str)
        ]
        for val, expected_type in inputs:
            result = convert_numpy(val)
            assert isinstance(result, expected_type)

    def test_dict_key_conversion_option(self):
        """
        Test optional string conversion of dictionary keys.
        Should convert non-string keys when convert_keys_to_str=True.
        """
        d = {0: np.int64(5), 1: np.array([1.0])}
        result = convert_numpy(d, convert_keys_to_str=True)
        assert result == {'0': 5, '1': [1.0]}

    def test_nested_structure_conversion(self):
        """Test recursive conversion in nested structures."""
        obj = {
            'arr': np.array([1.5, 2.5]),
            'nested': [np.float32(3.5), {'key': np.bool_(False)}]
        }
        result = convert_numpy(obj)
        expected = {
            'arr': [1.5, 2.5],
            'nested': [3.5, {'key': False}]
        }
        assert result == expected

    def test_non_convertible_preservation(self):
        """Test preservation of non-numpy types."""
        obj = {'name': 'Alice', 'age': 30, 'scores': [85, 90]}
        result = convert_numpy(obj)
        assert result == obj