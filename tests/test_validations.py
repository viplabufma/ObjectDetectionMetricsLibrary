import numpy as np
from detmet.validations import validate_normalized
from detmet.utils import convert_numpy
import pytest

class TestCheckNormalized:
    """Test suite for validate_normalized function."""
    
    def test_valid_values(self):
        """Test values within [0,1] range. Should not raise errors."""
        for value in [0.0, 0.5, 1.0]:
            # Should not raise exception
            validate_normalized(value)
    
    def test_invalid_values(self):
        """Test values outside [0,1] range. Should raise ValueError."""
        test_cases = [-0.1, 1.1, 2.5, -1.0]
        for value in test_cases:
            with pytest.raises(ValueError) as excinfo:
                validate_normalized(value)
            assert "must be between 0.0 and 1.0" in str(excinfo.value)
    
    def test_custom_param_name(self):
        """Test custom parameter name in error message."""
        with pytest.raises(ValueError) as excinfo:
            validate_normalized(1.5, "threshold")
        assert "'threshold'" in str(excinfo.value)
        assert "(got: 1.5)" in str(excinfo.value)

    def test_numpy_fallback_conversion(self):
        """Test fallback conversion for other numpy types using item()."""
        dt = np.datetime64('2025-07-02')
        result = convert_numpy(dt)
        assert result == dt.item()
        assert not isinstance(result, np.generic)

    def test_tuple_conversion(self):
        """Test recursive conversion of tuples with various numpy types."""
        # Tuple with different NumPy types
        obj = (
            np.array([1, 2, 3]),       # Array
            np.float32(0.5),            # Float scalar
            np.bool_(True),             # Boolean
            {0: np.int64(10)},          # Dictionary with integer key
            [np.datetime64('2025-01-01')]  # List with date
        )
        
        # Test without key conversion
        result = convert_numpy(obj)
        expected = (
            [1, 2, 3],     # Array converted to list
            0.5,            # Float converted
            True,           # Boolean converted
            {0: 10},        # Dictionary with integer key preserved
            [np.datetime64('2025-01-01').item()]  # Date converted
        )
        assert result == expected
        assert isinstance(result, tuple)
        
        # Test with key conversion enabled
        result = convert_numpy(obj, convert_keys_to_str=True)
        expected = (
            [1, 2, 3],
            0.5,
            True,
            {'0': 10},  # Key converted to string
            [np.datetime64('2025-01-01').item()]
        )
        assert result == expected