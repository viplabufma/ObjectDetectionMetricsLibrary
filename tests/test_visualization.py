import pytest
from unittest.mock import mock_open, patch
import numpy as np

# Mock input data
@pytest.fixture
def mock_pr_curves():
    """Fixture providing minimal valid PR curves data for testing"""
    return {
        'global': {
            'recall': np.array([0.0, 1.0]),
            'precision': np.array([1.0, 0.5]),
            'ap': 0.75
        },
        1: {
            'recall': np.array([0.0, 0.5, 1.0]),
            'precision': np.array([1.0, 0.8, 0.4]),
            'ap': 0.7
        }
    }

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
def test_plot_pr_curves_show_true(mock_savefig, mock_show, mock_pr_curves):
    """
    Test the execution path when show=True:
    - Should call plt.show()
    - Should return None
    - Should not save file (output_path=None)
    """
    from detmet.visualization import plot_pr_curves

    # Execute function with show=True
    result = plot_pr_curves(mock_pr_curves, show=True)
    
    # Verify calls and return value
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()
    assert result is None

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
def test_plot_pr_curves_show_true_with_output(mock_savefig, mock_show, mock_pr_curves):
    """
    Test execution path with show=True and output_path:
    - Should call plt.show()
    - Should save the file
    - Should return None
    """
    from detmet.visualization import plot_pr_curves

    # Execute function with show=True and output_path
    result = plot_pr_curves(mock_pr_curves, output_path="test.png", show=True)
    
    # Verify calls and return value
    mock_show.assert_called_once()
    mock_savefig.assert_called_once_with("test.png", dpi=100, bbox_inches='tight')
    assert result is None

@patch('detmet.visualization.plt.savefig')
@patch('detmet.visualization.plt.close')
@patch('detmet.visualization.sns.heatmap')
def test_save_confusion_matrix_with_background(mock_heatmap, mock_close, mock_savefig):
    """
    Test that background class is appended when background_class=True
    - Should modify class_names by appending "background"
    - Should pass updated class_names to sns.heatmap
    - Should maintain original class_names list length when background_class=False
    """
    from detmet.visualization import save_confusion_matrix
    
    # Test data
    matrix = [[10, 2], [3, 15]]
    original_class_names = ['cat', 'dog']
    
    # Test with background_class=True
    save_confusion_matrix(matrix, original_class_names, background_class=True)
    
    # Verify background was appended
    assert original_class_names == ['cat', 'dog', 'background']
    
    # Verify heatmap called with updated class_names
    mock_heatmap.assert_called_once()
    call_args = mock_heatmap.call_args[1]
    assert call_args['xticklabels'] == ['cat', 'dog', 'background']
    assert call_args['yticklabels'] == ['cat', 'dog', 'background']
    
    # Reset mocks for next test
    mock_heatmap.reset_mock()
    
    # Test with background_class=False
    class_names_copy = ['cat', 'dog'].copy()
    save_confusion_matrix(matrix, class_names_copy, background_class=False)
    
    # Verify class_names unchanged
    assert class_names_copy == ['cat', 'dog']
    
    # Verify heatmap called with original class_names
    call_args = mock_heatmap.call_args[1]
    assert call_args['xticklabels'] == ['cat', 'dog']
    assert call_args['yticklabels'] == ['cat', 'dog']

def test_plot_pr_curves_empty_input():
    """
    Test that ValueError is raised when empty pr_curves dictionary is passed
    - Should raise ValueError with specific message
    - Should not attempt to create any plot elements
    """
    from detmet.visualization import plot_pr_curves
    
    # Test with empty dictionary
    with pytest.raises(ValueError) as excinfo:
        plot_pr_curves({})
    
    # Verify error message
    assert "PR curves dictionary is empty" in str(excinfo.value)
    
    # Test with None input
    with pytest.raises(ValueError) as excinfo:
        plot_pr_curves(None)
    
    # Verify error message
    assert "PR curves dictionary is empty" in str(excinfo.value)
    
    # Test with empty dictionary and other parameters
    with pytest.raises(ValueError) as excinfo:
        plot_pr_curves({}, output_path="test.png", show=False)
    
    # Verify error message
    assert "PR curves dictionary is empty" in str(excinfo.value)

@patch('detmet.visualization.os.makedirs')
@patch('detmet.visualization.open', new_callable=mock_open)
def test_export_metrics_unsupported_format(mock_file, mock_makedirs):
    """
    Test that ValueError is raised for unsupported export formats
    - Should raise ValueError with specific message
    - Should not attempt to write any file
    - Should still create output directory
    """
    from detmet.visualization import export_metrics
    
    # Test metrics data
    metrics = {
        'precision': 0.85,
        'recall': 0.78,
        'f1_score': 0.81
    }
    
    # Test with unsupported format
    with pytest.raises(ValueError) as excinfo:
        export_metrics(metrics, format='yaml')
    
    # Verify error message
    assert "Unsupported format. Use 'json'." in str(excinfo.value)
    
    # Verify directory creation was attempted
    mock_makedirs.assert_called_once_with('.', exist_ok=True)
    
    # Verify no file write was attempted
    mock_file().write.assert_not_called()
    
    # Test with another unsupported format
    mock_makedirs.reset_mock()
    with pytest.raises(ValueError) as excinfo:
        export_metrics(metrics, output_path='output', format='xml')
    
    # Verify error message
    assert "Unsupported format. Use 'json'." in str(excinfo.value)
    
    # Verify directory creation was attempted for custom path
    mock_makedirs.assert_called_once_with('output', exist_ok=True)

