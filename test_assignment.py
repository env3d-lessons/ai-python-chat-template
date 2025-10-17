import pytest
from unittest.mock import patch
from main import complete_least_likely, complete_most_likely, complete_random, custom_template_chat

@pytest.fixture
def mock_get_top_tokens():
    return [
        (" Ottawa", 0.76),
        (" Toronto", 0.11),
        (" Montreal", 0.05),
        (" Vancouver", 0.03),
        (" Calgary", 0.02),
    ]

@patch("main.get_top_tokens")
def test_complete_least_likely(mock_get_top_tokens_func, mock_get_top_tokens):
    mock_get_top_tokens_func.return_value = mock_get_top_tokens
    result = complete_least_likely("The capital of Canada is", max_tokens=1)
    assert result == "The capital of Canada is Calgary"

@patch("main.get_top_tokens")
def test_complete_most_likely(mock_get_top_tokens_func, mock_get_top_tokens):
    mock_get_top_tokens_func.return_value = mock_get_top_tokens
    result = complete_most_likely("The capital of Canada is", max_tokens=1)
    assert result == "The capital of Canada is Ottawa"

@patch("main.get_top_tokens")
def test_complete_random(mock_get_top_tokens_func, mock_get_top_tokens):
    mock_get_top_tokens_func.return_value = mock_get_top_tokens
    result = complete_random("The capital of Canada is", max_tokens=1)
    assert result.startswith("The capital of Canada is ")
    assert result.split()[-1] in [token[0].strip() for token in mock_get_top_tokens]

@patch("main.complete")
def test_custom_template_chat(mock_complete_func):
    mock_complete_func.return_value = "Hello, how can I help you?"
    prompt = "Hello"
    result = custom_template_chat(prompt)

    # Extract the argument passed to mock_complete_func
    args, kwargs = mock_complete_func.call_args
    template_argument = args[0]  # The first positional argument

    # Assert the template was formatted correctly
    original_template = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

    assert template_argument != original_template