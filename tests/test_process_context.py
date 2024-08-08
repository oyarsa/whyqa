import pytest

from whyqa.preprocess.process_causalqa import process_context


def test_single_result():
    context = "(Sample Title) Jun 15, 2023 This is the content of a single result."
    expected = ["(Sample Title) Jun 15, 2023 This is the content of a single result."]
    assert process_context(context) == expected


def test_multiple_results():
    context = """(First Title) Jan 1, 2023 Content of first result.
(Second Title) Feb 2, 2023 Content of second result.
(Third Title) Mar 3, 2023 Content of third result."""
    expected = [
        "(First Title) Jan 1, 2023 Content of first result.",
        "(Second Title) Feb 2, 2023 Content of second result.",
        "(Third Title) Mar 3, 2023 Content of third result.",
    ]
    assert process_context(context) == expected


def test_results_with_empty_lines():
    context = """(Title One) Apr 4, 2023 First content.

(Title Two) May 5, 2023 Second content.

(Title Three) Jun 6, 2023 Third content."""
    expected = [
        "(Title One) Apr 4, 2023 First content.",
        "(Title Two) May 5, 2023 Second content.",
        "(Title Three) Jun 6, 2023 Third content.",
    ]
    assert process_context(context) == expected


@pytest.mark.xfail(
    reason="TODO: Function doesn't correctly handle results without newlines"
)
def test_results_with_nested_parentheses():
    context = "(Complex (Nested) Title) Jul 7, 2023 Content with (parentheses).(Another (Tricky) One) Aug 8, 2023 More (complex) content."
    expected = [
        "(Complex (Nested) Title) Jul 7, 2023 Content with (parentheses).",
        "(Another (Tricky) One) Aug 8, 2023 More (complex) content.",
    ]
    assert process_context(context) == expected


def test_empty_input():
    assert process_context("") == []


def test_input_without_valid_headers():
    context = "This is just some text without any valid headers."
    expected = ["This is just some text without any valid headers."]
    assert process_context(context) == expected
