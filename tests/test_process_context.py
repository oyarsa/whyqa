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


def test_empty_input():
    assert process_context("") == []


def test_results_with_multiple_lines_per_entry():
    context = """(Multi-line Entry) This is the first line.
This is the second line.
This is the third line.
(Next Entry) This is a single line entry."""
    expected = [
        "(Multi-line Entry) This is the first line.\nThis is the second line.\nThis is the third line.",
        "(Next Entry) This is a single line entry.",
    ]
    assert process_context(context) == expected


def test_results_with_special_characters():
    context = """(Special: !@#$%^&*) Content with special chars: !@#$%^&*.
(Números en español) Contenido en español: áéíóú."""
    expected = [
        "(Special: !@#$%^&*) Content with special chars: !@#$%^&*.",
        "(Números en español) Contenido en español: áéíóú.",
    ]
    assert process_context(context) == expected


def test_results_with_urls():
    context = """(Web Content) Check out this URL: https://www.example.com/page?param=value.
(Another Entry) More content here."""
    expected = [
        "(Web Content) Check out this URL: https://www.example.com/page?param=value.",
        "(Another Entry) More content here.",
    ]
    assert process_context(context) == expected
