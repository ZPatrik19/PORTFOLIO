"""EN: Plain-label and JSON response parsing into the common classification contract.

HU: A sima label- és JSON-válaszok egységes klasszifikációs szerződésbe történő parsingját ellenőrzi.
"""

from prompt_benchmark.evaluation.parsing import parse_prediction


def test_plain_label_parsing():
    """EN: Checks parsing of a simple one-label model response.

    HU: Ellenőrzi egy egyszerű egyszavas label válasz feldolgozását.
    """
    assert parse_prediction("billing", "label").label == "billing"
    assert not parse_prediction("The answer is billing", "label").valid_output


def test_json_parsing():
    """EN: Checks parsing and validation of JSON-formatted classification output.

    HU: Ellenőrzi a JSON-formátumú klasszifikációs output parsingját és validációját.
    """
    parsed = parse_prediction('{"label":"technical"}', "json")
    assert parsed.label == "technical" and parsed.valid_json and parsed.valid_output
    assert not parse_prediction("label: technical", "json").valid_json
