"""EN: Strict schema shape for Groq structured-output requests.

HU: A Groq structured-output requestek szigorú sémáját ellenőrzi.
"""

from prompt_benchmark.constants import LABELS
from prompt_benchmark.llm.client import GroqClient


def test_groq_structured_output_schema_is_strict():
    """EN: Ensures Groq schema-constrained output uses a strict label enum and disallows unexpected fields.

    HU: Ellenőrzi, hogy a Groq schema-constrained output szigorú label enumot használ és tiltja a váratlan mezőket.
    """
    response_format = GroqClient._response_format()
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["label"]["enum"] == list(LABELS)
