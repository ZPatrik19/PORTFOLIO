"""EN: Strict JSON Schema contract used for provider-enforced structured output.

HU: A provider által kikényszerített structured output szigorú JSON Schema szerződését ellenőrzi.
"""

from prompt_benchmark.constants import LABELS
from prompt_benchmark.llm.client import OpenAIResponsesClient


def test_structured_output_schema_is_strict_and_label_constrained():
    """EN: Ensures the OpenAI-style JSON Schema allows only the six known labels and rejects extra properties.

    HU: Ellenőrzi, hogy az OpenAI-szerű JSON Schema csak a hat ismert labelt engedi és tiltja az extra mezőket.
    """
    text_config = OpenAIResponsesClient._schema_format()
    schema_format = text_config["format"]
    assert schema_format["type"] == "json_schema"
    assert schema_format["strict"] is True
    assert schema_format["schema"]["properties"]["label"]["enum"] == list(LABELS)
    assert schema_format["schema"]["additionalProperties"] is False
