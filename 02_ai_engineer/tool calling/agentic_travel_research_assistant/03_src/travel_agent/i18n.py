from __future__ import annotations

SUPPORTED_LANGUAGES = {"en", "hu"}

_TRANSLATIONS = {
    "en": {
        "destination": "Destination",
        "weather": "Weather",
        "currency": "Currency",
        "hotel": "Hotels",
        "attractions": "Attractions",
        "transport": "Transport",
        "budget": "Budget",
        "no_tool": "No external tool is needed for this request. Try asking about weather, hotels, currency conversion, destination information, or a trip budget.",
        "stopped": "Agent stopped after reaching the maximum tool-call step limit.",
    },
    "hu": {
        "destination": "Úticél",
        "weather": "Időjárás",
        "currency": "Devizaátváltás",
        "hotel": "Szállás",
        "attractions": "Látnivalók",
        "transport": "Közlekedés",
        "budget": "Költségterv",
        "no_tool": "Ehhez a kéréshez nincs szükség külső toolra. Kérdezz időjárásról, szállásról, devizaátváltásról, úti cél információról vagy utazási költségtervről.",
        "stopped": "Az agent leállt, mert elérte a maximális tool-hívási lépésszámot.",
    },
}


def normalize_language(language: str | None) -> str:
    value = (language or "en").strip().lower()
    return value if value in SUPPORTED_LANGUAGES else "en"


def t(key: str, language: str = "en") -> str:
    lang = normalize_language(language)
    return _TRANSLATIONS[lang].get(key, _TRANSLATIONS["en"].get(key, key))


def language_instruction(language: str) -> str:
    lang = normalize_language(language)
    if lang == "hu":
        return "Write the final user-facing answer in Hungarian. Tool names and tool arguments remain in their canonical English schema."
    return "Write the final user-facing answer in English."
