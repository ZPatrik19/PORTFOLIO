from __future__ import annotations

from typing import Any

CATEGORY_LABELS = {
    "quick": {"hu": "Gyors lekérdezések", "en": "Quick lookups"},
    "stay_food": {"hu": "Szállás és étkezés", "en": "Stay & food"},
    "culture_mobility": {"hu": "Látnivalók és közlekedés", "en": "Culture & mobility"},
    "money_budget": {"hu": "Pénz és költségterv", "en": "Money & budget"},
    "multi_tool": {"hu": "Komplex utazástervezés", "en": "Multi-tool planning"},
    "edge": {"hu": "Nehezebb / edge case-ek", "en": "Harder / edge cases"},
}

# The 30 scenarios are intentionally heterogeneous. Each item is bilingual and
# records the expected deterministic tool route so that presets can be regression-tested.
PRESET_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "weather_vienna_3d", "category": "quick", "difficulty": "basic",
        "hu": "Milyen idő lesz Bécsben a következő 3 napban?",
        "en": "What will the weather be like in Vienna for the next 3 days?",
        "expected_tools": ["get_weather"],
    },
    {
        "id": "hotel_prague_budget", "category": "quick", "difficulty": "basic",
        "hu": "Keress Prágában 2 éjszakára 140 euró alatti hotelt.",
        "en": "Find a hotel in Prague for 2 nights under 140 EUR per night.",
        "expected_tools": ["search_hotels"],
    },
    {
        "id": "restaurant_rome_italian", "category": "quick", "difficulty": "basic",
        "hu": "Ajánlj Rómában olasz éttermet 40 euró/fő alatt.",
        "en": "Recommend an Italian restaurant in Rome under 40 EUR per person.",
        "expected_tools": ["search_restaurants"],
    },
    {
        "id": "museums_berlin", "category": "quick", "difficulty": "basic",
        "hu": "Mutass múzeumokat és történelmi látnivalókat Berlinben.",
        "en": "Show me museums and historic attractions in Berlin.",
        "expected_tools": ["search_attractions"],
    },
    {
        "id": "transport_london_4d", "category": "quick", "difficulty": "basic",
        "hu": "Mennyibe kerül a helyi tömegközlekedés 4 napra Londonban?",
        "en": "How much will local public transport cost for 4 days in London?",
        "expected_tools": ["get_transport_options"],
    },
    {
        "id": "fx_eur_huf", "category": "money_budget", "difficulty": "basic",
        "hu": "Mennyi 500 euró forintban?",
        "en": "How much is 500 EUR in HUF?",
        "expected_tools": ["convert_currency"],
    },
    {
        "id": "location_stockholm", "category": "money_budget", "difficulty": "basic",
        "hu": "Milyen pénznem, nyelv és időzóna van Stockholmban?",
        "en": "What currency, language and timezone are used in Stockholm?",
        "expected_tools": ["get_location_info"],
    },
    {
        "id": "vienna_weather_hotel", "category": "stay_food", "difficulty": "intermediate",
        "hu": "3 napra megyek Bécsbe. Nézd meg az időjárást és keress 150 euró alatti hotelt.",
        "en": "I am going to Vienna for 3 days. Check the weather and find a hotel under 150 EUR per night.",
        "expected_tools": ["get_weather", "search_hotels"],
    },
    {
        "id": "barcelona_food_sights", "category": "stay_food", "difficulty": "intermediate",
        "hu": "Barcelonában ajánlj éttermet és mutass látnivalókat is.",
        "en": "Recommend restaurants and attractions in Barcelona.",
        "expected_tools": ["search_attractions", "search_restaurants"],
    },
    {
        "id": "amsterdam_rain_transport", "category": "culture_mobility", "difficulty": "intermediate",
        "hu": "2 napra megyek Amszterdamba. Kell esernyő, és hogyan közlekedjek autó nélkül?",
        "en": "I am visiting Amsterdam for 2 days. Will I need an umbrella, and how can I get around without a car?",
        "expected_tools": ["get_weather", "get_transport_options"],
    },
    {
        "id": "madrid_museums_transport", "category": "culture_mobility", "difficulty": "intermediate",
        "hu": "Milyen múzeumokat érdemes megnézni Madridban, és mennyibe kerül 3 nap tömegközlekedés?",
        "en": "Which museums are worth seeing in Madrid, and what will 3 days of public transport cost?",
        "expected_tools": ["search_attractions", "get_transport_options"],
    },
    {
        "id": "paris_dinner_budget", "category": "stay_food", "difficulty": "intermediate",
        "hu": "Párizsban keress vacsorára éttermet 35 euró/fő alatt.",
        "en": "Find a dinner restaurant in Paris under 35 EUR per person.",
        "expected_tools": ["search_restaurants"],
    },
    {
        "id": "vienna_vegetarian", "category": "stay_food", "difficulty": "intermediate",
        "hu": "Keress Bécsben legalább 4-es értékelésű vegetáriánus éttermet.",
        "en": "Find a vegetarian restaurant in Vienna rated at least 4.0.",
        "expected_tools": ["search_restaurants"],
    },
    {
        "id": "munich_hotel_rating", "category": "stay_food", "difficulty": "intermediate",
        "hu": "Münchenben keress hotelt 170 euró alatt, legalább 4.5-ös értékeléssel, 3 éjszakára.",
        "en": "Find a hotel in Munich under 170 EUR with a rating of at least 4.5 for 3 nights.",
        "expected_tools": ["search_hotels"],
    },
    {
        "id": "rome_attraction_ticket", "category": "culture_mobility", "difficulty": "intermediate",
        "hu": "Rómában mutass történelmi látnivalókat és múzeumokat legfeljebb 25 eurós belépővel.",
        "en": "Show historic attractions and museums in Rome with tickets no more than 25 EUR.",
        "expected_tools": ["search_attractions"],
    },
    {
        "id": "copenhagen_no_car", "category": "culture_mobility", "difficulty": "basic",
        "hu": "Hogyan közlekedjek Koppenhágában 3 napig autó nélkül?",
        "en": "How should I get around Copenhagen for 3 days without a car?",
        "expected_tools": ["get_transport_options"],
    },
    {
        "id": "warsaw_basics_fx", "category": "money_budget", "difficulty": "intermediate",
        "hu": "Varsóba utazom. Mondd meg a helyi pénznemet és nyelvet, majd válts át 200 eurót PLN-re.",
        "en": "I am travelling to Warsaw. Tell me the local currency and language, then convert 200 EUR to PLN.",
        "expected_tools": ["get_location_info", "convert_currency"],
    },
    {
        "id": "zurich_fx_weather", "category": "money_budget", "difficulty": "intermediate",
        "hu": "Zürichbe megyek 2 napra. Milyen idő lesz, és mennyi 300 euró svájci frankban?",
        "en": "I am going to Zurich for 2 days. What will the weather be like, and how much is 300 EUR in CHF?",
        "expected_tools": ["get_weather", "convert_currency"],
    },
    {
        "id": "vienna_local_budget", "category": "money_budget", "difficulty": "advanced",
        "hu": "Készíts 3 napos teljes helyi költségtervet Bécsre szállással és közlekedéssel.",
        "en": "Create a 3-day local budget estimate for Vienna including accommodation and transport.",
        "expected_tools": ["get_location_info", "search_hotels", "get_transport_options", "calculate"],
    },
    {
        "id": "prague_budget_no_hotel", "category": "money_budget", "difficulty": "advanced",
        "hu": "A szállásom már megvan. Készíts 4 napos költségtervet Prágára szállás nélkül.",
        "en": "My accommodation is already booked. Create a 4-day Prague budget without a hotel.",
        "expected_tools": ["get_location_info", "get_transport_options", "calculate"],
    },
    {
        "id": "vienna_full_plan", "category": "multi_tool", "difficulty": "advanced",
        "hu": "3 napra megyek Bécsbe. Nézd meg az időjárást, keress 150 euró alatti hotelt, ajánlj éttermet és látnivalókat, valamint mondd meg a tömegközlekedési lehetőségeket.",
        "en": "I am going to Vienna for 3 days. Check the weather, find a hotel under 150 EUR, recommend restaurants and attractions, and show public transport options.",
        "expected_tools": ["get_weather", "search_hotels", "search_attractions", "search_restaurants", "get_transport_options"],
    },
    {
        "id": "budapest_full_fx", "category": "multi_tool", "difficulty": "advanced",
        "hu": "Tervezz 4 napot Budapestre: időjárás, 120 euró alatti hotel, étterem, múzeumok, közlekedés, és válts át 300 eurót forintra.",
        "en": "Plan 4 days in Budapest: weather, a hotel under 120 EUR, restaurants, museums, transport, and convert 300 EUR to HUF.",
        "expected_tools": ["get_weather", "convert_currency", "search_hotels", "search_attractions", "search_restaurants", "get_transport_options"],
    },
    {
        "id": "lisbon_weekend", "category": "multi_tool", "difficulty": "advanced",
        "hu": "Hétvégére Lisszabonba megyek. Nézd meg az időt, keress hotelt, helyi éttermet és látnivalókat.",
        "en": "I am going to Lisbon for the weekend. Check the weather and find a hotel, local restaurant and attractions.",
        "expected_tools": ["get_weather", "search_hotels", "search_attractions", "search_restaurants"],
    },
    {
        "id": "prague_no_hotel_multi", "category": "edge", "difficulty": "advanced",
        "hu": "Prágába megyek 3 napra, de a szállásom már megvan, ezért ne keress hotelt. Nézd meg az időt, ajánlj látnivalókat és közlekedést.",
        "en": "I am going to Prague for 3 days, but my accommodation is already booked, so do not search for a hotel. Check weather, attractions and transport.",
        "expected_tools": ["get_weather", "search_attractions", "get_transport_options"],
    },
    {
        "id": "vienna_no_accents", "category": "edge", "difficulty": "advanced",
        "hu": "Becsbe megyek 3 napra. Kell esernyo, es tudsz ajanlani ettermet?",
        "en": "I am going to Vienna for three days. Should I pack an umbrella and can you suggest somewhere good to eat?",
        "expected_tools": ["get_weather", "search_restaurants"],
    },
    {
        "id": "reykjavik_pack_weather", "category": "edge", "difficulty": "intermediate",
        "hu": "Reykjavíkba utazom. Milyen időre készüljek, vigyek kabátot?",
        "en": "I am travelling to Reykjavik. What weather should I pack for, and should I bring a jacket?",
        "expected_tools": ["get_weather"],
    },
    {
        "id": "budapest_location_only", "category": "edge", "difficulty": "intermediate",
        "hu": "Budapestre érkezés előtt add meg a legfontosabb helyi alapinformációkat: ország, pénznem, nyelv és időzóna.",
        "en": "Before I arrive in Budapest, give me the key local basics: country, currency, language and timezone.",
        "expected_tools": ["get_location_info"],
    },
    {
        "id": "krakow_food_transport", "category": "edge", "difficulty": "advanced",
        "hu": "Krakkóban 2 napig nem bérelek autót. Hogyan közlekedjek, és hol egyek valami helyit?",
        "en": "I will not rent a car in Krakow for 2 days. How should I get around, and where can I eat something local?",
        "expected_tools": ["search_restaurants", "get_transport_options"],
    },
    {
        "id": "paris_weather_museum_food", "category": "multi_tool", "difficulty": "advanced",
        "hu": "Párizsban 3 napot töltök. Nézd meg az időjárást, ajánlj múzeumokat és egy jó vacsorahelyet.",
        "en": "I am spending 3 days in Paris. Check the weather, recommend museums and a good place for dinner.",
        "expected_tools": ["get_weather", "search_attractions", "search_restaurants"],
    },
    {
        "id": "porto_complete_simple", "category": "multi_tool", "difficulty": "advanced",
        "hu": "Portóban 4 napra keress szállást 130 euró alatt, éttermet, látnivalókat és számold ki a közlekedési bérlet várható költségét.",
        "en": "For 4 days in Porto, find accommodation under 130 EUR, restaurants and attractions, and estimate the public transport pass cost.",
        "expected_tools": ["search_hotels", "search_attractions", "search_restaurants", "get_transport_options"],
    },
]


def preset_categories(language: str = "hu") -> list[tuple[str, str]]:
    lang = "hu" if str(language).lower().startswith("hu") else "en"
    return [(key, value[lang]) for key, value in CATEGORY_LABELS.items()]


def presets_for(category: str | None = None) -> list[dict[str, Any]]:
    if category in (None, "all"):
        return list(PRESET_QUESTIONS)
    return [item for item in PRESET_QUESTIONS if item["category"] == category]


def preset_text(item: dict[str, Any], language: str = "hu") -> str:
    lang = "hu" if str(language).lower().startswith("hu") else "en"
    return str(item[lang])
