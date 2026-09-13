from travel_agent.agent.heuristics import extract_city


def test_extract_city_standalone_hungarian_alias():
    assert extract_city("Bécs nagyon szép.") == "Vienna"


def test_extract_city_with_hungarian_direction_suffix():
    assert extract_city("3 napra megyek Bécsbe.") == "Vienna"


def test_extract_city_with_hungarian_inessive_suffix():
    assert extract_city("Budapesten keresek éttermet.") == "Budapest"


def test_extract_city_with_vowel_lengthening_before_suffix():
    assert extract_city("Rómába utazom négy napra.") == "Rome"


def test_extract_city_with_source_suffix():
    assert extract_city("Londonból utazom tovább.") == "London"

from travel_agent.agent.heuristics import detect_intents, hotel_constraints, restaurant_constraints


def test_hungarian_umbrella_maps_to_weather():
    assert detect_intents("Bécsbe megyek. Kell esernyő?")["weather"] is True


def test_heldout_lodging_word_maps_to_hotel_and_budget():
    q = "I still need lodging in Vienna for 4 nights, with a budget of 155 EUR per night."
    assert detect_intents(q)["hotel"] is True
    price, _, _ = hotel_constraints(q)
    assert price == 155.0


def test_heldout_food_wording_maps_to_restaurant_and_amount():
    q = "Where would you go for food in Vienna with roughly 38 EUR a head?"
    assert detect_intents(q)["restaurant"] is True
    _, max_meal, _, _, _ = restaurant_constraints(q)
    assert max_meal == 38.0


def test_hungarian_local_travel_wording_maps_to_transport():
    assert detect_intents("Tervezd meg a helyi utazásomat Bécsben 3 napra.")["transport"] is True
