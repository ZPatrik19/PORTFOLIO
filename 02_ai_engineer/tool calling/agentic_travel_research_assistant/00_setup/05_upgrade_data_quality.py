"""Regenerate the synthetic travel corpus with substantially higher lexical diversity.

Why this exists
---------------
The original large-data version had many rows but too few underlying templates. That
can create template leakage and unrealistically high classifier scores. This script
keeps the reproducible synthetic/offline nature of the project while making the
entity names and language corpus much more heterogeneous.

It is deterministic: running it with the same seed reproduces the same files.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_data" / "raw"
BENCH = ROOT / "01_data" / "benchmark"
PROCESSED = ROOT / "01_data" / "processed"
SEED = 20260912
rng = random.Random(SEED)

HU_NAMES = {"Vienna":"Bécs","Prague":"Prága","Paris":"Párizs","Rome":"Róma","Munich":"München","Krakow":"Krakkó"}
INTENTS = ["weather", "hotel", "attractions", "restaurant", "transport", "location"]


def _pick(seq, key: int):
    return seq[key % len(seq)]


def improve_entity_names() -> None:
    hotels = pd.read_csv(RAW / "hotels.csv")
    attractions = pd.read_csv(RAW / "attractions.csv")
    restaurants = pd.read_csv(RAW / "restaurants.csv")

    hotel_adjectives = ["Amber","Atlas","Aurora","Bluebird","Canopy","Cedar","Copper","Crown","Dawn","Ember","Evergreen","Fern","Golden","Harbour","Ivory","Juniper","Lantern","Laurel","Linden","Marble","Meadow","Meridian","Moonlight","Northlight","Olive","Orchid","Pearl","Pine","Raven","Rosewood","Saffron","Silver","Solstice","Stone","Sunrise","Terrace","Velvet","Willow","Wren","Arcade","Belvedere","Canvas","Courtyard","Foundry","Gallery","Hearth","Mosaic","Pavilion","Quayside","Regent","Riverview","Vantage","Voyager"]
    hotel_nouns = ["Courtyard","Lantern","Compass","Bridge","Garden","Square","House","Atelier","Loft","Quay","Terrace","Arcade","Harbour","Hearth","Pavilion","Gallery","Grove","Station","Canal","Crescent","Mews","Passage","Promenade","Vista","Court","Exchange","Foundry","Oasis","Parlour","Retreat","Sanctuary","Workshop","Manor","Hall","Corner","Quarter","Boulevard","Hideaway","Gateway","Nest","Roost","Atrium"]
    hotel_types = ["Hotel","Suites","Residence","Rooms","House","Inn","Apartments","Lodge","Boutique Hotel","City Stay","Guesthouse","Studios"]
    hnames=[]
    for i,row in hotels.iterrows():
        a=hotel_adjectives[i % len(hotel_adjectives)]
        n=hotel_nouns[(i // len(hotel_adjectives)) % len(hotel_nouns)]
        t=hotel_types[(i // (len(hotel_adjectives)*len(hotel_nouns))) % len(hotel_types)]
        mode=i%5
        if mode==0: name=f"The {a} {n} {t} · {row.city}"
        elif mode==1: name=f"{a} & {n} {t} · {row.city}"
        elif mode==2: name=f"{n} {t} by {a} · {row.city}"
        elif mode==3: name=f"{a} {row.area} {t} · {row.city}"
        else: name=f"{row.area} {a} {n} · {row.city}"
        hnames.append(name)
    hotels["name"]=hnames
    hotels.to_csv(RAW/"hotels.csv",index=False)

    themes = ["Modern Art","City History","Natural History","Design","Photography","Science","Railway","Maritime","Architecture","Music","Literature","Craft","Ceramics","Textiles","Local Life","Industrial Heritage","Contemporary Culture","Archaeology","Aviation","Technology","Typography","Riverside","Botanical","Urban Ecology","Sculpture","Folk Culture","Migration","Cinema","Food Heritage","Innovation","Public Art","Transport","Astronomy","Medicine","Postal History","Performing Arts","Children's Discovery","Geology","Fashion","Graphic Arts","City Memory","Local Makers","Cartography","Printmaking","Ecology","Exploration","Engineering","Decorative Arts"]
    prefixes=["North","South","Old","New","Royal","Civic","Riverside","Hilltop","Central","East","West","Grand","Hidden","Green","Upper","Lower","Harbour","Market","Garden","Canal"]
    areas=["Old Town","Riverside","Museum Quarter","Market Quarter","Parkside","University Quarter","Harbour","City Centre","Garden District","Station Quarter","Arts Quarter","Canal Side"]
    # Additional deterministic micro-location signatures keep the 90k inventory
    # lexically diverse without adding meaningless numeric suffixes.
    sig_left = [
        "Alder","Ash","Birch","Briar","Brook","Cedar","Clover","Copper","Elm","Fern","Flint","Garnet","Hazel","Iris","Juniper","Larch",
        "Laurel","Linden","Maple","Moss","Oak","Olive","Orchid","Pearl","Pine","Plum","Reed","Rose","Rowan","Sage","Silver","Spruce",
        "Stone","Thistle","Violet","Walnut","Willow","Wren","Amber","Aurora","Canvas","Cobalt","Coral","Dawn","Ember","Frost","Golden","Harbour",
        "Indigo","Ivory","Lantern","Marble","Meadow","Meridian","Moon","North","Opal","Raven","Saffron","Solstice","Sunrise","Velvet","West","Zephyr"
    ]
    sig_right = [
        "Arcade","Arch","Basin","Bridge","Canal","Court","Crescent","Crossing","Dock","Exchange","Garden","Gate","Grove","Hall","Harbour","Heights",
        "Hill","Lane","Market","Mews","Passage","Pavilion","Pier","Plaza","Promenade","Quay","Quarter","Row","Square","Terrace","Walk","Wharf",
        "Atrium","Boulevard","Commons","Corner","Courtyard","Esplanade","Foundry","Gallery","Green","Landing","Loft","Orchard","Parade","Point","Portico","Ridge",
        "Rise","Steps","Studio","Tower","Trail","Vale","View","Yard","Circle","Forum","Junction","Park","Reach","Station","Vista","Way"
    ]
    pnames=[]
    for i,row in attractions.iterrows():
        t1=themes[(i*7)%len(themes)]
        t2=themes[((i//len(themes))*11+17)%len(themes)]
        pref=prefixes[(i*13 + i//97)%len(prefixes)]
        area=areas[(i*17 + i//53)%len(areas)]
        signature=f"{sig_left[i % len(sig_left)]} {sig_right[(i // len(sig_left)) % len(sig_right)]}"
        typ=str(row.category).replace('_',' ').title()
        mode=i%8
        if mode==0:name=f"{signature} {typ} of {t1}"
        elif mode==1:name=f"{t1} & {t2} {typ} at {signature}"
        elif mode==2:name=f"{area} {signature} {typ}: {t1}"
        elif mode==3:name=f"{pref} {t1} {typ} — {signature}"
        elif mode==4:name=f"{t1} {typ} at {signature}"
        elif mode==5:name=f"{signature} {typ} of {t1} and {t2}"
        elif mode==6:name=f"{pref} {signature} {t1} {typ}"
        else:name=f"{signature} {t2} {typ} in {area}"
        pnames.append(name)
    attractions["name"]=pnames
    attractions.to_csv(RAW/"attractions.csv",index=False)

    food_words=["Apron","Basil","Bowl","Brick","Canteen","Caraway","Cedar","Copper","Coriander","Fig","Fork","Garden","Grain","Hearth","Juniper","Ladle","Lemon","Market","Mint","Olive","Orchard","Oven","Pantry","Pepper","Plate","Rosemary","Saffron","Salt","Skillet","Spoon","Table","Thyme","Walnut","Willow","Ember","Flour","Kettle","Lantern","Mill","Mosaic","Nook","Oak","Pear","Poppy","Quince","Stone","Vine","Yard","Fennel","Clove","Cinnamon","Barley","Honey","Maple","Sage"]
    food_mod=["Little","Green","Golden","Local","Old","New","Wild","Urban","Neighbourhood","Riverside","Corner","Daily","Shared","Open","Honest","Slow","Bright","Red","Blue","North","South","East","West","Tiny","Common","Sunday","Market","Garden","Copper","Stone"]
    forms=["Kitchen","Table","Bistro","Café","Eatery","Dining Room","Kitchen & Bar","Cantina","Trattoria","Grill","Deli","Noodle House","Tavern","Bakery","Supper Club","Brasserie","Counter","Food Lab"]
    rnames=[]
    for i,row in restaurants.iterrows():
        word=food_words[i%len(food_words)]
        word2=food_words[(i//len(food_words)+7)%len(food_words)]
        mod=food_mod[(i//13)%len(food_mod)]
        form=forms[(i//17)%len(forms)]
        mode=i%6
        if mode==0:name=f"{mod} {word} {form} · {row.city}"
        elif mode==1:name=f"{word} & {word2} · {row.city}"
        elif mode==2:name=f"The {word} {form} · {row.city}"
        elif mode==3:name=f"{row.area} {word} {form} · {row.city}"
        elif mode==4:name=f"{mod} {word} & {word2} {form} · {row.city}"
        else:name=f"{word} House by {mod} · {row.city}"
        rnames.append(name)
    restaurants["name"]=rnames
    restaurants.to_csv(RAW/"restaurants.csv",index=False)


EN = {
    "train": {
        "weather": ["What's the weather looking like in {city} for {days} days?", "Check the forecast in {city} for my {days}-day stay.", "Will I need an umbrella in {city} over the next {days} days?", "Give me temperatures and rain chances for {city} for {days} days.", "I'm packing for {city}; what conditions should I expect for {days} days?", "Is {city} likely to be rainy during a {days}-day visit?"],
        "hotel": ["Find somewhere to stay in {city} for {days} nights under {price} EUR per night.", "I need a {days}-night hotel in {city}; keep it below {price} EUR and rating {rating}+.", "Show accommodation in {city} for {days} nights with a nightly cap of {price} EUR.", "Help me shortlist hotels in {city}: {days} nights, max {price} EUR/night.", "Where can I sleep in {city} for {days} nights without going over {price} EUR a night?", "Look for well-rated rooms in {city} for {days} nights, preferably below {price} EUR."],
        "attractions": ["What should I see in {city}?", "Build a sightseeing list for {city}, especially museums and landmarks.", "Suggest places worth visiting in {city} with tickets under {ticket} EUR.", "I want culture and interesting sights in {city}; what do you recommend?", "Fill half a day in {city} with attractions.", "Find museums, parks or historic places in {city}."],
        "restaurant": ["Where should I eat in {city} for about {meal} EUR per person?", "Find good {cuisine} restaurants in {city} under {meal} EUR each.", "Give me dinner options in {city} with a budget around {meal} EUR.", "I need food recommendations in {city}; keep meals below {meal} EUR.", "Suggest somewhere good to eat in {city}.", "Find restaurants in {city}, preferably {cuisine}, rated at least {rating}."],
        "transport": ["How should I get around {city} for {days} days?", "Compare local public-transport tickets in {city} for a {days}-day trip.", "What transit pass makes sense in {city} for {days} days?", "Estimate public transport costs in {city} for {days} days.", "Can I move around {city} without a car, and what will it cost for {days} days?", "Show metro, bus and pass options for {city}."],
        "location": ["Give me the practical basics for {city}: currency, language and timezone.", "What should I know about {city} before I arrive?", "Tell me the local currency and timezone for {city}.", "Give me destination facts for {city}.", "I need practical local information about {city}.", "Which country, currency and language should I expect in {city}?"],
    },
    "validation": {
        "weather": ["What sort of conditions are forecast for {city} across {days} days?", "Would you pack rain gear for {city} for the coming {days} days?", "How warm or wet should {city} be during my {days}-day break?"],
        "hotel": ["Shortlist a place to sleep in {city} for {days} nights; ceiling {price} EUR nightly.", "Accommodation needed in {city} for {days} nights, preferably {rating}+ and no more than {price} EUR.", "Can you find a bed in {city} for {days} nights inside a {price} EUR nightly budget?"],
        "attractions": ["How would you spend sightseeing time in {city}?", "Which cultural stops in {city} are worth my time?", "Give me a compact list of things to explore in {city}."],
        "restaurant": ["Where could I have a decent meal in {city} without exceeding {meal} EUR?", "Point me to {cuisine} food in {city}.", "I need a dinner shortlist for {city}, around {meal} EUR each."],
        "transport": ["What's the easiest way to travel locally in {city} for {days} days?", "Which fare product should I buy for {city} over {days} days?", "Work out local transit choices for {city}."],
        "location": ["Brief me on the essentials of {city} before departure.", "What local facts matter for a first visit to {city}?", "Give me currency/language/time-zone basics for {city}."],
    },
    "test": {
        "weather": ["Do I need a coat or umbrella for {city} on a {days}-day trip?", "What conditions should I plan around in {city} for {days} days?", "Is outdoor sightseeing sensible in {city} over the next {days} days?"],
        "hotel": ["Find me a base in {city} for {days} nights with {price} EUR as the nightly limit.", "Where can I stay in {city} for {days} nights on at most {price} EUR each night?", "I still need lodging in {city}; {days} nights, budget {price} EUR/night."],
        "attractions": ["What is actually worth seeing in {city}?", "Give me a culture-heavy itinerary for {city}.", "Which sights would you prioritise in {city}?"],
        "restaurant": ["Where would you go for food in {city} with roughly {meal} EUR a head?", "Find a few places to eat in {city}, ideally {cuisine}.", "I want a good meal in {city}; what fits around {meal} EUR?"],
        "transport": ["How do locals get around {city}, and what should I buy for {days} days?", "Plan my city transport in {city} for a {days}-day stay.", "What's the sensible transit setup for {city}?"],
        "location": ["Before landing in {city}, what local basics should I know?", "Give me the key practical facts for {city}.", "What money, language and time zone apply in {city}?"],
    },
}

HU = {
    "train": {
        "weather": ["Milyen idő várható {city_hu} városában {days} napig?", "Nézd meg {city_hu} {days} napos időjárás-előrejelzését.", "Kell esernyőt vinnem {city_hu} városába a következő {days} napra?", "Milyen hőmérsékletre és csapadékra számítsak {city_hu} városában?", "{days} napot töltök {city_hu} városában; milyen időre készüljek?", "Várható eső {city_hu} környékén a {days} napos utam alatt?"],
        "hotel": ["Keress szállást {city_hu} városában {days} éjszakára, {price} EUR/éj alatt.", "{days} éjszakára kell hotel {city_hu} városában, maximum {price} euróért éjszakánként.", "Ajánlj {rating}+ értékelésű szállást {city_hu} városában {price} EUR alatt.", "Hol szálljak meg {city_hu} városában {days} éjszakára, ha {price} EUR a napi keret?", "Mutass szobákat {city_hu} városában {days} éjre legfeljebb {price} euróért.", "Szükségem van egy jó szállásra {city_hu} városában {days} éjszakára."],
        "attractions": ["Mit érdemes megnézni {city_hu} városában?", "Ajánlj múzeumokat és nevezetességeket {city_hu} városában.", "Keress {ticket} EUR alatti belépőjű látnivalókat {city_hu} városában.", "Mivel töltenél el egy fél napot {city_hu} városában?", "Kulturális programokat keresek {city_hu} városában.", "Milyen parkok, múzeumok és történelmi helyek vannak {city_hu} városában?"],
        "restaurant": ["Hol egyek {city_hu} városában körülbelül {meal} EUR/fő keretből?", "Keress {cuisine} éttermet {city_hu} városában {meal} EUR/fő alatt.", "Ajánlj vacsorahelyeket {city_hu} városában.", "Jó éttermeket keresek {city_hu} városában, maximum {meal} eurós étkezéssel.", "Hol lehet jót enni {city_hu} városában?", "Mutass legalább {rating} értékelésű éttermeket {city_hu} városában."],
        "transport": ["Hogyan közlekedjek {city_hu} városában {days} napig?", "Milyen tömegközlekedési bérletet érdemes venni {city_hu} városában {days} napra?", "Mennyibe kerül a helyi közlekedés {city_hu} városában {days} napra?", "Autó nélkül hogyan járjam be {city_hu} városát?", "Mutasd a metró-, busz- és bérletlehetőségeket {city_hu} városában.", "Számolj közlekedési keretet {city_hu} városára {days} napra."],
        "location": ["Adj gyakorlati információkat {city_hu} városáról: pénznem, nyelv és időzóna.", "Mit érdemes tudnom {city_hu} városáról indulás előtt?", "Mi a pénznem és az időzóna {city_hu} városában?", "Adj alapvető helyi tudnivalókat {city_hu} városáról.", "Melyik országban van {city_hu}, és milyen pénzt használnak?", "Milyen nyelvre és pénznemre készüljek {city_hu} városában?"],
    },
    "validation": {
        "weather": ["Milyen körülményekre készüljek {city_hu} városában a következő {days} nap során?", "Vigyünk esőkabátot {city_hu} városába egy {days} napos útra?", "Mennyire lesz meleg vagy esős {city_hu} a {days} napos kiruccanás alatt?"],
        "hotel": ["{city_hu} városában keresek alvóhelyet {days} éjre, éjszakánként legfeljebb {price} EUR-ért.", "Tudsz szállást találni {city_hu} városában {days} éjszakára {price} eurós plafonnal?", "Szállás kell {city_hu} városában, {days} éjszaka, legalább {rating} értékeléssel."],
        "attractions": ["Hogyan töltenél egy városnézős napot {city_hu} városában?", "Mely kulturális helyeket vennéd előre {city_hu} városában?", "Készíts rövid felfedeznivaló-listát {city_hu} városához."],
        "restaurant": ["Hol vacsorázzak {city_hu} városában {meal} euró körüli kerettel?", "Mutass {cuisine} helyeket {city_hu} városában.", "Kérek néhány jó étkezési lehetőséget {city_hu} városában."],
        "transport": ["Mi a legegyszerűbb helyi közlekedés {city_hu} városában {days} napra?", "Melyik jegyet vagy bérletet vegyem {city_hu} városában?", "Tervezd meg a helyi utazásomat {city_hu} városában."],
        "location": ["Foglalj össze minden fontos helyi alapinformációt {city_hu} városáról.", "Első utazás előtt mit kell tudni {city_hu} városáról?", "Pénznem, nyelv, időzóna: mi vonatkozik {city_hu} városára?"],
    },
    "test": {
        "weather": ["Kabát vagy esernyő kell inkább {city_hu} városába {days} napra?", "Milyen időjáráshoz igazítsam a programot {city_hu} városában?", "Alkalmas lesz az idő szabadtéri programokra {city_hu} városában?"],
        "hotel": ["Legyen egy bázisom {city_hu} városában {days} éjszakára, {price} EUR legyen a felső határ.", "Hol tudok megszállni {city_hu} városában {days} éjszakára legfeljebb {price} euróért?", "Még nincs szállásom {city_hu} városában: {days} éj, {price} EUR/éj."],
        "attractions": ["Mi az, amit tényleg kár lenne kihagyni {city_hu} városában?", "Kulturális programokból álló listát kérek {city_hu} városára.", "Mely látnivalókat priorizálnád {city_hu} városában?"],
        "restaurant": ["Hol ennél {city_hu} városában nagyjából {meal} euróból fejenként?", "Kérek néhány étteremötletet {city_hu} városában, lehetőleg {cuisine} konyhával.", "Jó vacsorát szeretnék {city_hu} városában {meal} euró körül."],
        "transport": ["A helyiek hogyan járnak {city_hu} városában, és mit vegyek {days} napra?", "Szervezd meg a városi közlekedésemet {city_hu} városában {days} napra.", "Mi az ésszerű tömegközlekedési megoldás {city_hu} városában?"],
        "location": ["Érkezés előtt milyen helyi alapokat tudjak {city_hu} városáról?", "Add meg a legfontosabb gyakorlati tényeket {city_hu} városáról.", "Milyen pénz, nyelv és időzóna van {city_hu} városában?"],
    },
}

INTRO = {
    "en": ["I'm planning a trip.", "Quick travel question:", "For an upcoming city break,", "I'm putting together my itinerary.", "A practical question:", "Help me plan this properly.", "Before I book the rest of the trip,", ""],
    "hu": ["Utazást tervezek.", "Gyors utazási kérdés:", "Épp összeállítom a programomat.", "Szeretném rendesen megtervezni az utat.", "Egy gyakorlati kérdés:", "Mielőtt mindent lefoglalok,", ""],
}
TAIL = {
    "en": ["Please keep the answer concise.", "I prefer practical options.", "This is for a first visit.", "I'd rather avoid unnecessary detours.", "A short ranked list is enough.", "Please use the constraints above.", ""],
    "hu": ["Legyen inkább praktikus a válasz.", "Első alkalommal megyek.", "Rövid, rangsorolt lista is elég.", "Tartsd be a fenti kereteket.", "Nem szeretnék felesleges kerülőket.", ""],
}
CONNECT = {
    "en": [" Also, ", " Then ", " On top of that, ", " While you're at it, ", " I also need this: ", "; ", " And ", " Next, "],
    "hu": [" Emellett ", " Ezután ", " Továbbá ", " Ha már itt tartunk, ", " Arra is szükségem van, hogy ", "; ", " És ", " Végül "],
}
CUISINES = ["local", "Italian", "Asian", "Mediterranean", "vegan", "street-food", "seafood", "Indian", "Middle Eastern", "bakery/café"]


def render_clause(intent: str, split: str, lang: str, p: dict, variant: int) -> str:
    bank = EN if lang == "en" else HU
    arr = bank[split][intent]
    template = arr[variant % len(arr)]
    return template.format(**p)


def lightly_noisify(text: str, lang: str, key: int) -> str:
    if key % 7 == 0:
        text = text.replace("?", "").replace(".", "")
    if key % 11 == 0:
        text = text.lower()
    if lang == "hu" and key % 13 == 0:
        table = str.maketrans("áéíóöőúüű", "aeiooouuu")
        text = text.translate(table)
    if key % 17 == 0:
        text = re.sub(r"\bplease\b", "pls", text, flags=re.I)
    return text


def generate_router_corpus() -> tuple[list[dict], list[dict]]:
    cities = pd.read_csv(RAW / "cities.csv")["city"].astype(str).tolist()
    split_sizes = {"train": 192000, "validation": 24000, "test": 24000}
    rows: list[dict] = []
    qid = 1
    local_rng = random.Random(SEED + 101)
    used_queries: set[str] = set()
    context = {
        "en": ["I'm travelling solo.", "We're travelling as a couple.", "This is a first visit.", "I'm travelling with family.", "It's a short city break.", "I prefer practical options.", "I'd like to stay fairly central.", "I care more about value than luxury.", "I don't mind walking.", "I'd rather keep the plan flexible.", ""],
        "hu": ["Egyedül utazom.", "Páros utazás lesz.", "Most járok ott először.", "Családdal utazom.", "Rövid városnézés lesz.", "Inkább praktikus megoldásokat keresek.", "Szeretnék viszonylag központi helyeket.", "Az ár-érték arány fontosabb a luxusnál.", "Nem gond, ha sokat kell sétálni.", "Rugalmas programot szeretnék.", ""],
    }
    temporal = {
        "en": ["next week", "this weekend", "later this month", "for an upcoming trip", "for a spring break", "for a quick getaway", ""],
        "hu": ["jövő héten", "ezen a hétvégén", "még ebben a hónapban", "egy közelgő útra", "egy tavaszi kiruccanásra", "egy rövid utazásra", ""],
    }

    def make_unique(split: str, i: int) -> dict:
        nonlocal qid
        # Retry with independent random choices until the surface query is unique.
        for attempt in range(60):
            lang = "hu" if local_rng.random() < 0.5 else "en"
            city = local_rng.choice(cities); city_hu = HU_NAMES.get(city, city)
            days = local_rng.randint(1, 10); price = local_rng.randrange(70, 231, 10)
            rating = local_rng.choice([3.5, 3.8, 4.0, 4.2, 4.5, 4.7]); ticket = local_rng.randrange(5, 51, 5)
            meal = local_rng.randrange(15, 61, 3); cuisine = local_rng.choice(CUISINES)
            p = dict(city=city, city_hu=city_hu, days=days, price=price, rating=rating, ticket=ticket, meal=meal, cuisine=cuisine)
            r = local_rng.random(); k = 1 if r < 0.43 else 2 if r < 0.78 else 3 if r < 0.94 else 4
            labels = local_rng.sample(INTENTS, k=k)
            clauses = [render_clause(intent, split, lang, p, local_rng.randrange(10000)) for intent in labels]
            if local_rng.random() < 0.5: local_rng.shuffle(clauses)
            query = clauses[0]
            for clause in clauses[1:]:
                query += local_rng.choice(CONNECT[lang]) + clause[0].lower() + clause[1:]
            if local_rng.random() < 0.55:
                intro = local_rng.choice(INTRO[lang])
                if intro: query = f"{intro} {query}"
            if local_rng.random() < 0.42:
                ctx = local_rng.choice(context[lang])
                if ctx: query = f"{ctx} {query}"
            if local_rng.random() < 0.32:
                when = local_rng.choice(temporal[lang])
                if when:
                    query = (f"I'm planning this {when}. {query}" if lang == "en" else f"{when.capitalize()} készülök. {query}")
            if local_rng.random() < 0.45:
                tail = local_rng.choice(TAIL[lang])
                if tail: query = f"{query} {tail}"
            if "hotel" not in labels and local_rng.random() < 0.045:
                query += " I already booked my hotel, so don't search accommodation." if lang == "en" else " A szállás már megvan, hotelt ne keress."
            if "restaurant" not in labels and local_rng.random() < 0.035:
                query += " No restaurant suggestions are needed." if lang == "en" else " Étteremajánlásra nincs szükségem."
            if split == "train" and local_rng.random() < 0.14:
                query = lightly_noisify(query, lang, i + attempt + qid)
            query = re.sub(r"\s+", " ", query).strip()
            if query not in used_queries:
                used_queries.add(query)
                return {
                    "query_id": f"RT{qid:06d}", "split": split, "language": lang, "query": query,
                    "tool_labels": "|".join(sorted(labels)), "city": city, "days": days,
                    "hotel_max_price_eur": price if "hotel" in labels else "",
                    "hotel_min_rating": rating if "hotel" in labels else "",
                    "restaurant_max_meal_eur": meal if "restaurant" in labels else "",
                    "attraction_max_ticket_eur": ticket if "attractions" in labels else "",
                    "difficulty": "multi" if len(labels) > 1 else "single",
                    "template_family": f"{split}:{lang}:" + "+".join(sorted(labels)) + f":{local_rng.randrange(1000):03d}",
                }
        raise RuntimeError(f"Could not generate unique query for {split} row {i}")

    for split, size in split_sizes.items():
        for i in range(size):
            rows.append(make_unique(split, i)); qid += 1

    challenge_phrases = {
        "weather": ["Should I pack an umbrella for {city}?", "Would outdoor plans be risky in {city}?", "Am I likely to get soaked in {city}?", "Will the weather ruin outdoor plans in {city}?", "Kabát vagy esernyő legyen nálam {city_hu} városában?", "Szabadtéri programhoz milyen körülmények lesznek {city_hu} városában?", "Elázhatok {city_hu} városában?"],
        "hotel": ["I still need somewhere to sleep in {city}.", "I haven't sorted a base in {city} yet.", "I need a roof over my head in {city}.", "Még nincs hol aludnom {city_hu} városában.", "Kellene egy bázis {city_hu} városában az éjszakákra.", "A szállás kérdése még nyitott {city_hu} városában."],
        "attractions": ["Fill an afternoon in {city} with worthwhile stops.", "What would you not skip in {city}?", "Give me reasons to leave the hotel in {city}.", "Mivel töltenél el egy délutánt {city_hu} városában?", "Mi az, amit kár lenne kihagyni {city_hu} városában?", "Miket néznél meg elsőként {city_hu} városában?"],
        "restaurant": ["Where would you actually go for dinner in {city}?", "I need somewhere good to eat in {city}.", "Find me a table worth booking in {city}.", "Hol vacsoráznál {city_hu} városában?", "Kellene egy jó hely enni {city_hu} városában.", "Hová ülnél be enni {city_hu} városában?"],
        "transport": ["I won't have a car in {city}; how do I move around?", "What's the smartest way to get around {city}?", "Help me avoid taxis in {city}.", "Nem lesz autóm {city_hu} városában; hogyan járjak?", "Mi a legegyszerűbb módja, hogy bejárjam {city_hu} városát?", "Taxik nélkül hogyan közlekedjek {city_hu} városában?"],
        "location": ["Give me the local basics before I land in {city}.", "What practical facts should I know about {city}?", "Brief me before I arrive in {city}.", "Érkezés előtt mik a legfontosabb helyi tudnivalók {city_hu} városáról?", "Milyen alapinformációkat tudjak {city_hu} városáról?", "Gyors helyi eligazítást kérek {city_hu} városáról."],
    }
    challenge=[]; challenge_used=set(); challenge_rng=random.Random(SEED+202)
    distract_en=["I'm trying to travel light.","I already have flights.","This is my first time there.","Keep it practical.","I'm flexible on timing.",""]
    distract_hu=["Kevés csomaggal megyek.","A repülőjegy már megvan.","Most leszek ott először.","Legyen praktikus.","Az időzítés rugalmas.",""]
    for i in range(36000):
        for attempt in range(40):
            city=challenge_rng.choice(cities); city_hu=HU_NAMES.get(city,city); days=challenge_rng.randint(1,9)
            p={"city":city,"city_hu":city_hu,"days":days}
            r=challenge_rng.random(); k=1 if r<0.48 else 2 if r<0.86 else 3
            labels=challenge_rng.sample(INTENTS,k)
            parts=[challenge_rng.choice(challenge_phrases[label]).format(**p) for label in labels]
            challenge_rng.shuffle(parts); q=challenge_rng.choice([" "," Also, "," / ","; "]).join(parts)
            lang="hu" if sum(ch in q for ch in "áéíóöőúüű")>0 else "en"
            distract=challenge_rng.choice(distract_hu if lang=="hu" else distract_en)
            if distract and challenge_rng.random()<0.65:q=f"{distract} {q}"
            if "hotel" not in labels and challenge_rng.random()<0.08:q += " I already have accommodation booked; don't search hotels." if lang=="en" else " A szállás már megvan, hotelt ne keress."
            if "restaurant" not in labels and challenge_rng.random()<0.06:q += " Food is already arranged, so skip restaurant ideas." if lang=="en" else " Az étkezés megvan, éttermet ne ajánlj."
            if challenge_rng.random()<0.18:q=lightly_noisify(q,lang,i+attempt)
            q=re.sub(r"\s+"," ",q).strip()
            if q not in challenge_used:
                challenge_used.add(q);break
        challenge.append({"query_id":f"CH{i+1:06d}","query":q,"tool_labels":"|".join(sorted(labels)),"city":city,"difficulty":"indirect_noisy"})
    return rows, challenge

def write_router_data(rows: list[dict], challenge: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(RAW / "intent_router_dataset.csv", index=False)
    pd.DataFrame(challenge).to_csv(RAW / "intent_router_challenge.csv", index=False)
    # User-facing examples come from all splits, not only training, and retain metadata.
    sample = pd.DataFrame(rows).sample(n=min(90000, len(rows)), random_state=SEED)[["query_id","language","query","tool_labels","difficulty"]]
    sample.to_csv(RAW / "sample_user_queries.csv", index=False)


def _args_for(label: str, city: str, days: int, price: float, rating: float, ticket: float, meal: float) -> dict:
    if label == "weather": return {"city":city,"days":days,"unit":"celsius"}
    if label == "hotel": return {"city":city,"nights":days,"max_price_per_night_eur":float(price),"min_rating":0.0,"top_k":5}
    if label == "attractions": return {"city":city,"categories":None,"max_ticket_eur":None,"top_k":6}
    if label == "restaurant": return {"city":city,"cuisines":None,"max_meal_eur":float(meal),"min_rating":0.0,"vegetarian_only":False,"top_k":6}
    if label == "transport": return {"city":city,"days":days}
    if label == "location": return {"city":city}
    raise KeyError(label)


def generate_benchmark() -> None:
    cities = pd.read_csv(RAW / "cities.csv")["city"].astype(str).tolist()
    bench_en = {
        "weather": [
            "Check the next {days} days of weather in {city}.",
            "What conditions should I expect in {city} over the next {days} days?",
            "For a {days}-day stay in {city}, do I need rain gear?",
            "Give me temperature and rain information for {city} for {days} days.",
        ],
        "hotel": [
            "I still need lodging in {city} for {days} nights, with a budget of {price} EUR per night.",
            "Find a base in {city} for {days} nights; nightly limit {price} EUR.",
            "Where can I stay in {city} for {days} nights under {price} EUR a night?",
            "Shortlist accommodation in {city}: {days} nights, maximum {price} EUR/night.",
        ],
        "attractions": [
            "What sights would you prioritise in {city}?",
            "Give me worthwhile cultural stops in {city}.",
            "What should I actually see in {city}?",
            "Build a sightseeing shortlist for {city}.",
        ],
        "restaurant": [
            "Where would you go for food in {city} with roughly {meal} EUR a head?",
            "Find a good meal in {city} for about {meal} EUR per person.",
            "Give me restaurant ideas in {city} around {meal} EUR each.",
            "Where can I eat in {city} while keeping it under {meal} EUR per person?",
        ],
        "transport": [
            "Plan local transport in {city} for {days} days.",
            "How should I get around {city} during a {days}-day trip?",
            "Which transit option makes sense in {city} for {days} days?",
            "Estimate public-transport needs in {city} for {days} days.",
        ],
        "location": [
            "Give me the key practical facts for {city}.",
            "What local basics should I know before arriving in {city}?",
            "Tell me the currency, language and timezone for {city}.",
            "Brief me on essential destination facts for {city}.",
        ],
    }
    bench_hu = {
        "weather": [
            "Nézd meg {city_hu} következő {days} napos időjárását.",
            "Milyen körülményekre készüljek {city_hu} városában a következő {days} nap során?",
            "Egy {days} napos {city_hu} utazáshoz kell esernyő?",
            "Adj hőmérséklet- és csapadékinformációt {city_hu} városára {days} napra.",
        ],
        "hotel": [
            "Még nincs szállásom {city_hu} városában: {days} éjszaka, maximum {price} EUR/éj.",
            "Keress bázist {city_hu} városában {days} éjre, {price} eurós éjszakánkénti kerettel.",
            "Hol szálljak meg {city_hu} városában {days} éjszakára {price} EUR alatt?",
            "Ajánlj szállást {city_hu} városában {days} éjszakára, legfeljebb {price} EUR/éj áron.",
        ],
        "attractions": [
            "Mely látnivalókat priorizálnád {city_hu} városában?",
            "Adj kulturális programötleteket {city_hu} városában.",
            "Mit érdemes tényleg megnézni {city_hu} városában?",
            "Készíts városnézős listát {city_hu} városához.",
        ],
        "restaurant": [
            "Hol ennél {city_hu} városában nagyjából {meal} EUR/fő keretből?",
            "Keress jó éttermet {city_hu} városában körülbelül {meal} EUR/fő áron.",
            "Adj vacsoraötleteket {city_hu} városában {meal} EUR/fő körül.",
            "Hol lehet enni {city_hu} városában legfeljebb {meal} EUR/fő kerettel?",
        ],
        "transport": [
            "Tervezd meg a helyi közlekedésemet {city_hu} városában {days} napra.",
            "Hogyan járjam be {city_hu} városát egy {days} napos úton?",
            "Melyik tömegközlekedési megoldás jó {city_hu} városában {days} napra?",
            "Becsüld meg a helyi közlekedési igényt {city_hu} városában {days} napra.",
        ],
        "location": [
            "Add meg a legfontosabb gyakorlati tényeket {city_hu} városáról.",
            "Milyen helyi alapinformációkat tudjak {city_hu} városáról érkezés előtt?",
            "Mi a pénznem, a nyelv és az időzóna {city_hu} városában?",
            "Adj rövid helyi eligazítást {city_hu} városáról.",
        ],
    }
    connectors={"en":[" Also, "," Then, ","; "," On top of that, "],"hu":[" Emellett "," Ezután ","; "," Továbbá "]}
    bench=[]
    brng=random.Random(SEED+303)
    for i in range(22500):
        lang="hu" if i%2==0 else "en"; bank=bench_hu if lang=="hu" else bench_en
        city=cities[(i*17+7)%len(cities)];city_hu=HU_NAMES.get(city,city);days=1+i%7;price=80+(i%10)*15;rating=4.0;ticket=10+(i%5)*5;meal=18+(i%8)*4;cuisine=CUISINES[i%len(CUISINES)]
        p=dict(city=city,city_hu=city_hu,days=days,price=price,rating=rating,ticket=ticket,meal=meal,cuisine=cuisine)
        k=1 if i%10<4 else 2 if i%10<7 else 3 if i%10<9 else 4
        labels=brng.sample(INTENTS,k)
        clauses=[bank[lab][(i+j*3)%len(bank[lab])].format(**p) for j,lab in enumerate(labels)]
        query=clauses[0]
        for clause in clauses[1:]: query+=brng.choice(connectors[lang])+clause[0].lower()+clause[1:]
        expected=[{"name": {"weather":"get_weather","hotel":"search_hotels","attractions":"search_attractions","restaurant":"search_restaurants","transport":"get_transport_options","location":"get_location_info"}[lab], "arguments":_args_for(lab,city,days,price,rating,ticket,meal)} for lab in labels]
        bench.append({"id":f"CASE{i+1:05d}","language":lang,"query":query,"expected_calls":expected,"difficulty":"heldout_surface","labels":labels})
    (BENCH / "agent_tasks.json").write_text(json.dumps(bench,ensure_ascii=False,indent=2),encoding="utf-8")
    with (BENCH / "agent_tasks.jsonl").open("w",encoding="utf-8") as f:
        for row in bench:f.write(json.dumps(row,ensure_ascii=False)+"\n")

def update_manifest(rows: list[dict], challenge: list[dict]) -> None:
    path=PROCESSED/"dataset_manifest.json"
    old=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    counts={}
    for name in ["cities.csv","hotels.csv","attractions.csv","restaurants.csv","transport.csv","weather_fallback.csv","fx_rates_fallback.csv","sample_user_queries.csv","intent_router_dataset.csv","intent_router_challenge.csv"]:
        p=RAW/name
        counts[name]=sum(1 for _ in p.open(encoding="utf-8"))-1
    counts["benchmark/agent_tasks.json"]=len(json.loads((BENCH/"agent_tasks.json").read_text(encoding="utf-8")))
    old.update({
        "generator_seed":SEED,"quality_upgrade":"v2-compositional-heldout-surfaces",
        "counts":counts,
        "files":counts,
        "splits":{"intent_router_train":192000,"intent_router_validation":24000,"intent_router_test":24000,"challenge":36000},
        "notes":{
            "entity_inventory":"Synthetic deterministic inventory with diversified names and attributes; not live/bookable.",
            "router_corpus":"Synthetic bilingual compositional corpus with split-specific surface forms, hard negatives, noise and indirect challenge queries.",
            "evaluation":"Validation/test phrase families are held out from training to reduce surface-template leakage.",
        },
    })
    path.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding="utf-8")


def main() -> None:
    improve_entity_names()
    rows,challenge=generate_router_corpus()
    write_router_data(rows,challenge)
    generate_benchmark()
    update_manifest(rows,challenge)
    print("Data-quality upgrade complete.")
    print(f"intent_router_dataset.csv: {len(rows):,}")
    print(f"intent_router_challenge.csv: {len(challenge):,}")
    print("agent_tasks.json: 22,500")


if __name__ == "__main__":
    main()
