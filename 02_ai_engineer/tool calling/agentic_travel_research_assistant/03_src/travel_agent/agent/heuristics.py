from __future__ import annotations

import re
from functools import lru_cache

from travel_agent.data_store import read_csv

CURRENCIES = {
    "eur":"EUR","euro":"EUR","euró":"EUR","€":"EUR","huf":"HUF","forint":"HUF","forints":"HUF",
    "usd":"USD","dollar":"USD","$":"USD","gbp":"GBP","font":"GBP","£":"GBP","czk":"CZK","pln":"PLN",
    "chf":"CHF","sek":"SEK","nok":"NOK","dkk":"DKK","ron":"RON","try":"TRY",
}
NUMBER_WORDS={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"egy":1,"két":2,"ket":2,"kettő":2,"három":3,"harom":3,"négy":4,"negy":4,"öt":5,"ot":5,"hat":6,"hét":7,"het":7}

@lru_cache(maxsize=1)
def city_aliases() -> dict[str,str]:
    aliases={r["city"].casefold():r["city"] for r in read_csv("cities.csv")}
    aliases.update({
        "bécs":"Vienna", "becs":"Vienna",
        "prága":"Prague", "praga":"Prague",
        "párizs":"Paris", "parizs":"Paris",
        "róma":"Rome", "roma":"Rome",
        "münchen":"Munich", "munchen":"Munich",
        "krakkó":"Krakow", "krakko":"Krakow",
        "amszterdam":"Amsterdam",
        "koppenhága":"Copenhagen", "koppenhaga":"Copenhagen",
        "varsó":"Warsaw", "varso":"Warsaw",
        "zürich":"Zurich", "zurich":"Zurich",
        "lisszabon":"Lisbon",
        "reykjavík":"Reykjavik", "reykjavik":"Reykjavik",
        "portó":"Porto", "porto":"Porto",
    })
    return aliases


HU_LOCATION_SUFFIXES = (
    "ban", "ben", "ból", "ből", "ról", "ről", "tól", "től",
    "nál", "nél", "hoz", "hez", "höz", "nak", "nek",
    "ba", "be", "ra", "re", "on", "en", "ön", "ig",
)


def _hungarian_city_stems(alias: str) -> set[str]:
    """Return forms that can take Hungarian case suffixes.

    Hungarian place names ending in ``a``/``e`` often lengthen that final vowel
    before a suffix (Róma -> Rómába, Vienna -> Viennába). Keeping this logic
    here makes city extraction language-aware without hard-coding every declined
    city form into the dataset.
    """
    stems = {alias}
    if alias.endswith("a"):
        stems.add(alias[:-1] + "á")
    elif alias.endswith("e"):
        stems.add(alias[:-1] + "é")
    return stems


def extract_city(text:str)->str|None:
    low=text.casefold()
    suffix_pattern = "|".join(re.escape(s) for s in HU_LOCATION_SUFFIXES)

    # Longest aliases first prevents accidental substring preference. Match both
    # standalone city names ("Bécs") and Hungarian declined forms
    # ("Bécsbe", "Budapesten", "Rómába", "Londonból").
    for alias,canonical in sorted(city_aliases().items(), key=lambda kv: -len(kv[0])):
        for stem in _hungarian_city_stems(alias):
            if re.search(rf"(?<!\w){re.escape(stem)}(?:(?:{suffix_pattern}))?(?!\w)", low):
                return canonical
    return None


def extract_days_enhanced(text:str)->int:
    low=text.casefold()
    for p in [r"(\d+)\s*(?:napra|napos|days?|day)",r"(\d+)\s*(?:éjszakára|éjre|nights?|night)"]:
        m=re.search(p,low)
        if m:return max(1,min(int(m.group(1)),30))
    for word,n in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\s+(?:days?|nights?|nap(?:ra|os)?|éj(?:re|szakára)?)\b",low):return n
    if "weekend" in low or "hétvége" in low:return 2
    return 3


def currency_request_enhanced(text:str)->tuple[float,str,str]|None:
    low=text.casefold().replace(",", ".")
    low = (low.replace("svájci frankban", "chf").replace("svajci frankban", "chf")
               .replace("svájci frank", "chf").replace("svajci frank", "chf"))

    # Prefer explicit conversion grammar so unrelated prices (for example a hotel
    # budget of 150 EUR) are not mistaken for the amount to convert.
    code_pattern = r"eur|euro|euró|huf|forints?|usd|dollar|gbp|font|czk|pln|chf|sek|nok|dkk|ron|try"
    explicit = re.search(
        rf"(?:convert|exchange|how much is|mennyi)\s*(?:([€$£])\s*)?(\d+(?:\.\d+)?)\s*({code_pattern})?\s*(?:to|into|in|=|->|forintban|forintra)\s*({code_pattern})",
        low,
    )
    if explicit:
        symbol, amount_s, src_token, dst_token = explicit.groups()
        src = CURRENCIES[symbol] if symbol else CURRENCIES[src_token]
        dst = CURRENCIES[dst_token]
        if src != dst:
            return float(amount_s), src, dst

    # Common Hungarian form: "mennyi 500 euró forintban".
    hu = re.search(rf"(?:mennyi\s*)?(\d+(?:\.\d+)?)\s*({code_pattern}).{{0,30}}?(forint|huf|eur|usd|gbp|czk|pln|chf)", low)
    if hu:
        amount, src_token, dst_token = hu.groups(); src=CURRENCIES[src_token]; dst=CURRENCIES[dst_token]
        if src != dst: return float(amount), src, dst

    # Symbol-before-amount form, e.g. €300 in HUF.
    m=re.search(rf"([€$£])\s*(\d+(?:\.\d+)?).{{0,30}}?(?:to|in|into)?\s*({code_pattern})",low)
    if m:
        src=CURRENCIES[m.group(1)];dst=CURRENCIES[m.group(3)]
        if src!=dst:return float(m.group(2)),src,dst

    # Last-resort: inspect all amount/currency mentions and prefer the last one
    # that has a different target currency later in the sentence.
    matches=list(re.finditer(rf"(\d+(?:\.\d+)?)\s*({code_pattern})",low))
    for m in reversed(matches):
        src=CURRENCIES[m.group(2)];target=_find_target(low,m.end(),src)
        if target:return float(m.group(1)),src,target
    return None


def _find_target(low:str,start:int,src:str)->str|None:
    suffix=low[start:]
    for token,code in CURRENCIES.items():
        if token in {"€","$","£"}:continue
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)",suffix) and code!=src:return code
    if ("forint" in low or "huf" in low) and src!="HUF":return "HUF"
    return None


def detect_intents(text:str,*,enhanced:bool=True)->dict[str,bool]:
    low=text.casefold()
    neg_hotel=any(x in low for x in [
        "no hotel", "without hotel", "without a hotel", "without accommodation", "do not include a hotel", "don't include a hotel",
        "do not search for a hotel", "don't search for a hotel", "accommodation is already booked", "my accommodation is already booked",
        "hotel nélkül", "szállás nélkül", "ne keress hotelt", "ne keress szállást", "szállásom már megvan",
        "a szállásom már megvan", "már foglaltam hotelt", "már van szállásom",
    ])
    return {
        "weather":any(k in low for k in ["weather","forecast","időjár","idő lesz","nézd meg az időt","nezd meg az idot","rain","eső","temperature","hőmér","umbrella","jacket","esernyő","esernyo","kabát","kabat","outdoor conditions","outdoor sightseeing","conditions should i plan","körülményekre","szabadtéri","pack for the weather"]),
        "hotel":(any(k in low for k in ["hotel","accommodation","szállás","stay","room","sleep","place to stay","overnight","lodging","base in","somewhere to sleep","megszáll","alvóhely","bázis"]) and not neg_hotel),
        "attractions":any(k in low for k in ["attraction","museum","landmark","historic","places to visit","visit in","látnival","múzeum","nevezetess","sightseeing","worth seeing","things to see","culture","cultural","sights","városnéz","kulturális hely","worth my time"]),
        "transport":any(k in low for k in ["transport","public transport","transit","metro","bus","pass price","fare product","city transport","transit setup","travel locally","közleked","tömegközleked","bérlet","helyi utazás","utazásomat","get around","move around","without a car"]),
        "restaurant":any(k in low for k in ["restaurant","restaurants","food options","food in","for food","meal in","good meal","where can i eat","where to eat","eat in","étterem","étterm","etterem","etterm","hol egyek","étkezés","vacsora","dinner","lunch","somewhere good to eat","food recommendations"]),
        "budget":any(k in low for k in ["budget","trip budget","travel budget","cost estimate","cost plan","total trip cost","overall cost","költségterv","költségbecslés","utazási költség","teljes költség","teljes helyi költség"]),
        "location":any(k in low for k in ["destination information","destination details","practical destination","country","currency","timezone","language","helyi inform","helyi alap","gyakorlati tény","alapinformáció","ország","pénznem","nyelv","időzóna","idozona","before landing","before i land","practical basics","practical facts","key practical facts","local basics","local facts","before visiting","érkezés előtt","erkezes elott"]),
        "exclude_hotel":neg_hotel,
    }


def hotel_constraints(text:str)->tuple[float|None,float,int]:
    low=text.casefold().replace(",", ".")
    price=None;rating=0.0;top_k=5

    # English / prefix-style Hungarian: "under 150 EUR", "legfeljebb 150 EUR".
    m=re.search(r"(?:under|below|max(?:imum)?|alatt|legfeljebb)\s*(\d+(?:\.\d+)?)\s*(?:eur|euró|€)?",low)
    if m:
        price=float(m.group(1))

    # Natural Hungarian amount-first order: "150 euró alatti hotel",
    # "120 EUR alatt", "100 € alá".
    if price is None:
        m=re.search(r"(\d+(?:\.\d+)?)\s*(?:eur|euró|€)\s*(?:alatt|alatti|alá|legfeljebb)",low)
        if m:
            price=float(m.group(1))
    if price is None:
        m=re.search(r"(?:budget|ceiling|limit|keret|plafon)[^\d]{0,20}(\d+(?:\.\d+)?)\s*(?:eur|euró|€)",low)
        if m:
            price=float(m.group(1))

    m=re.search(r"(?:rating|értékelés)(?:\s+of)?(?:\s+at\s+least|\s*>=?|\s+minimum|\s+legalább)?\s*(\d(?:\.\d)?)",low)
    if not m:
        m=re.search(r"(?:legalább|minimum|at least)\s*(\d(?:\.\d)?)(?:-?ös|-?es)?\s*(?:értékelés|rating)",low)
    if m:rating=float(m.group(1))
    return price,rating,top_k


def attraction_constraints(text:str)->tuple[list[str]|None,float|None,int]:
    low=text.casefold().replace(",", ".")
    cats=[]
    mapping={"museum":"museum","múzeum":"museum","landmark":"landmark","historic":"historic_site","történelmi":"historic_site","tortenelmi":"historic_site","park":"park","gallery":"gallery","galéria":"gallery","galeria":"gallery","market":"market","piac":"market"}
    for token,cat in mapping.items():
        if token in low and cat not in cats:cats.append(cat)
    max_ticket=None
    m=re.search(r"(?:ticket|tickets|belépő)(?:\s+price)?(?:\s+under|\s+below|\s+maximum|\s+max|\s+no more than)?\s*(\d+(?:\.\d+)?)\s*(?:eur|euró|€)",low)
    if not m:m=re.search(r"(?:maximum|no more than|legfeljebb)\s+(?:ticket\s+)?(\d+(?:\.\d+)?)\s*(?:eur|euró|€)",low)
    if not m:m=re.search(r"(?:legfeljebb\s*)?(\d+(?:\.\d+)?)\s*(?:eur|euró|€)(?:-?s|s)?\s*belépő",low)
    if m:max_ticket=float(m.group(1))
    return cats or None,max_ticket,6


def restaurant_constraints(text:str)->tuple[list[str]|None,float|None,float,bool,int]:
    low=text.casefold().replace(",", ".")
    cuisines=[]
    mapping={"local":"local","helyi":"local","italian":"italian","olasz":"italian","asian":"asian","ázsiai":"asian","azsiai":"asian","mediterranean":"mediterranean","mediterrán":"mediterranean","mediterran":"mediterranean","vegan":"vegan","vegán":"vegan","street food":"street_food","utcai étel":"street_food","fine dining":"fine_dining","bakery":"bakery","pékség":"bakery","pekseg":"bakery","cafe":"cafe","kávézó":"cafe","kavezo":"cafe","seafood":"seafood","tenger gyümölcsei":"seafood","middle eastern":"middle_eastern","közel-keleti":"middle_eastern","kozel-keleti":"middle_eastern","indian":"indian","indiai":"indian"}
    for token,cuisine in mapping.items():
        if token in low and cuisine not in cuisines:cuisines.append(cuisine)
    max_meal=None
    m=re.search(r"(?:under|below|max(?:imum)?|around|roughly|about|alatt|legfeljebb|körülbelül|nagyjából)\s*(\d+(?:\.\d+)?)\s*(?:eur|€)(?:\s*(?:per person|a head|/fő|per meal|fejenként))?",low)
    if m:max_meal=float(m.group(1))
    min_rating=0.0
    m=re.search(r"(?:rated|rating|értékelés)(?:\s+of)?(?:\s+at\s+least|\s*>=?|\s+minimum|\s+legalább)?\s*(\d(?:\.\d)?)",low)
    if not m:
        m=re.search(r"(?:legalább|minimum|at least)\s*(\d(?:\.\d)?)(?:-?ös|-?es)?\s*(?:értékelés|rating)",low)
    if m:min_rating=float(m.group(1))
    vegetarian=any(x in low for x in ["vegetarian","veggie","vegetáriánus","vega "])
    return cuisines or None,max_meal,min_rating,vegetarian,6
