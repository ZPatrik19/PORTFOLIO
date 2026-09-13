from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_data" / "raw"
BENCH = ROOT / "01_data" / "benchmark"
PROCESSED = ROOT / "01_data" / "processed"
for p in (RAW, BENCH, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Deliberately compact but useful European destination catalogue.
# Coordinates/timezones/currencies are static reference metadata, while generated hotels/POIs are synthetic.
CITIES = [
    ("Vienna","Austria","AT","EUR","German","Europe/Vienna",48.2082,16.3738,42,9,4.7),
    ("Budapest","Hungary","HU","HUF","Hungarian","Europe/Budapest",47.4979,19.0402,30,7,4.7),
    ("Prague","Czechia","CZ","CZK","Czech","Europe/Prague",50.0755,14.4378,32,6,4.7),
    ("Berlin","Germany","DE","EUR","German","Europe/Berlin",52.5200,13.4050,38,10,4.6),
    ("Paris","France","FR","EUR","French","Europe/Paris",48.8566,2.3522,48,11,4.9),
    ("Rome","Italy","IT","EUR","Italian","Europe/Rome",41.9028,12.4964,40,7,4.9),
    ("Milan","Italy","IT","EUR","Italian","Europe/Rome",45.4642,9.1900,45,7,4.5),
    ("Venice","Italy","IT","EUR","Italian","Europe/Rome",45.4408,12.3155,50,8,4.8),
    ("Florence","Italy","IT","EUR","Italian","Europe/Rome",43.7696,11.2558,43,6,4.8),
    ("Amsterdam","Netherlands","NL","EUR","Dutch","Europe/Amsterdam",52.3676,4.9041,48,10,4.7),
    ("Brussels","Belgium","BE","EUR","Dutch/French","Europe/Brussels",50.8503,4.3517,42,8,4.3),
    ("Lisbon","Portugal","PT","EUR","Portuguese","Europe/Lisbon",38.7223,-9.1393,32,7,4.7),
    ("Porto","Portugal","PT","EUR","Portuguese","Europe/Lisbon",41.1579,-8.6291,29,6,4.6),
    ("Madrid","Spain","ES","EUR","Spanish","Europe/Madrid",40.4168,-3.7038,36,7,4.6),
    ("Barcelona","Spain","ES","EUR","Catalan/Spanish","Europe/Madrid",41.3874,2.1686,39,8,4.8),
    ("Seville","Spain","ES","EUR","Spanish","Europe/Madrid",37.3891,-5.9845,31,5,4.7),
    ("Valencia","Spain","ES","EUR","Spanish","Europe/Madrid",39.4699,-0.3763,31,6,4.5),
    ("London","United Kingdom","GB","GBP","English","Europe/London",51.5072,-0.1276,52,12,4.8),
    ("Edinburgh","United Kingdom","GB","GBP","English","Europe/London",55.9533,-3.1883,42,8,4.7),
    ("Dublin","Ireland","IE","EUR","English/Irish","Europe/Dublin",53.3498,-6.2603,49,8,4.5),
    ("Copenhagen","Denmark","DK","DKK","Danish","Europe/Copenhagen",55.6761,12.5683,52,11,4.6),
    ("Stockholm","Sweden","SE","SEK","Swedish","Europe/Stockholm",59.3293,18.0686,48,10,4.6),
    ("Oslo","Norway","NO","NOK","Norwegian","Europe/Oslo",59.9139,10.7522,55,11,4.5),
    ("Helsinki","Finland","FI","EUR","Finnish/Swedish","Europe/Helsinki",60.1699,24.9384,43,9,4.4),
    ("Tallinn","Estonia","EE","EUR","Estonian","Europe/Tallinn",59.4370,24.7536,29,4,4.5),
    ("Riga","Latvia","LV","EUR","Latvian","Europe/Riga",56.9496,24.1052,28,4,4.4),
    ("Vilnius","Lithuania","LT","EUR","Lithuanian","Europe/Vilnius",54.6872,25.2797,27,4,4.4),
    ("Warsaw","Poland","PL","PLN","Polish","Europe/Warsaw",52.2297,21.0122,28,5,4.3),
    ("Krakow","Poland","PL","PLN","Polish","Europe/Warsaw",50.0647,19.9450,26,4,4.7),
    ("Gdansk","Poland","PL","PLN","Polish","Europe/Warsaw",54.3520,18.6466,27,4,4.5),
    ("Bratislava","Slovakia","SK","EUR","Slovak","Europe/Bratislava",48.1486,17.1077,29,5,4.2),
    ("Ljubljana","Slovenia","SI","EUR","Slovenian","Europe/Ljubljana",46.0569,14.5058,31,5,4.5),
    ("Zagreb","Croatia","HR","EUR","Croatian","Europe/Zagreb",45.8150,15.9819,30,5,4.3),
    ("Split","Croatia","HR","EUR","Croatian","Europe/Zagreb",43.5081,16.4402,34,4,4.6),
    ("Dubrovnik","Croatia","HR","EUR","Croatian","Europe/Zagreb",42.6507,18.0944,42,4,4.8),
    ("Athens","Greece","GR","EUR","Greek","Europe/Athens",37.9838,23.7275,32,5,4.7),
    ("Thessaloniki","Greece","GR","EUR","Greek","Europe/Athens",40.6401,22.9444,28,4,4.4),
    ("Bucharest","Romania","RO","RON","Romanian","Europe/Bucharest",44.4268,26.1025,25,4,4.1),
    ("Sofia","Bulgaria","BG","BGN","Bulgarian","Europe/Sofia",42.6977,23.3219,24,4,4.2),
    ("Belgrade","Serbia","RS","RSD","Serbian","Europe/Belgrade",44.7866,20.4489,25,4,4.3),
    ("Sarajevo","Bosnia and Herzegovina","BA","BAM","Bosnian","Europe/Sarajevo",43.8563,18.4131,23,3,4.5),
    ("Skopje","North Macedonia","MK","MKD","Macedonian","Europe/Skopje",41.9981,21.4254,22,3,4.0),
    ("Tirana","Albania","AL","ALL","Albanian","Europe/Tirane",41.3275,19.8187,24,3,4.2),
    ("Istanbul","Türkiye","TR","TRY","Turkish","Europe/Istanbul",41.0082,28.9784,27,4,4.8),
    ("Zurich","Switzerland","CH","CHF","German","Europe/Zurich",47.3769,8.5417,60,13,4.4),
    ("Geneva","Switzerland","CH","CHF","French","Europe/Zurich",46.2044,6.1432,62,12,4.3),
    ("Munich","Germany","DE","EUR","German","Europe/Berlin",48.1351,11.5820,43,9,4.6),
    ("Hamburg","Germany","DE","EUR","German","Europe/Berlin",53.5511,9.9937,39,8,4.4),
    ("Cologne","Germany","DE","EUR","German","Europe/Berlin",50.9375,6.9603,37,8,4.3),
    ("Lyon","France","FR","EUR","French","Europe/Paris",45.7640,4.8357,40,7,4.4),
    ("Nice","France","FR","EUR","French","Europe/Paris",43.7102,7.2620,44,6,4.6),
    ("Marseille","France","FR","EUR","French","Europe/Paris",43.2965,5.3698,36,6,4.3),
    ("Salzburg","Austria","AT","EUR","German","Europe/Vienna",47.8095,13.0550,39,5,4.7),
    ("Innsbruck","Austria","AT","EUR","German","Europe/Vienna",47.2692,11.4041,40,5,4.6),
    ("Reykjavik","Iceland","IS","ISK","Icelandic","Atlantic/Reykjavik",64.1466,-21.9426,58,8,4.6),
    ("Luxembourg","Luxembourg","LU","EUR","Luxembourgish/French","Europe/Luxembourg",49.6116,6.1319,48,0,4.2),
    ("Nicosia","Cyprus","CY","EUR","Greek/Turkish","Asia/Nicosia",35.1856,33.3823,31,4,4.1),
    ("Malta","Malta","MT","EUR","Maltese/English","Europe/Malta",35.8997,14.5147,34,4,4.6),
    ("Bologna","Italy","IT","EUR","Italian","Europe/Rome",44.4949,11.3426,39,5,4.5),
    ("Naples","Italy","IT","EUR","Italian","Europe/Rome",40.8518,14.2681,31,5,4.5),
]

CITY_FIELDS = ["city","country","country_code","currency","language","timezone","latitude","longitude","daily_food_budget_eur","daily_transport_eur","tourism_score"]
with (RAW / "cities.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(CITY_FIELDS); w.writerows(CITIES)

prefixes = ["Central","Grand","Urban","Riverside","Old Town","City","Boutique","Park","Station","Garden","Royal","Market","Harbour","Panorama","Metro","Heritage","Design","Local","Skyline","Liberty"]
suffixes = ["Hotel","Residence","Suites","Rooms","House","Stay","Inn","Lodge","Apartments","Hostel"]
areas = ["Centre","Old Town","Riverside","Museum District","Station Quarter","University District","Market District","Parkside","Business District","Harbour","West End","East Side"]

# ---------------------------------------------------------------------------
# 1) HOTEL INVENTORY — 180,000 records (3,000 per city)
# ---------------------------------------------------------------------------
hotel_rows = []
for city_idx, c in enumerate(CITIES):
    city, country, *_ = c
    base = 52 + city_idx % 11 * 4 + c[10] * 9
    for i in range(3000):
        stars = 1 + (i % 5)
        rating = round(min(5.0, 3.0 + stars * 0.24 + random.random() * 0.75), 1)
        nightly = int(base + stars * 15 + random.randint(-25, 95))
        hotel_rows.append({
            "hotel_id": f"H{city_idx+1:03d}{i+1:04d}", "city": city, "country": country,
            "name": f"{prefixes[(i+city_idx)%len(prefixes)]} {city} {suffixes[i%len(suffixes)]} {i+1:03d}",
            "area": areas[(i*3+city_idx)%len(areas)], "stars": stars, "rating": rating,
            "nightly_eur": max(28, nightly), "review_count": 25 + (i*137 + city_idx*53) % 9500,
            "breakfast_included": int((i+city_idx)%3 != 0), "refundable": int((i*2+city_idx)%4 != 0),
            "distance_to_center_km": round(0.1 + ((i*17+city_idx)%120)/10, 1),
            "wifi": 1, "workspace": int(i%2==0), "family_rooms": int(i%5==0),
            "air_conditioning": int((i+city_idx)%4 != 0), "parking": int(i%6 in {0,1}),
            "pet_friendly": int(i%7==0), "accessible": int(i%8!=3),
        })
with (RAW / "hotels.csv").open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=hotel_rows[0].keys()); w.writeheader(); w.writerows(hotel_rows)

# ---------------------------------------------------------------------------
# 2) ATTRACTIONS — 90,000 records (1,500 per city)
# ---------------------------------------------------------------------------
poi_types = ["museum","landmark","park","market","gallery","historic_site","viewpoint","food_hall","church","neighborhood","castle","science_museum","zoo","aquarium","botanical_garden"]
poi_rows=[]
for city_idx,c in enumerate(CITIES):
    city,country,*_=c
    for i in range(1500):
        typ=poi_types[i%len(poi_types)]
        poi_rows.append({
            "poi_id":f"P{city_idx+1:03d}{i+1:04d}","city":city,"country":country,
            "name":f"{city} {typ.replace('_',' ').title()} {i+1:03d}","category":typ,
            "rating":round(min(5.0,3.4+((i*13+city_idx)%17)/10),1),"estimated_visit_hours":round(0.5+(i%10)*0.5,2),
            "ticket_eur": [0,0,0,4,6,8,10,12,15,18,22,25,30,35,40][(i+city_idx)%15],
            "indoor":int(typ in {"museum","gallery","food_hall","church","science_museum","aquarium"}),
            "family_friendly":int(i%5!=1), "wheelchair_accessible":int(i%6!=2),
            "recommended_duration": ["quick","half_day","half_day","full_day"][(i+city_idx)%4],
        })
with (RAW / "attractions.csv").open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=poi_rows[0].keys()); w.writeheader(); w.writerows(poi_rows)

# ---------------------------------------------------------------------------
# 3) RESTAURANTS — 90,000 records (1,500 per city)
# ---------------------------------------------------------------------------
cuisines=["local","italian","asian","mediterranean","vegan","street_food","fine_dining","bakery","cafe","seafood","middle_eastern","indian"]
restaurant_rows=[]
for city_idx,c in enumerate(CITIES):
    city,country,*_=c
    for i in range(1500):
        cuisine=cuisines[(i+city_idx)%len(cuisines)]
        price_level=1+(i%4)
        restaurant_rows.append({
            "restaurant_id":f"R{city_idx+1:03d}{i+1:04d}","city":city,"country":country,
            "name":f"{city} {cuisine.replace('_',' ').title()} Kitchen {i+1:03d}","cuisine":cuisine,
            "rating":round(min(5.0,3.2+((i*11+city_idx)%18)/10),1),"price_level":price_level,
            "avg_meal_eur":round(8+price_level*9+((i*7+city_idx)%23),2),
            "vegetarian_options":int(i%3!=0),"vegan_options":int(i%4 in {0,1}),
            "reservation_recommended":int(price_level>=3),"family_friendly":int(i%5!=2),
            "area":areas[(i*5+city_idx)%len(areas)],
        })
with (RAW / "restaurants.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=restaurant_rows[0].keys());w.writeheader();w.writerows(restaurant_rows)

# ---------------------------------------------------------------------------
# 4) TRANSPORT PROFILES — richer city-level records
# ---------------------------------------------------------------------------
transport_rows=[]
for c in CITIES:
    city,country,code,currency,lang,tz,lat,lon,food,transport,score=c
    transport_rows.append({
        "city":city,"single_ticket_eur":round(max(1.2,transport/5),2),"day_pass_eur":transport,
        "three_day_pass_eur":round(transport*2.55,2),"weekly_pass_eur":round(transport*5.4,2),
        "airport_transfer_eur":round(transport*1.8+2,2),"bike_day_eur":round(8+transport*0.8,2),
        "taxi_start_eur":round(2.5+transport*0.25,2),"taxi_per_km_eur":round(0.9+transport*0.08,2),
        "walkability_score":round(min(10,5.8+score/1.5),1),
        "metro_available":int((int(abs(lat)*10)+len(city))%3!=0),"night_service":int(len(city)%2==0),
    })
with (RAW / "transport.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=transport_rows[0].keys());w.writeheader();w.writerows(transport_rows)

# ---------------------------------------------------------------------------
# 5) WEATHER FALLBACK — 1,200 days × 60 cities = 72,000 records
# ---------------------------------------------------------------------------
weather_rows=[]
ref=date(2026,9,12)
conditions=["clear","partly_cloudy","cloudy","light_rain","clear","clear","showers","windy","fog","heavy_rain"]
for city_idx,c in enumerate(CITIES):
    city=c[0]; lat=c[6]
    seasonal=19 - abs(lat-45)*0.22
    for d in range(1200):
        temp=seasonal + math.sin((city_idx+d)*0.31)*5.2 - d*0.055
        weather_rows.append({
            "city":city,"date":str(ref+timedelta(days=d)),"temp_min_c":round(temp-5.0,1),"temp_max_c":round(temp+4.5,1),
            "precipitation_mm":round(max(0, math.sin(city_idx*0.7+d*1.14))*9.5,1),
            "condition":conditions[(city_idx+d)%len(conditions)],"wind_kph":round(5+((city_idx*5+d*7)%42),1),
            "humidity_pct":45+((city_idx*11+d*7)%50),"uv_index":round(max(0,5.5-d*0.025+math.sin(d/8)*1.8),1),
        })
with (RAW / "weather_fallback.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=weather_rows[0].keys());w.writeheader();w.writerows(weather_rows)

# Offline cross-rate table relative to EUR.
fx = {
    "EUR":1.0,"HUF":395.0,"USD":1.17,"GBP":0.87,"CZK":24.4,"PLN":4.27,"CHF":0.94,"DKK":7.46,
    "SEK":10.95,"NOK":11.55,"RON":5.08,"BGN":1.96,"RSD":117.2,"BAM":1.96,"MKD":61.6,"ALL":97.0,
    "TRY":48.5,"ISK":143.0,"JPY":182.0,"CAD":1.61,"AUD":1.78,"NZD":1.91,"CNY":8.42,"KRW":1635.0,
    "SGD":1.51,"AED":4.29,"MXN":21.7,"BRL":6.35,"INR":103.0,"ZAR":20.3,
}
with (RAW / "fx_rates_fallback.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["base","quote","rate","snapshot_date","source"])
    for code,rate in fx.items(): w.writerow(["EUR",code,rate,"2026-09-12","offline-fallback-snapshot"])

# ---------------------------------------------------------------------------
# Query generation helpers
# ---------------------------------------------------------------------------
city_names=[c[0] for c in CITIES]
EN_TEMPLATES={
 "weather":["What will the weather be in {city} for {days} days?","Give me the {days}-day forecast for {city}.","Will it rain in {city} during my {days}-day trip?","Temperatures in {city} for the next {days} days please."],
 "hotel":["Find hotels in {city} for {days} nights under {price} EUR with rating at least {rating}.","I need a room in {city} for {days} nights, max {price} EUR per night.","Search accommodation in {city}: {days} nights, rating {rating}+ and under {price} EUR."],
 "attractions":["Show museums and landmarks in {city} under {ticket} EUR.","What are the best places to visit in {city}?","Find attractions in {city}, preferably museums, parks and historic sites."],
 "transport":["What local transport options and pass prices are available in {city} for {days} days?","How much should I budget for public transport in {city} for {days} days?","Do I need a day pass in {city} for a {days}-day visit?"],
 "restaurant":["Find good restaurants in {city} under {meal} EUR per person.","Show {cuisine} food in {city}, preferably rated 4 or better.","Where can I eat in {city} for around {meal} EUR?"],
 "location":["Give me destination information for {city}.","What currency, language and timezone does {city} use?","Practical travel information for {city}."],
}
HU_TEMPLATES={
 "weather":["Milyen idő lesz {city_hu} városában {days} napig?","Kérek {days} napos időjárás-előrejelzést {city_hu} területére.","Fog esni {city_hu} városában a következő {days} napban?"],
 "hotel":["Keress szállást {city_hu} városában {days} éjszakára, legfeljebb {price} EUR/éj áron.","{days} éjszakára megyek {city_hu} városába, legalább {rating} értékelésű hotelt szeretnék {price} EUR alatt."],
 "attractions":["Milyen látnivalók vannak {city_hu} városában?","Keress múzeumokat és nevezetességeket {city_hu} városában {ticket} EUR alatti belépővel."],
 "transport":["Mennyibe kerül a tömegközlekedés {city_hu} városában {days} napra?","Milyen bérletet vegyek {city_hu} városában {days} napra?"],
 "restaurant":["Keress jó éttermeket {city_hu} városában {meal} EUR/fő alatt.","Hol egyek {city_hu} városában körülbelül {meal} EUR-ból?"],
 "location":["Adj gyakorlati utazási információkat {city_hu} városáról.","Mi a pénznem és az időzóna {city_hu} városában?"],
}
HU_NAMES={"Vienna":"Bécs","Prague":"Prága","Paris":"Párizs","Rome":"Róma","Munich":"München","Krakow":"Krakkó"}

def render(template, city, days, price, rating, ticket, meal, cuisine):
    return template.format(city=city,city_hu=HU_NAMES.get(city,city),days=days,price=price,rating=rating,ticket=ticket,meal=meal,cuisine=cuisine)

# ---------------------------------------------------------------------------
# 6) ROUTING TRAIN/VALIDATION/TEST — 50,000 labeled queries
# ---------------------------------------------------------------------------
route_rows=[]
intent_names=["weather","hotel","attractions","transport","restaurant","location"]
for i in range(50000):
    city=city_names[(i*17+3)%len(city_names)];days=1+(i%10);price=70+(i%13)*15;rating=round(3.5+(i%4)*0.4,1);ticket=(i%8)*5+5;meal=12+(i%10)*4;cuisine=cuisines[i%len(cuisines)]
    # single-intent 60%, multi-intent 40%
    k=1 if i%5<3 else (2 if i%5==3 else 3)
    chosen=[]
    for j in range(k):
        name=intent_names[(i+j*2)%len(intent_names)]
        if name not in chosen:chosen.append(name)
    lang="hu" if i%4==0 else "en"
    templates=HU_TEMPLATES if lang=="hu" else EN_TEMPLATES
    parts=[render(templates[name][(i+j)%len(templates[name])],city,days,price,rating,ticket,meal,cuisine) for j,name in enumerate(chosen)]
    query=" ".join(parts)
    route_rows.append({
        "query_id":f"RT{i+1:05d}","split":"train" if i<40000 else ("validation" if i<45000 else "test"),
        "language":lang,"query":query,"tool_labels":"|".join(chosen),"city":city,"days":days,
        "hotel_max_price_eur":price if "hotel" in chosen else "","hotel_min_rating":rating if "hotel" in chosen else "",
        "restaurant_max_meal_eur":meal if "restaurant" in chosen else "","attraction_max_ticket_eur":ticket if "attractions" in chosen else "",
    })
# Add 10,000 harder training examples with indirect wording, distractors and explicit exclusions.
hard_train_templates={
 "weather":["Do I need an umbrella or jacket in {city} over the next {days} days?","What should I wear outdoors in {city} this week?","Will outdoor plans work in {city} for {days} days?"],
 "hotel":["I still need somewhere to sleep in {city} for {days} nights, budget {price} EUR.","Find me a place to stay overnight in {city}.","I have not booked accommodation in {city}; help for {days} nights."],
 "attractions":["Help me fill a sightseeing afternoon in {city}.","What cultural things are worth seeing in {city}?","Give me ideas for what to see around {city}."],
 "transport":["How can I get around {city} without renting a car?","What is the easiest way to move around {city} for {days} days?","I need a pass for getting around {city}."],
 "restaurant":["Where should we have dinner in {city}?","Recommend somewhere good to eat in {city} around {meal} EUR.","I need food options in {city}."],
 "location":["What should I know before I land in {city}?","Give me practical basics before visiting {city}.","Essential local facts for a first trip to {city}."],
}
for i in range(10000):
    city=city_names[(i*23+5)%len(city_names)];days=1+(i%9);price=75+(i%10)*20;meal=15+(i%8)*5
    primary=intent_names[i%len(intent_names)]
    chosen=[primary]
    parts=[hard_train_templates[primary][i%len(hard_train_templates[primary])].format(city=city,days=days,price=price,meal=meal)]
    if i%3==0:
        second=intent_names[(i+2)%len(intent_names)]
        if second not in chosen:
            chosen.append(second);parts.append(hard_train_templates[second][(i+1)%len(hard_train_templates[second])].format(city=city,days=days,price=price,meal=meal))
    # Negative keyword examples teach that mentioning a tool can explicitly mean NOT to call it.
    if "hotel" not in chosen and i%7==0: parts.append("I already booked my hotel, so do not search accommodation.")
    if "restaurant" not in chosen and i%11==0: parts.append("I do not need restaurant or food recommendations.")
    route_rows.append({
        "query_id":f"HT{i+1:05d}","split":"train","language":"en","query":" ".join(parts),"tool_labels":"|".join(chosen),
        "city":city,"days":days,"hotel_max_price_eur":price if "hotel" in chosen else "","hotel_min_rating":"",
        "restaurant_max_meal_eur":meal if "restaurant" in chosen else "","attraction_max_ticket_eur":"",
    })

with (RAW / "intent_router_dataset.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=route_rows[0].keys());w.writeheader();w.writerows(route_rows)

# ---------------------------------------------------------------------------
# 7) NATURAL-LANGUAGE QUERY CORPUS — 25,000 examples
# ---------------------------------------------------------------------------
queries=[]
for i in range(25000):
    row=route_rows[i]
    queries.append({"query_id":f"Q{i+1:05d}","language":row["language"],"query":row["query"]})
with (RAW / "sample_user_queries.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=queries[0].keys());w.writeheader();w.writerows(queries)

# ---------------------------------------------------------------------------
# 8) AGENT BENCHMARK — 5,000 ground-truth tasks, including mixed intents
# ---------------------------------------------------------------------------
bench=[]
for i in range(5000):
    city=city_names[i%len(city_names)];days=1+(i%7);price=90+(i%8)*20;rating=4.0;ticket=20.0;meal=35.0
    mode=i%15
    calls=[]
    if mode==0:
        q=f"What will the weather be in {city} for {days} days?";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}}]
    elif mode==1:
        q=f"Find hotels in {city} for {days} nights under {price} EUR with rating at least 4.0.";calls=[{"name":"search_hotels","arguments":{"city":city,"nights":days,"max_price_per_night_eur":float(price),"min_rating":4.0,"top_k":5}}]
    elif mode==2:
        q=f"Show me top museums and landmarks in {city}, maximum ticket 20 EUR.";calls=[{"name":"search_attractions","arguments":{"city":city,"categories":["museum","landmark"],"max_ticket_eur":20.0,"top_k":6}}]
    elif mode==3:
        q=f"What local transport options and pass prices are available in {city} for {days} days?";calls=[{"name":"get_transport_options","arguments":{"city":city,"days":days}}]
    elif mode==4:
        q="Convert 500 EUR to HUF.";calls=[{"name":"convert_currency","arguments":{"amount":500.0,"from_currency":"EUR","to_currency":"HUF"}}]
    elif mode==5:
        q=f"Give me destination information for {city}.";calls=[{"name":"get_location_info","arguments":{"city":city}}]
    elif mode==6:
        q=f"Find restaurants in {city} under 35 EUR per person.";calls=[{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":35.0,"min_rating":0.0,"vegetarian_only":False,"top_k":6}}]
    elif mode==7:
        q=f"Plan a {days}-day stay in {city}: weather and a hotel under {price} EUR per night.";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}},{"name":"search_hotels","arguments":{"city":city,"nights":days,"max_price_per_night_eur":float(price),"min_rating":0.0,"top_k":5}}]
    elif mode==8:
        q=f"For {city}, find attractions and tell me transport prices for {days} days.";calls=[{"name":"search_attractions","arguments":{"city":city,"categories":None,"max_ticket_eur":None,"top_k":6}},{"name":"get_transport_options","arguments":{"city":city,"days":days}}]
    elif mode==9:
        q=f"I am going to {city} for {days} days. Give me weather, hotel options, attractions and transport information.";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}},{"name":"search_hotels","arguments":{"city":city,"nights":days,"max_price_per_night_eur":None,"min_rating":0.0,"top_k":5}},{"name":"search_attractions","arguments":{"city":city,"categories":None,"max_ticket_eur":None,"top_k":6}},{"name":"get_transport_options","arguments":{"city":city,"days":days}}]
    elif mode==10:
        q=f"Find a hotel and restaurants in {city} for {days} nights. Keep the hotel under {price} EUR and meals under 35 EUR.";calls=[{"name":"search_hotels","arguments":{"city":city,"nights":days,"max_price_per_night_eur":float(price),"min_rating":0.0,"top_k":5}},{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":35.0,"min_rating":0.0,"vegetarian_only":False,"top_k":6}}]
    elif mode==11:
        q=f"In {city}, suggest museums and restaurants, and show the weather for {days} days.";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}},{"name":"search_attractions","arguments":{"city":city,"categories":["museum"],"max_ticket_eur":None,"top_k":6}},{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":None,"min_rating":0.0,"vegetarian_only":False,"top_k":6}}]
    elif mode==12:
        q=f"Give me destination information, transport and restaurants for {city}.";calls=[{"name":"get_location_info","arguments":{"city":city}},{"name":"get_transport_options","arguments":{"city":city,"days":3}},{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":None,"min_rating":0.0,"vegetarian_only":False,"top_k":6}}]
    elif mode==13:
        q=f"I have {days} days in {city}. No hotel needed. Show weather, attractions, transport and food options.";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}},{"name":"search_attractions","arguments":{"city":city,"categories":None,"max_ticket_eur":None,"top_k":6}},{"name":"get_transport_options","arguments":{"city":city,"days":days}},{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":None,"min_rating":0.0,"vegetarian_only":False,"top_k":6}}]
    else:
        q=f"Plan {days} days in {city}: weather, hotel under {price} EUR, attractions, restaurants and transport.";calls=[{"name":"get_weather","arguments":{"city":city,"days":days,"unit":"celsius"}},{"name":"search_hotels","arguments":{"city":city,"nights":days,"max_price_per_night_eur":float(price),"min_rating":0.0,"top_k":5}},{"name":"search_attractions","arguments":{"city":city,"categories":None,"max_ticket_eur":None,"top_k":6}},{"name":"search_restaurants","arguments":{"city":city,"cuisines":None,"max_meal_eur":None,"min_rating":0.0,"vegetarian_only":False,"top_k":6}},{"name":"get_transport_options","arguments":{"city":city,"days":days}}]
    bench.append({"id":f"case_{i+1:05d}","query":q,"expected_calls":calls})
with (BENCH / "agent_tasks.json").open("w",encoding="utf-8") as f: json.dump(bench,f,ensure_ascii=False,indent=2)
with (BENCH / "agent_tasks.jsonl").open("w",encoding="utf-8") as f:
    for row in bench:f.write(json.dumps(row,ensure_ascii=False)+"\n")

# ---------------------------------------------------------------------------
# 9) CHALLENGE ROUTING SET — 5,000 indirect/noisy/negated requests
# ---------------------------------------------------------------------------
challenge=[]
challenge_templates=[
    ("weather", ["Should I pack an umbrella for {city} for the next {days} days?", "Will I need a jacket in {city} this week?", "Outdoor conditions in {city} for my {days}-day visit?"]),
    ("hotel", ["Where can I sleep in {city} for {days} nights below {price} EUR?", "I still need a place to stay overnight in {city}; budget {price} EUR.", "Find somewhere to stay in {city} for {days} nights."]),
    ("attractions", ["Fill an afternoon in {city} with sightseeing ideas.", "What should I see in {city} if I like culture?", "Build me a list of things worth seeing in {city}."]),
    ("transport", ["How do I get around {city} without a car for {days} days?", "Best way to move around {city} for {days} days?", "What should I buy for getting around {city}?"]),
    ("restaurant", ["Where should we have dinner in {city}?", "I want somewhere good to eat in {city} around {meal} EUR.", "Food recommendations in {city}, please."]),
    ("location", ["What should I know before landing in {city}?", "Practical basics before visiting {city}.", "Tell me the essential local facts about {city}."]),
]
for i in range(5000):
    city=city_names[(i*19+11)%len(city_names)];days=1+(i%8);price=80+(i%9)*20;meal=15+(i%9)*5
    k=1 if i%4<2 else 2
    labels=[];parts=[]
    for j in range(k):
        name,templates=challenge_templates[(i+j*3)%len(challenge_templates)]
        if name not in labels:labels.append(name)
        parts.append(templates[(i+j)%len(templates)].format(city=city,days=days,price=price,meal=meal))
    # Add distractors/explicit exclusions in a subset; excluded tools are not labels.
    if i%10==0:parts.append("I already booked accommodation, so do not look for a hotel.")
    if i%14==0:parts.append("No restaurant suggestions are needed.")
    challenge.append({"query_id":f"CH{i+1:05d}","query":" ".join(parts),"tool_labels":"|".join(labels),"city":city})
with (RAW / "intent_router_challenge.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=challenge[0].keys());w.writeheader();w.writerows(challenge)

manifest={
    "generated_at":"2026-09-12",
    "files":{
        "cities.csv":len(CITIES),"hotels.csv":len(hotel_rows),"attractions.csv":len(poi_rows),"restaurants.csv":len(restaurant_rows),
        "transport.csv":len(transport_rows),"weather_fallback.csv":len(weather_rows),"fx_rates_fallback.csv":len(fx),
        "sample_user_queries.csv":len(queries),"intent_router_dataset.csv":len(route_rows),"intent_router_challenge.csv":len(challenge),"benchmark/agent_tasks.json":len(bench),
    },
    "splits":{"intent_router_train":50000,"intent_router_validation":5000,"intent_router_test":5000},
    "notes":{
        "hotels_attractions_restaurants":"Synthetic deterministic portfolio data; not bookable/live.",
        "weather":"Local fallback fixture; live mode uses Open-Meteo when network is available.",
        "fx":"Local fallback snapshot; live mode uses Frankfurter when network is available.",
        "intent_router":"Synthetic bilingual labeled corpus designed for multi-label tool-routing experiments.",
    }
}
(PROCESSED / "dataset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,indent=2))
