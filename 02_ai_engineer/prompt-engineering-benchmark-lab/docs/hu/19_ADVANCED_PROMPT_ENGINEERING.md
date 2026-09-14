# Advanced Prompt Engineering — részletes útmutató

## Prompt anatomy
A P16 külön kezeli a persona, instruction, context, audience, tone, reference data, examples, constraints, untrusted input és output format blokkokat. Ezek hipotézisek, nem dogmák: a benchmark azt méri, hogy melyik komponens ad valódi quality gain-t a token/latency overheadhez képest.

## P0–P16 értelmezés
P0 kontrollcsoport. P1 label-szemantika; P2 system role; P3 few-shot; P4 constraints; P5 decision policy; P6 prompt-only JSON; P7 provider-enforced structured output; P8 persona; P9 explicit instruction/context blockok; P10 machine audience/tone/format; P11 delimitált untrusted data; P12 contrastive példák; P13 reasoning-mode konfiguráció; P14 független branch-and-vote; P15 grammar/schema constrained generation; P16 a teljes advanced architektúra.

## Reasoning biztonság
A projekt nem kér és nem tárol privát chain-of-thoughtot. P13 provider által exponált reasoning controlt használ, ahol támogatott. P14 több független végső label döntést aggregál determinisztikus vote-tal; csak a végső döntések tárolódnak.

## Mit mérünk?
Quality, output validity, token, latency, cost, uncertainty, scenario performance, fixed/regressed példák és hard-case viselkedés. A leghosszabb prompt nem automatikusan a legjobb production prompt.

## Anti-patternök
Ne optimalizálj final holdouton, ne mutass mock score-t valódi LLM evidence-ként, ne változtass modellt és promptot egyszerre prompt-only experimentben, és ne keverd kontroll nélkül a decoding paramétert a promptváltozással.
