# Prompt Engineering

## Kísérleti alapelv
A prompt design független változó. A benchmark ugyanazt a datasetet, model/provider környezetet, parser- és metric implementációt, valamint evaluation protokollt tartja fixen, miközben a promptstratégiát változtatja.

## Stratégialétra
- P0: naiv zero-shot baseline.
- P1: explicit kategóriadefiníciók.
- P2: role/system instruction.
- P3: few-shot példák.
- P4: explicit constraints.
- P5: decision policy / strukturált reasoning privát chain-of-thought kikérése nélkül.
- P6: prompt-only JSON output.
- P7: schema-constrained structured output.
- P8–P12: persona, context/instruction szétválasztás, audience/tone, delimitált user data, contrastive példák.
- P13: provider által támogatott reasoning konfiguráció.
- P14: Tree-of-Thought ihletésű független ágak + vote; csak a végső labelek kerülnek tárolásra.
- P15: grammar/schema-constrained generation.
- P16: teljes advanced prompt template.

## Custom promptok
A felhasználó system és user template-et adhat meg, strukturált output/reasoning beállítást konfigurálhat, presetet menthet, Playgroundban futtathatja, majd P0–P16 stratégiákkal benchmarkolhatja.

## Fair-comparison szabály
A promptstratégiák összehasonlítása csak azonos sample-ek és azonos model/provider beállítások mellett értelmes. Ezért a sampling-parameter sweep külön experiment a promptstratégia benchmarktól.
