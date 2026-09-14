# Temperature, Top-p és Top-k – miért külön benchmark?

A prompt tartalma és a decoding paraméter két külön független változó. Ha egyszerre módosítjuk őket, nem tudjuk megmondani, mi okozta a teljesítményváltozást.

## Temperature
A mintavételezés véletlenszerűségét szabályozza. Klasszifikációnál általában alacsony érték indokolt, mert stabil outputot akarunk.

A projekt alap sweepje:

`0.0, 0.2, 0.5, 0.8`

## Top-p
Nucleus sampling: csak addig a tokenhalmazig mintavételezünk, amelynek kumulatív valószínűsége eléri a `top_p` küszöböt.

Sweep:

`0.5, 0.8, 0.95, 1.0`

## Top-k
Legfeljebb a K legvalószínűbb következő tokenből enged mintavételezni.

Sweep:

`10, 20, 40, 80`

## Provider capability

- Ollama: temperature + top_p + top_k.
- Gemini: temperature + top_p, valamint csak olyan modelleknél top_k, amelyek ezt engedélyezik.
- Groq adapter: temperature + top_p; top_k-t nem küldünk az API-nak.
- OpenAI adapter: temperature + top_p; top_k nincs kitéve ebben a benchmark adapterben.
- Mock: mindhárom mezőt naplózza, de a mock classifier determinisztikus, ezért a sampling sweepje NEM modellkutatási eredmény.

## One-variable-at-a-time elv

A `05_scripts/06_run_parameter_sweep.py` egy paramétert változtat egyszerre. Ez csökkenti a confoundingot.

Eredmény:

`07_outputs/results/parameter_sweeps/<provider>/parameter_sweep_summary.csv`

Ábrák:

`07_outputs/reports/figures/<provider>/parameter_sweeps/`
