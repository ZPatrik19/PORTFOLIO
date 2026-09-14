# Ábraértelmezési útmutató

## Quality bizonytalansággal
Accuracy Wilson CI-val és Macro F1 mintaszámmal együtt értelmezendő. Kis mintán az 1.0 score bizonytalanság nélkül félrevezető.

## Quality vs token / latency / cost
Azt mutatja, hogy a quality gain indokolja-e az operational overheadet. Pareto szempontból gyenge az a prompt, amely drágább/lassabb, de nem jobb.

## Scenario heatmap
Megmutatja, *hol* segít a stratégia: ambiguity, prompt injection, long context, multi-intent stb. Gyakran informatívabb egy globális F1-nél.

## Confusion matrix
Megmutatja a keveredő intent párokat. Az `__invalid__` parse/contract hiba, nem szemantikai osztály.

## Fixed vs regressed
Baseline-nal párosított összehasonlításban különválasztja a javított és az elrontott példákat.

## Parameter sweep
Csak fix prompt mellett értelmezhető, különben a decoding és prompt hatás összekeveredik.
