from pathlib import Path
try:
    from graphviz import Source
except ImportError as exc:
    raise SystemExit("Install graphviz Python package: pip install graphviz") from exc

ROOT=Path(__file__).resolve().parents[1]
dot_path=ROOT/"05_graphviz/workflow.dot"
out=ROOT/"04_results/figures/workflow_graph"
Source(dot_path.read_text(encoding="utf-8")).render(str(out), format="png", cleanup=True)
print(f"Saved {out}.png")
