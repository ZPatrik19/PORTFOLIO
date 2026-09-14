from __future__ import annotations

from pathlib import Path

from prompt_benchmark.paths import PATHS

import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _box(ax, x: float, y: float, text: str, width: float = 0.17, height: float = 0.12) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "fill": False, "linewidth": 1.5},
    )


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "linewidth": 1.5})


def experiment_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = [0.07, 0.21, 0.35, 0.49, 0.63, 0.77, 0.91]
    labels = [
        "Public\nlabelled data",
        "Clean +\nvalidate",
        "Leakage-safe\nsplits",
        "Prompt\nstrategies P0–P16",
        "Same model +\nsame holdout",
        "Metrics +\nerror analysis",
        "Production\ntrade-off decision",
    ]
    for x, label in zip(xs, labels):
        _box(ax, x, 0.52, label)
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(ax, left + 0.055, 0.52, right - 0.055, 0.52)
    ax.set_title("Prompt Engineering Benchmark — Experimental Pipeline", fontsize=14, pad=16)
    ax.text(0.5, 0.16, "Only the prompt strategy changes in the final benchmark; the model, holdout and evaluation logic stay fixed.", ha="center", fontsize=10)
    _save(fig, path)


def leakage_safe_split(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.12, 0.55, "Clean source\ndataset")
    _box(ax, 0.38, 0.78, "Few-shot\nexamples")
    _box(ax, 0.38, 0.50, "Development\nset")
    _box(ax, 0.38, 0.22, "Final holdout\nbenchmark")
    _arrow(ax, 0.19, 0.55, 0.31, 0.78)
    _arrow(ax, 0.19, 0.55, 0.31, 0.50)
    _arrow(ax, 0.19, 0.55, 0.31, 0.22)
    _box(ax, 0.70, 0.50, "Prompt tuning +\nablation only")
    _box(ax, 0.70, 0.22, "One-time final\nevaluation")
    _arrow(ax, 0.45, 0.50, 0.62, 0.50)
    _arrow(ax, 0.45, 0.22, 0.62, 0.22)
    ax.text(0.73, 0.78, "No overlap", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.73, 0.70, "few-shot ≠ development ≠ holdout", ha="center", va="center", fontsize=10)
    ax.set_title("Leakage-Safe Data Design", fontsize=14, pad=16)
    _save(fig, path)


def prompt_strategy_ladder(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    groups = [
        ("P0–P2", "Baseline + definitions + role", "Core prompting"),
        ("P3–P5", "Few-shot + constraints + policy", "Demonstration & control"),
        ("P6–P7", "JSON + Structured Output", "Machine-readable output"),
        ("P8–P11", "Persona + context + audience/tone + delimiters", "Prompt anatomy"),
        ("P12", "Contrastive few-shot", "Advanced examples"),
        ("P13", "Reasoning-model mode", "Reasoning control"),
        ("P14", "Tree-inspired branch + vote", "Multi-branch execution"),
        ("P15", "Grammar/schema constrained", "Constrained generation"),
        ("P16", "Full advanced template", "Combined prompt architecture"),
    ]
    y_values = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    for (pid, name, why), y in zip(groups, y_values):
        _box(ax, 0.14, y, pid)
        _box(ax, 0.47, y, name)
        _box(ax, 0.81, y, why)
        _arrow(ax, 0.21, y, 0.34, y)
        _arrow(ax, 0.59, y, 0.69, y)
    ax.set_title("Prompt Engineering Technique Ladder — P0 to P16", fontsize=14, pad=12)
    _save(fig, path)


def advanced_prompt_anatomy(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    labels = [
        (0.12, 0.78, "Persona"), (0.36, 0.78, "Instruction"), (0.60, 0.78, "Context"), (0.84, 0.78, "Audience + Tone"),
        (0.12, 0.42, "Reference Data"), (0.36, 0.42, "Examples"), (0.60, 0.42, "Constraints"), (0.84, 0.42, "Format / Schema"),
    ]
    for x,y,label in labels: _box(ax,x,y,label)
    _box(ax, 0.50, 0.12, "Final Prompt Template")
    for x,y,_ in labels: _arrow(ax,x,y-0.06,0.50,0.19)
    ax.set_title("Advanced Prompt Anatomy: What P16 Combines", fontsize=14, pad=14)
    _save(fig, path)


def decoding_parameters(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    _box(ax,0.16,0.66,"Temperature\nrandomness")
    _box(ax,0.50,0.66,"Top-p\ncumulative probability")
    _box(ax,0.84,0.66,"Top-k\nmax candidate tokens")
    _box(ax,0.50,0.28,"One-variable-at-a-time\nparameter sweep")
    for x in (0.16,0.50,0.84): _arrow(ax,x,0.58,0.50,0.36)
    ax.text(0.5,0.08,"Prompt strategy is held fixed so decoding effects are not confused with prompt effects.",ha="center",fontsize=10)
    ax.set_title("Decoding Parameter Experiment Design", fontsize=14, pad=14)
    _save(fig,path)


def tree_branch_vote(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    _box(ax,0.10,0.52,"Ticket")
    ys=[0.78,0.52,0.26]
    labs=["Lexical expert","Policy expert","Ambiguity expert"]
    for y,label in zip(ys,labs):
        _box(ax,0.42,y,label); _arrow(ax,0.17,0.52,0.32,y); _arrow(ax,0.52,y,0.70,0.52)
    _box(ax,0.79,0.52,"Majority vote")
    _box(ax,0.94,0.52,"Final label")
    _arrow(ax,0.86,0.52,0.90,0.52)
    ax.text(0.5,0.08,"No hidden chain-of-thought is collected; only final branch labels are aggregated.",ha="center",fontsize=10)
    ax.set_title("Tree-of-Thought-Inspired Branch-and-Vote",fontsize=14,pad=14)
    _save(fig,path)


def production_tradeoff(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.50, 0.78, "Quality\nMacro F1 + per-class F1")
    _box(ax, 0.22, 0.28, "Efficiency\nTokens + cost + latency")
    _box(ax, 0.78, 0.28, "Reliability\nValidity + parsing + errors")
    _box(ax, 0.50, 0.48, "Selected\nproduction prompt")
    _arrow(ax, 0.50, 0.69, 0.50, 0.56)
    _arrow(ax, 0.30, 0.33, 0.43, 0.44)
    _arrow(ax, 0.70, 0.33, 0.57, 0.44)
    ax.text(0.5, 0.10, "The highest F1 is not automatically the best production choice.", ha="center", fontsize=11)
    ax.set_title("Multi-Objective Prompt Selection", fontsize=14, pad=14)
    _save(fig, path)



def free_provider_options(path: Path) -> None:
    """Visual decision map for choosing a benchmark provider without paying."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.50, 0.88, "How do I run the benchmark?")
    _box(ax, 0.13, 0.61, "MOCK\nNo key\nPipeline test only")
    _box(ax, 0.38, 0.61, "OLLAMA\nLocal real LLM\nNo API bill")
    _box(ax, 0.63, 0.61, "GROQ\nCloud real LLM\nFree-plan quota")
    _box(ax, 0.87, 0.61, "GEMINI\nCloud real LLM\nFree-tier quota")
    for x in (0.13, 0.38, 0.63, 0.87):
        _arrow(ax, 0.50, 0.82, x, 0.69)

    _box(ax, 0.13, 0.30, "Use for\nsmoke tests + demos")
    _box(ax, 0.38, 0.30, "Best zero-bill\nreal benchmark")
    _box(ax, 0.63, 0.30, "Fast cloud\nexperiments")
    _box(ax, 0.87, 0.30, "Alternative cloud\nbenchmark")
    for x in (0.13, 0.38, 0.63, 0.87):
        _arrow(ax, x, 0.53, x, 0.38)

    ax.text(
        0.50, 0.09,
        "Mock proves the software pipeline. Ollama/Groq/Gemini provide real LLM evidence. OpenAI remains an optional comparison provider.",
        ha="center", fontsize=10,
    )
    ax.set_title("Provider Decision Map — Free-First Benchmarking", fontsize=14, pad=14)
    _save(fig, path)

def generate_all(output_dir: str | Path = PATHS.figures) -> None:
    output_dir = Path(output_dir)
    experiment_pipeline(output_dir / "00_experiment_pipeline.png")
    leakage_safe_split(output_dir / "00b_leakage_safe_split.png")
    prompt_strategy_ladder(output_dir / "00c_prompt_strategy_ladder.png")
    production_tradeoff(output_dir / "00d_production_tradeoff.png")
    free_provider_options(output_dir / "00e_free_provider_options.png")
    advanced_prompt_anatomy(output_dir / "00f_advanced_prompt_anatomy.png")
    decoding_parameters(output_dir / "00g_decoding_parameters.png")
    tree_branch_vote(output_dir / "00h_tree_branch_vote.png")
