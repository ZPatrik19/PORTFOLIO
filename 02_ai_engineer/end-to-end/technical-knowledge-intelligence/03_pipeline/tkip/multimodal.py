"""Optional multimodal enrichment for figures extracted from source documents."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from .config import gemini_api_key
from .logging_config import get_logger
from .models import ParsedBlock

LOGGER = get_logger(__name__)


class MultimodalFigureAnalyzer:
    """Describe extracted technical figures with Gemini when explicitly enabled."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.client = None
        api_key = gemini_api_key()
        if not api_key:
            return

        try:
            from google import genai

            self.client = genai.Client(api_key=api_key)
        except (ImportError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Gemini multimodal client unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self.client is not None

    def describe(self, image_path: Path, context: str = "") -> str | None:
        """Return a retrieval-oriented figure description or ``None`` when disabled."""

        if not self.available:
            return None

        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        prompt = (
            "Describe this technical figure for retrieval. Identify diagram type, labeled components, "
            "equations/code if readable, and the concept it supports. Do not invent unreadable text."
        )
        if context:
            prompt += f" Nearby section context: {context[:1500]}"

        interaction = self.client.interactions.create(
            model=self.config["gemini"]["model"],
            input=[
                {"type": "image", "data": image_base64, "mime_type": mime_type},
                {"type": "text", "text": prompt},
            ],
        )
        return interaction.output_text


def enrich_figure_blocks(
    blocks: list[ParsedBlock],
    config: dict[str, Any],
    project_root: Path,
) -> list[ParsedBlock]:
    """Optionally add Gemini-generated figure descriptions to parsed figure blocks."""

    multimodal_config = config.get("multimodal", {})
    if not multimodal_config.get("enabled", True):
        return blocks
    if not multimodal_config.get("analyze_figures_during_ingestion", False):
        return blocks

    analyzer = MultimodalFigureAnalyzer(config)
    if not analyzer.available:
        return blocks

    for block in blocks:
        if block.block_type != "figure" or not block.asset_path:
            continue
        try:
            description = analyzer.describe(
                project_root / block.asset_path,
                block.section or "",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning(
                "Figure analysis failed for %s: %s",
                block.asset_path,
                exc,
            )
            block.text = f"{block.text}\n[Figure analysis unavailable: {type(exc).__name__}]"
            continue

        if description:
            block.text = f"{block.text}\nGemini figure description: {description}"

    return blocks
