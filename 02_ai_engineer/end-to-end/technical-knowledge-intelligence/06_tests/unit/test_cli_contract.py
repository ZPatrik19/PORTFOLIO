from __future__ import annotations

from tkip.cli import build_parser


def test_cli_parses_ingest_command() -> None:
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["ingest"])

    # Assert
    assert args.command == "ingest"


def test_cli_parses_ask_configuration() -> None:
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(
        [
            "ask",
            "How does RAG work?",
            "--index-variant",
            "semantic",
            "--prompt-optimization",
            "technical_deep_dive",
        ]
    )

    # Assert
    assert args.command == "ask"
    assert args.question == "How does RAG work?"
    assert args.index_variant == "semantic"
    assert args.prompt_optimization == "technical_deep_dive"


def test_cli_index_variants_has_safe_defaults() -> None:
    # Arrange
    parser = build_parser()

    # Act
    args = parser.parse_args(["index-variants"])

    # Assert
    assert "fixed" in args.strategies
    assert "recursive" in args.strategies
    assert "semantic" in args.strategies
    assert args.force is False
