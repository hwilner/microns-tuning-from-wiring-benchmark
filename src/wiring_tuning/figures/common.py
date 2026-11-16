"""Shared plotting helpers for deterministic figure generation."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt

#: Reference fonts by name instead of embedding glyph paths: keeps the
#: committed SVGs small and text selectable/searchable. DejaVu Sans is
#: matplotlib's default and universally available.
plt.rcParams["svg.fonttype"] = "none"

#: Fixed, colour-blind-friendly palette shared by all figures.
PALETTE = {
    "features_mlp": "#0072B2",
    "gnn": "#D55E00",
    "degree_mlp": "#009E73",
    "mean_null": "#999999",
    "neutral": "#56B4E9",
    "accent": "#E69F00",
}

#: Fixed SVG metadata so regenerated figures are byte-identical (the
#: default embeds a wall-clock date).
SVG_METADATA = {"Date": None, "Creator": "wiring_tuning figure pipeline"}

#: Matplotlib salts generated element ids per process; match them so they
#: can be renumbered deterministically after rendering.
_ID_PATTERN = re.compile(
    r'(?<=id=")[a-zA-Z][0-9a-f]{8,}(?=")'
    r'|(?<=url\(#)[a-zA-Z][0-9a-f]{8,}(?=\))'
    r'|(?<=xlink:href="#)[a-zA-Z][0-9a-f]{8,}(?=")'
)


def _normalise_svg_ids(path: Path) -> None:
    """Renumber random element ids in a rendered SVG in order of appearance.

    Matplotlib derives clip-path ids from a salted hash, so two renders of
    the same figure differ only in these ids. Renumbering them in order of
    first appearance makes the output byte-identical across runs and
    processes.

    Args:
        path: SVG file to rewrite in place.
    """
    text = path.read_text()
    mapping: dict[str, str] = {}

    def _sub(match: re.Match) -> str:
        token = match.group(0)
        if token not in mapping:
            mapping[token] = f"n{len(mapping):04d}"
        return mapping[token]

    text = _ID_PATTERN.sub(_sub, text)
    # Canonicalise line endings: no trailing whitespace, single trailing
    # newline, so the committed artifact is diff-friendly.
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(text)


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Save a matplotlib figure as a deterministic SVG.

    Args:
        fig: figure to save.
        path: output SVG path; parent directories are created.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)
    _normalise_svg_ids(path)
    return path
