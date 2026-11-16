"""Formatting helpers for the paper tables.

All helpers are pure functions of their arguments so table regeneration is
deterministic: same committed artifacts in, byte-identical tables out.
"""

from __future__ import annotations

import pandas as pd


def format_mean_ci(mean: float, ci: float, decimals: int = 3) -> str:
    """Format a mean ± 95% CI pair with fixed decimals.

    Args:
        mean: point estimate.
        ci: half-width of the 95% confidence interval.
        decimals: number of decimal places.

    Returns:
        A string like ``"0.060 ± 0.015"``. Negative zero is normalised to
        zero so output never depends on float sign noise.
    """

    def _fix(x: float) -> float:
        return 0.0 if x == 0 else x

    return f"{_fix(mean):.{decimals}f} ± {_fix(ci):.{decimals}f}"


def latex_escape(text: object) -> str:
    """Escape the minimal set of LaTeX special characters in table cells.

    Args:
        text: cell content.

    Returns:
        The cell content safe for a LaTeX tabular.
    """
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("±", r"$\pm$")
        .replace("≈", r"$\approx$")
        .replace("−", "-")
    )


def to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Args:
        frame: table to render; the index is dropped.

    Returns:
        Markdown table string with a trailing newline.
    """
    cols = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def to_latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    """Render a DataFrame as a LaTeX booktabs-style table.

    Args:
        frame: table to render; the index is dropped.
        caption: table caption.
        label: LaTeX label for cross-referencing.

    Returns:
        A complete ``table`` environment string with a trailing newline.
    """
    cols = [str(c) for c in frame.columns]
    spec = "l" * len(cols)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{latex_escape(caption)}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{spec}}}",
        "    \\toprule",
        "    " + " & ".join(latex_escape(c) for c in cols) + " \\\\",
        "    \\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append("    " + " & ".join(latex_escape(v) for v in row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def write_table(
    frame: pd.DataFrame,
    out_base: "str | 'Path'",
    caption: str,
    label: str,
) -> "list['Path']":
    """Write one table as CSV, Markdown and LaTeX side by side.

    Args:
        frame: table to write; the index is dropped before writing.
        out_base: output path without extension; ``.csv``, ``.md`` and
            ``.tex`` are appended.
        caption: caption for the LaTeX table.
        label: LaTeX label.

    Returns:
        The list of written paths, in ``csv, md, tex`` order.
    """
    from pathlib import Path

    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True)
    paths = [base.with_suffix(s) for s in (".csv", ".md", ".tex")]
    frame.to_csv(paths[0], index=False, lineterminator="\n")
    paths[1].write_text(to_markdown(frame))
    paths[2].write_text(to_latex(frame, caption, label))
    return paths
