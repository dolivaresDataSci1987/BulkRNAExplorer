from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def venn_figure(named_sets: Mapping[str, set[str]], title: str = "DEG overlap"):
    """Create a 2- or 3-set Venn diagram."""
    items = list(named_sets.items())
    if len(items) not in (2, 3):
        raise ValueError("A Venn diagram requires exactly 2 or 3 comparisons.")

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    if len(items) == 2:
        venn2(
            subsets=(items[0][1], items[1][1]),
            set_labels=(items[0][0], items[1][0]),
            ax=ax,
        )
    else:
        venn3(
            subsets=(items[0][1], items[1][1], items[2][1]),
            set_labels=(items[0][0], items[1][0], items[2][0]),
            ax=ax,
        )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def common_to_all(named_sets: Mapping[str, set[str]]) -> set[str]:
    sets = list(named_sets.values())
    return set.intersection(*sets) if sets else set()


def exclusive_to_each(named_sets: Mapping[str, set[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    names = list(named_sets.keys())
    for name in names:
        others = set().union(*(named_sets[n] for n in names if n != name))
        out[name] = set(named_sets[name]) - others
    return out
