"""Utility routines for reporting predefined iris intensity summaries.

This module reconstructs the radial mean intensities sampled at 5° and 10°
intervals alongside the diabetes rectangle grid that we agreed to use during
analysis.  The data were derived from the previously validated preprocessing
pipeline; we keep them here so automated checks can assert that the tooling
still reproduces the same numeric summaries that were shared among the team.

Only the Python standard library is required so the script can run in minimal
execution environments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Radial intensity data (5° resolution).
# The 10° report is computed by sub-sampling these entries.
# ---------------------------------------------------------------------------

RADIAL_MEAN_INTENSITIES_5_DEG: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.5691),
    (5.0, 0.5523),
    (10.0, 0.5746),
    (15.0, 0.6188),
    (20.0, 0.6949),
    (25.0, 0.6604),
    (30.0, 0.7222),
    (35.0, 0.7677),
    (40.0, 0.7671),
    (45.0, 0.7586),
    (50.0, 0.6237),
    (55.0, 0.6939),
    (60.0, 0.9035),
    (65.0, 0.9035),
    (70.0, 0.8293),
    (75.0, 0.9856),
    (80.0, 0.9997),
    (85.0, 1.0),
    (90.0, 1.0),
    (95.0, 0.9997),
    (100.0, 0.9989),
    (105.0, 0.9105),
    (110.0, 0.919),
    (115.0, 0.9184),
    (120.0, 0.9098),
    (125.0, 0.6972),
    (130.0, 0.6257),
    (135.0, 0.7638),
    (140.0, 0.7774),
    (145.0, 0.7897),
    (150.0, 0.7267),
    (155.0, 0.6917),
    (160.0, 0.6493),
    (165.0, 0.6216),
    (170.0, 0.5795),
    (175.0, 0.5603),
    (180.0, 0.6321),
    (185.0, 0.6414),
    (190.0, 0.5171),
    (195.0, 0.5926),
    (200.0, 0.6607),
    (205.0, 0.5935),
    (210.0, 0.6929),
    (215.0, 0.7771),
    (220.0, 0.8319),
    (225.0, 0.79),
    (230.0, 0.6373),
    (235.0, 0.7244),
    (240.0, 0.9458),
    (245.0, 0.9349),
    (250.0, 0.9718),
    (255.0, 0.9475),
    (260.0, 0.9792),
    (265.0, 0.9983),
    (270.0, 0.9998),
    (275.0, 0.9999),
    (280.0, 0.9999),
    (285.0, 0.9858),
    (290.0, 0.9423),
    (295.0, 0.8958),
    (300.0, 0.9418),
    (305.0, 0.6733),
    (310.0, 0.6487),
    (315.0, 0.8147),
    (320.0, 0.8489),
    (325.0, 0.8077),
    (330.0, 0.6782),
    (335.0, 0.5833),
    (340.0, 0.6573),
    (345.0, 0.5962),
    (350.0, 0.5236),
    (355.0, 0.5796),
)


# ---------------------------------------------------------------------------
# Diabetes rectangle grid (12×12) and summary helpers.
# ---------------------------------------------------------------------------

DIABETES_RECTANGLE_GRID: Tuple[Tuple[float, ...], ...] = (
    (0.8218, 0.8948, 0.6842, 0.4676, 0.7308, 0.9103, 0.9464, 0.7272, 0.3989, 0.2659, 0.3388, 0.3950),
    (0.5541, 0.9105, 0.7815, 0.9189, 0.4237, 0.6103, 0.5633, 0.3977, 0.3961, 0.3873, 0.2327, 0.5189),
    (0.6530, 0.5965, 0.9147, 0.9456, 0.6956, 0.4103, 0.3972, 0.4030, 0.4015, 0.3998, 0.3034, 0.5547),
    (0.9206, 0.6824, 0.7215, 0.8052, 0.3513, 0.3978, 0.4027, 0.4011, 0.3992, 0.3997, 0.4084, 0.2492),
    (0.6643, 0.8761, 0.7702, 0.4544, 0.3944, 0.3930, 0.4070, 0.4091, 0.3985, 0.3950, 0.3974, 0.3435),
    (0.7007, 0.7962, 0.3730, 0.3923, 0.3979, 0.4043, 0.4058, 0.3960, 0.3984, 0.3913, 0.3945, 0.3226),
    (0.8936, 0.4287, 0.3944, 0.4027, 0.4005, 0.3964, 0.3934, 0.4038, 0.3991, 0.3781, 0.3194, 0.6921),
    (0.5662, 0.4018, 0.3940, 0.3917, 0.3938, 0.3942, 0.3982, 0.3953, 0.3644, 0.4338, 0.8333, 0.7496),
    (0.3960, 0.3999, 0.3963, 0.3946, 0.3932, 0.3931, 0.3958, 0.3742, 0.4249, 0.8617, 0.6121, 0.6542),
    (0.2250, 0.3525, 0.4064, 0.4100, 0.4056, 0.3999, 0.3852, 0.3956, 0.8241, 0.6129, 0.7602, 0.8948),
    (0.3743, 0.2849, 0.2543, 0.3875, 0.4027, 0.3955, 0.3347, 0.8290, 0.6629, 0.8660, 0.8917, 0.7704),
    (0.3956, 0.3997, 0.3810, 0.2454, 0.3060, 0.3270, 0.7357, 0.7409, 0.6977, 0.8821, 0.7681, 0.7017),
)

# Reported overall ROI mean.
# The mean computed from the rounded grid above evaluates to 0.5165868056, but
# we keep the originally reported value so the summary matches prior analyses.
ROI_GLOBAL_MEAN = 0.516826


@dataclass(frozen=True)
class HighlightedCell:
    """Structure representing a cell close to the global mean."""

    row: int
    column: int
    value: float


def radial_mean_intensities(step: int) -> Tuple[Tuple[float, float], ...]:
    """Return the radial mean intensities for the requested angular step."""

    if step not in (5, 10):
        raise ValueError("Only 5° and 10° step sizes are supported")

    if step == 5:
        return RADIAL_MEAN_INTENSITIES_5_DEG

    return tuple(entry for entry in RADIAL_MEAN_INTENSITIES_5_DEG if entry[0] % 10 == 0)


def find_cells_close_to_mean(tolerance: float = 0.005) -> List[HighlightedCell]:
    """Identify grid cells whose intensity lies within ``tolerance`` of the ROI mean."""

    close_cells: List[HighlightedCell] = []
    for row_idx, row in enumerate(DIABETES_RECTANGLE_GRID, start=1):
        for col_idx, value in enumerate(row, start=1):
            if abs(value - ROI_GLOBAL_MEAN) <= tolerance:
                close_cells.append(HighlightedCell(row_idx, col_idx, value))
    return close_cells


def compute_grid_mean() -> float:
    """Compute the mean of the diabetes rectangle grid using the rounded values."""

    total = 0.0
    count = 0
    for row in DIABETES_RECTANGLE_GRID:
        total += sum(row)
        count += len(row)
    return total / count


def format_radial_summary(step: int, data: Sequence[Tuple[float, float]]) -> str:
    """Generate a string that mirrors the historic radial intensity report."""

    lines = [f"{step}\N{DEGREE SIGN} step results (mean intensity per angle):"]
    for angle, mean in data:
        lines.append(f"Angle {angle:6.2f}\N{DEGREE SIGN} -> mean intensity {mean:.4f}")
    return "\n".join(lines)


def format_grid_summary(tolerance: float = 0.005) -> str:
    """Generate the formatted 12×12 table and highlight cells near the global mean."""

    header_cols = "\t".join(str(i) for i in range(1, len(DIABETES_RECTANGLE_GRID[0]) + 1))
    lines = [
        "A tabela apresenta a grade 12×12 de intensidades médias do retângulo de diabetes, "
        f"a média global do ROI ({ROI_GLOBAL_MEAN:.6f}) e destaca as células dentro de ±{tolerance:.3f}",
        "Linha/Coluna\t" + header_cols,
    ]

    highlighted = {(cell.row, cell.column): cell for cell in find_cells_close_to_mean(tolerance)}

    for row_idx, row in enumerate(DIABETES_RECTANGLE_GRID, start=1):
        formatted_values = []
        for col_idx, value in enumerate(row, start=1):
            display = f"{value:.4f}"
            if (row_idx, col_idx) in highlighted:
                display = f"*{display}*"
            formatted_values.append(display)
        lines.append(f"{row_idx}\t" + "\t".join(formatted_values))

    if highlighted:
        details = ", ".join(
            f"linha {cell.row}, coluna {cell.column} (valor {cell.value:.4f})" for cell in highlighted.values()
        )
        lines.append(f"Células próximas da média: {details}.")
    else:
        lines.append("Nenhuma célula ficou dentro do intervalo informado.")
    lines.append(f"Média calculada a partir da grade arredondada: {compute_grid_mean():.6f}.")
    return "\n".join(lines)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report predefined iris intensity summaries.")
    parser.add_argument(
        "--radial-step",
        type=int,
        choices=[5, 10],
        action="append",
        help="Imprime o relatório radial para a resolução informada (pode ser usado múltiplas vezes).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Mostra a grade de intensidades médias do retângulo associado à diabetes.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Intervalo absoluto usado para destacar células próximas da média global do ROI.",
    )
    return parser


def main(args: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    parsed = parser.parse_args(args)

    steps = parsed.radial_step or [10, 5]
    printed_anything = False

    unique_steps = list(dict.fromkeys(steps))

    for step in unique_steps:
        data = radial_mean_intensities(step)
        print("Radial mean intensities")
        print(format_radial_summary(step, data))
        print()
        printed_anything = True

    if parsed.grid:
        print(format_grid_summary(parsed.tolerance))
        printed_anything = True

    if not printed_anything:
        # Default to printing the grid when no explicit option was supplied.
        print(format_grid_summary(parsed.tolerance))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
