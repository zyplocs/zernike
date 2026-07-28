from __future__ import annotations

from math import factorial, sqrt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def ansi_index(n: int, m: int) -> int:
    """
    Convert the double index (n, m) to the ANSI/OSA single index j.

    Examples:
        (0,  0) -> j = 0   piston
        (1, -1) -> j = 1   vertical tilt
        (1,  1) -> j = 2   horizontal tilt
        (2,  0) -> j = 4   defocus
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(
            "A valid Zernike mode requires |m| <= n and n - |m| even."
        )

    return (n * (n + 2) + m) // 2


def ansi_modes(max_radial_order: int = 5) -> list[tuple[int, int, int]]:
    """
    Return modes as (j, n, m) tuples in ANSI/OSA order.

    Through n = 5, this produces 21 modes:
        1 + 2 + 3 + 4 + 5 + 6 = 21.
    """
    modes: list[tuple[int, int, int]] = []

    for n in range(max_radial_order + 1):
        for m in range(-n, n + 1, 2):
            modes.append((ansi_index(n, m), n, m))

    return modes


def radial_zernike(
    n: int,
    m_absolute: int,
    rho: FloatArray,
) -> FloatArray:
    """
    Evaluate the radial Zernike polynomial R_n^|m|(rho).
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if (
        m_absolute < 0
        or m_absolute > n
        or (n - m_absolute) % 2 != 0
    ):
        raise ValueError(
            "Require 0 <= |m| <= n and n - |m| even."
        )

    radial = np.zeros_like(rho, dtype=np.float64)
    largest_k = (n - m_absolute) // 2

    for k in range(largest_k + 1):
        coefficient = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + m_absolute) // 2 - k)
                * factorial((n - m_absolute) // 2 - k)
            )
        )

        radial += coefficient * rho ** (n - 2 * k)

    return radial


def zernike(
    n: int,
    m: int,
    rho: FloatArray,
    theta: FloatArray,
    *,
    normalize: bool = True,
) -> FloatArray:
    """
    Evaluate Z_n^m over a polar coordinate grid.

    Convention:
        m < 0  -> sine angular component
        m = 0  -> rotationally symmetric
        m > 0  -> cosine angular component
    """
    m_absolute = abs(m)
    radial = radial_zernike(n, m_absolute, rho)

    if m < 0:
        angular = np.sin(m_absolute * theta)
    elif m > 0:
        angular = np.cos(m_absolute * theta)
    else:
        angular = np.ones_like(theta)

    if normalize:
        normalization = (
            sqrt(n + 1)
            if m == 0
            else sqrt(2 * (n + 1))
        )
    else:
        normalization = 1.0

    values = normalization * radial * angular

    # Everything outside the normalized pupil becomes transparent.
    return np.where(rho <= 1.0, values, np.nan)


def pupil_grid(
    samples: int = 301,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Construct Cartesian and polar coordinates over [-1, 1] x [-1, 1].
    """
    if samples < 2:
        raise ValueError("samples must be at least 2.")

    axis = np.linspace(-1.0, 1.0, samples, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)

    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    return x, y, rho, theta


def plot_first_21_zernikes(
    samples: int = 301,
) -> Figure:
    """
    Plot ANSI/OSA modes j = 0 through j = 20.

    Modes are arranged by radial order:
        row 0 -> 1 mode
        row 1 -> 2 modes
        ...
        row 5 -> 6 modes
    """
    _, _, rho, theta = pupil_grid(samples)
    modes = ansi_modes(max_radial_order=5)

    evaluated_modes: list[
        tuple[int, int, int, FloatArray]
    ] = []

    for j, n, m in modes:
        values = zernike(n, m, rho, theta)
        evaluated_modes.append((j, n, m, values))

    # A shared scale makes amplitude comparisons meaningful.
    amplitude_limit = max(
        np.nanmax(np.abs(values))
        for _, _, _, values in evaluated_modes
    )

    color_normalization = TwoSlopeNorm(
        vmin=-amplitude_limit,
        vcenter=0.0,
        vmax=amplitude_limit,
    )

    # Eleven columns let us center rows containing 1, 2, ..., 6 modes.
    figure = plt.figure(
        figsize=(13, 9),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(nrows=6, ncols=11)

    axes = []
    image = None

    for j, n, m, values in evaluated_modes:
        position_within_row = (m + n) // 2

        # Produces columns such as:
        # n = 0:             5
        # n = 1:           4, 6
        # n = 2:          3, 5, 7
        # ...
        column = 5 - n + 2 * position_within_row

        axis = figure.add_subplot(grid[n, column])

        image = axis.imshow(
            values,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap="RdBu_r",
            norm=color_normalization,
            interpolation="bilinear",
        )

        axis.add_patch(
            Circle(
                (0.0, 0.0),
                radius=1.0,
                fill=False,
                linewidth=0.5,
            )
        )

        axis.set_title(
            rf"$j={j}$"
            "\n"
            rf"$(n,m)=({n},{m})$",
            fontsize=8,
        )
        axis.set_aspect("equal")
        axis.set_axis_off()
        axes.append(axis)

    if image is None:
        raise RuntimeError("No Zernike modes were plotted.")

    colorbar = figure.colorbar(
        image,
        ax=axes,
        shrink=0.80,
        pad=0.02,
    )
    colorbar.set_label("Normalized Zernike value")

    figure.suptitle(
        "First 21 Zernike polynomials — ANSI/OSA ordering",
        fontsize=16,
    )

    return figure


if __name__ == "__main__":
    atlas = plot_first_21_zernikes(samples=401)

    # Uncomment this to save a high-resolution copy:
    # atlas.savefig(
    #     "first_21_zernike_polynomials.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    plt.show()

def plot_zernike_surface(
    j: int,
    samples: int = 201,
) -> Figure:
    """
    Plot one ANSI/OSA Zernike mode as a 3D wavefront surface.
    """
    modes = ansi_modes(max_radial_order=5)

    if not 0 <= j < len(modes):
        raise ValueError("For the first 21 modes, j must be from 0 to 20.")

    _, n, m = modes[j]
    x, y, rho, theta = pupil_grid(samples)

    values = zernike(n, m, rho, theta)
    masked_values = np.ma.masked_invalid(values)

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(projection="3d")

    surface = axis.plot_surface(
        x,
        y,
        masked_values,
        cmap="RdBu_r",
        linewidth=0,
        antialiased=True,
        rcount=100,
        ccount=100,
    )

    axis.set_title(
        rf"ANSI $j={j}$: $Z_{{{n}}}^{{{m}}}$"
    )
    axis.set_xlabel("Normalized pupil x")
    axis.set_ylabel("Normalized pupil y")
    axis.set_zlabel("Normalized wavefront value")
    axis.set_box_aspect((1.0, 1.0, 0.65))

    figure.colorbar(
        surface,
        ax=axis,
        shrink=0.65,
        pad=0.10,
    )

    return figure
