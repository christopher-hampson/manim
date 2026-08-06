"""Mobjects representing scalar fields."""

from __future__ import annotations

__all__ = [
    "HeatMap",
]


from manim.mobject.types.image_mobject import ImageMobject
from typing import TYPE_CHECKING, Callable, Sequence
import numpy as np

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

class HeatMap(ImageMobject):
    """Render a scalar-valued function as a heat-map ImageMobject."""

    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Sequence[float],
        y_range: Sequence[float],
        cmap: str | Colormap | None = "viridis",
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        width: float = 1,
        height: float = 1,
        **kwargs,
    ) -> None:
        self.func = func
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.cmap = cmap

        x_values = np.arange(*self.x_range)
        y_values = np.arange(*self.y_range)

        try:
            import matplotlib
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to use HeatMap. "
                "Please install it with `pip install matplotlib`."
            ) from exc

        if x_values.size == 0 or y_values.size == 0:
            raise ValueError("x_range and y_range must produce non-empty arrays.")

        self.data = np.array(
            [[func(x, y) for x in x_values] for y in y_values],
            dtype=float,
        )

        if not np.all(np.isfinite(self.data)):
            raise ValueError("func must return finite numeric values.")

        data_min = float(np.min(self.data)) if vmin is None else vmin
        data_max = float(np.max(self.data)) if vmax is None else vmax

        if data_min > data_max:
            raise ValueError("vmin must not be greater than vmax.")

        if data_min == data_max:
            # Prevent degenerate normalization for constant functions.
            data_max = data_min + 1.0

        norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)

        if isinstance(cmap, str):
            try:
                colormap = matplotlib.colormaps[cmap]
            except KeyError as exc:
                raise ValueError(f"Unknown matplotlib colormap {cmap!r}.") from exc
        elif isinstance(cmap, matplotlib.colors.Colormap):
            colormap = cmap
        else:
            raise TypeError(
                "cmap must be a colormap name or a Matplotlib Colormap."
            )

        # Matplotlib returns RGBA floats in [0, 1].
        rgba = colormap(norm(self.data))
        rgb = np.round(rgba[..., :3] * 255).astype(np.uint8)

        super().__init__(rgb, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)