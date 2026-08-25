r"""Mobjects that are curved.

Examples
--------
.. manim:: UsefulAnnotations
    :save_last_frame:

    class UsefulAnnotations(Scene):
        def construct(self):
            m0 = Dot()
            m1 = AnnotationDot()
            m2 = LabeledDot("ii")
            m3 = LabeledDot(MathTex(r"\alpha").set_color(ORANGE))
            m4 = CurvedArrow(2*LEFT, 2*RIGHT, radius= -5)
            m5 = CurvedArrow(2*LEFT, 2*RIGHT, radius= 8)
            m6 = CurvedDoubleArrow(ORIGIN, 2*RIGHT)

            self.add(m0, m1, m2, m3, m4, m5, m6)
            for i, mobj in enumerate(self.mobjects):
                mobj.shift(DOWN * (i-3))

"""

from __future__ import annotations

from manim.typing import Vector3D

__all__ = [
    "TipableVMobject",
]


from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from manim import config
from manim.constants import *
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.bezier import bezier
from manim.utils.space_ops import (
    angle_of_vector,
    cartesian_to_spherical,
    normalize,
)

if TYPE_CHECKING:
    from manim.mobject.geometry.tips import ArrowTip
    from manim.typing import Callable, Point3D, Point3DLike, Vector3DLike


class TipableVMobject(VMobject, metaclass=ConvertToOpenGL):
    """A VMobject supporting arrow tips at its start and/or end."""

    def __init__(
        self,
        tip_length: float = DEFAULT_ARROW_TIP_LENGTH,
        stroke_width: float = DEFAULT_STROKE_WIDTH,
        normal_vector: Vector3DLike = OUT,
        tip_style: dict | None = None,
        max_tip_length_to_length_ratio: float = 0.25,
        max_stroke_width_to_length_ratio: float = 5,
        **kwargs: Any,
    ) -> None:
        self.tip_length = tip_length
        self.normal_vector = normal_vector
        self.tip_style = tip_style if tip_style is not None else {}

        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        self.initial_stroke_width = stroke_width

        super().__init__(
            stroke_width=stroke_width,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Untipped geometry
    # ------------------------------------------------------------------

    def _store_untipped_points(self) -> None:
        """Store the exact path before it is first trimmed for a tip."""
        if not hasattr(self, "_untipped_points"):
            self._untipped_points = self.points.copy()

    def get_untipped_copy(self) -> Self:
        result = self.copy()

        if result.has_tip():
            result.remove(result.tip)

        if result.has_start_tip():
            result.remove(result.start_tip)

        if hasattr(result, "_untipped_points"):
            result.set_points(result._untipped_points.copy())

            result.put_start_and_end_on(
                self.get_start(),
                self.get_end(),
            )

        return result

    def shift(self, *vectors: Vector3DLike) -> Self:
        """Shift the mobject and its stored untipped geometry."""
        result = super().shift(*vectors)

        if hasattr(self, "_untipped_points"):
            total_vector = sum(
                (np.asarray(vector) for vector in vectors),
                start=np.zeros(3),
            )
            self._untipped_points += total_vector

        return result

    def _transform_untipped_points(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        about_point: Point3DLike,
    ) -> None:
        if not hasattr(self, "_untipped_points"):
            return

        about_point = np.asarray(about_point)

        points = self._untipped_points.copy()
        points -= about_point
        points = func(points)
        points += about_point

        self._untipped_points = points

    def apply_points_function_about_point(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Apply a transformation to both current and untipped geometry."""
        # OpenGL uses apply_points_function instead.
        if config.renderer == RendererType.OPENGL:
            return self.apply_points_function(
                func,
                about_point=about_point,
                about_edge=about_edge,
            )

        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_critical_point(about_edge)

        about_point = np.array(about_point, copy=True)

        result = super().apply_points_function_about_point(
            func,
            about_point,
            about_edge,
        )

        self._transform_untipped_points(
            func,
            about_point,
        )

        return result

    def apply_points_function(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        works_on_bounding_box: bool = False,
    ) -> Self:
        """OpenGL counterpart of apply_points_function_about_point."""
        if config.renderer != RendererType.OPENGL:
            return self.apply_points_function_about_point(
                func,
                about_point,
                about_edge,
            )

        if about_point is None:
            if about_edge is None:
                about_point = ORIGIN
            else:
                about_point = self.get_bounding_box_point(about_edge)

        about_point = np.array(about_point, copy=True)

        result = super().apply_points_function(
            func,
            about_point=about_point,
            about_edge=about_edge,
            works_on_bounding_box=works_on_bounding_box,
        )

        self._transform_untipped_points(
            func,
            about_point,
        )

        return result

    # ------------------------------------------------------------------
    # Adding / creating tips
    # ------------------------------------------------------------------

    def add_tip(
        self,
        tip: ArrowTip | None = None,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
        at_start: bool = False,
    ) -> Self:
        """Add a tip while preserving the original underlying path."""
        self._store_untipped_points()

        if tip is None:
            tip = self.create_tip(
                tip_shape,
                tip_length,
                tip_width,
                at_start,
            )
        else:
            self.position_tip(tip, at_start)

        self.reset_endpoints_based_on_tip(
            tip,
            at_start,
        )

        self.assign_tip_attr(
            tip,
            at_start,
        )

        self.add(tip)

        return self

    def create_tip(
        self,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
        at_start: bool = False,
    ) -> ArrowTip:
        tip = self.get_unpositioned_tip(
            tip_shape,
            tip_length,
            tip_width,
        )

        self.position_tip(
            tip,
            at_start,
        )

        return tip

    def get_unpositioned_tip(
        self,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
    ) -> ArrowTip:
        from manim.mobject.geometry.tips import ArrowTriangleFilledTip

        style: dict[str, Any] = {}

        if tip_shape is None:
            tip_shape = ArrowTriangleFilledTip

        if tip_shape is ArrowTriangleFilledTip:
            if tip_width is None:
                tip_width = self.get_default_tip_length()

            style["width"] = tip_width

        if tip_length is None:
            tip_length = self.get_default_tip_length()

        color = self.get_color()

        style.update(
            {
                "fill_color": color,
                "stroke_color": color,
            }
        )
        style.update(self.tip_style)

        return tip_shape(
            length=tip_length,
            **style,
        )

    def position_tip(
        self,
        tip: ArrowTip,
        at_start: bool = False,
    ) -> ArrowTip:
        """Position a tip according to the local endpoint tangent."""
        if at_start:
            anchor = self.get_start()
            handle = self.get_first_handle()
        else:
            handle = self.get_last_handle()
            anchor = self.get_end()

        angles = cartesian_to_spherical(handle - anchor)

        tip.rotate(
            angles[1] - PI - tip.tip_angle,
        )

        if not hasattr(self, "_init_positioning_axis"):
            axis = np.array(
                [
                    np.sin(angles[1]),
                    -np.cos(angles[1]),
                    0,
                ]
            )

            tip.rotate(
                -angles[2] + PI / 2,
                axis=axis,
            )

            self._init_positioning_axis = axis

        tip.shift(anchor - tip.tip_point)

        return tip

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def reset_endpoints_based_on_tip(
        self,
        tip: ArrowTip,
        at_start: bool,
    ) -> Self:
        return self._trim_for_tip(
            tip,
            at_start,
        )

    def _point_at_parameter(
        self,
        path: TipableVMobject,
        alpha: float,
    ) -> Point3D | None:
        """Evaluate using pointwise_become_partial's parameterisation."""
        n_curves = path.get_num_curves()

        if n_curves == 0:
            return None

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        if alpha >= 1:
            curve_index = n_curves - 1
            local_t = 1.0
        else:
            scaled = alpha * n_curves

            curve_index = min(
                int(np.floor(scaled)),
                n_curves - 1,
            )

            local_t = scaled - curve_index

        points = path.get_nth_curve_points(curve_index)

        return bezier(points)(local_t)

    def _bisect(
        self,
        func: Callable[[float], float],
        low: float,
        high: float,
        iterations: int = 30,
    ) -> float:
        low_value = func(low)

        for _ in range(iterations):
            mid = (low + high) / 2
            mid_value = func(mid)

            if np.signbit(mid_value) == np.signbit(low_value):
                low = mid
                low_value = mid_value
            else:
                high = mid

        return (low + high) / 2

    def _parameter_before_end_by_chord(
        self,
        path: TipableVMobject,
        distance: float,
        *,
        samples: int = 64,
        iterations: int = 30,
    ) -> float:
        if distance <= 0:
            return 1.0

        endpoint = path.get_end()

        def chord_error(
            alpha: float,
        ) -> float:
            point = self._point_at_parameter(
                path,
                alpha,
            )

            if point is None:
                return -distance

            return np.linalg.norm(endpoint - point) - distance

        previous_alpha = 1.0

        for i in range(
            1,
            samples + 1,
        ):
            alpha = 1.0 - i / samples

            if chord_error(alpha) >= 0:
                return self._bisect(
                    chord_error,
                    alpha,
                    previous_alpha,
                    iterations,
                )

            previous_alpha = alpha

        return 0.0

    def _parameter_after_start_by_chord(
        self,
        path: TipableVMobject,
        distance: float,
        *,
        samples: int = 64,
        iterations: int = 30,
    ) -> float:
        if distance <= 0:
            return 0.0

        endpoint = path.get_start()

        def chord_error(
            alpha: float,
        ) -> float:
            point = self._point_at_parameter(
                path,
                alpha,
            )

            if point is None:
                return -distance

            return np.linalg.norm(point - endpoint) - distance

        previous_alpha = 0.0

        for i in range(
            1,
            samples + 1,
        ):
            alpha = i / samples

            if chord_error(alpha) >= 0:
                return self._bisect(
                    chord_error,
                    previous_alpha,
                    alpha,
                    iterations,
                )

            previous_alpha = alpha

        return 1.0

    def _position_tip_between(
        self,
        tip: ArrowTip,
        base_point: Point3DLike,
        tip_point: Point3DLike,
    ) -> None:
        """Rigidly place a tip between two prescribed points."""
        current_axis = tip.tip_point - tip.base
        target_axis = np.asarray(tip_point) - np.asarray(base_point)

        current_length = np.linalg.norm(current_axis)
        target_length = np.linalg.norm(target_axis)

        if np.isclose(current_length, 0) or np.isclose(target_length, 0):
            return

        angle = angle_of_vector(target_axis) - angle_of_vector(current_axis)

        tip.rotate(
            angle,
            about_point=tip.base,
        )

        tip.shift(np.asarray(base_point) - tip.base)

    def _trim_for_tip(
        self,
        tip: ArrowTip,
        at_start: bool = False,
    ) -> Self:
        """Trim the path without geometrically deforming it."""
        if self.get_num_curves() == 0:
            return self

        source = self.copy()

        tip_length = np.linalg.norm(tip.tip_point - tip.base)

        if at_start:
            alpha = self._parameter_after_start_by_chord(
                source,
                tip_length,
            )

            base = self._point_at_parameter(
                source,
                alpha,
            )

            if base is None:
                return self

            tip_point = source.get_start().copy()

            self._position_tip_between(
                tip,
                base,
                tip_point,
            )

            self.pointwise_become_partial(
                source,
                alpha,
                1.0,
            )

        else:
            alpha = self._parameter_before_end_by_chord(
                source,
                tip_length,
            )

            base = self._point_at_parameter(
                source,
                alpha,
            )

            if base is None:
                return self

            tip_point = source.get_end().copy()

            self._position_tip_between(
                tip,
                base,
                tip_point,
            )

            self.pointwise_become_partial(
                source,
                0.0,
                alpha,
            )

        return self

    # ------------------------------------------------------------------
    # Tip bookkeeping
    # ------------------------------------------------------------------

    def assign_tip_attr(
        self,
        tip: ArrowTip,
        at_start: bool,
    ) -> Self:
        if at_start:
            self.start_tip = tip
        else:
            self.tip = tip

        return self

    def has_tip(self) -> bool:
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self) -> bool:
        return hasattr(self, "start_tip") and self.start_tip in self

    def pop_tips(self) -> VGroup:
        """Remove tips and restore the exact untrimmed path."""
        result = self.get_group_class()()

        if self.has_tip():
            result.add(self.tip)
            self.remove(self.tip)

        if self.has_start_tip():
            result.add(self.start_tip)
            self.remove(self.start_tip)

        if result.submobjects and hasattr(self, "_untipped_points"):
            self.set_points(self._untipped_points.copy())

        return result

    def get_tips(self) -> VGroup:
        result = self.get_group_class()()

        if self.has_tip():
            result.add(self.tip)

        if self.has_start_tip():
            result.add(self.start_tip)

        return result

    def get_tip(self) -> ArrowTip:
        tips = self.get_tips()

        if len(tips) == 0:
            raise Exception("tip not found")

        return tips[0]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_first_handle(self) -> Point3D:
        return self.points[1]

    def get_last_handle(self) -> Point3D:
        return self.points[-2]

    def get_end(self) -> Point3D:
        if self.has_tip():
            return self.tip.get_start()

        return super().get_end()

    def get_start(self) -> Point3D:
        if self.has_start_tip():
            return self.start_tip.get_start()

        return super().get_start()

    def get_length(self) -> float:
        start, end = self.get_start_and_end()

        return float(np.linalg.norm(start - end))

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale(
        self,
        factor: float,
        scale_tips: bool = True,
        **kwargs: Any,
    ) -> Self:
        if not self.has_tip() and not self.has_start_tip():
            return super().scale(
                factor,
                **kwargs,
            )

        if np.isclose(
            self.get_arc_length(),
            0,
        ):
            return self

        if scale_tips:
            super().scale(
                factor,
                **kwargs,
            )

            self.initial_stroke_width *= factor
            self._set_stroke_width_from_length()

            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()

        old_tips = self.pop_tips()

        super().scale(
            factor,
            **kwargs,
        )

        # At this point the full, untrimmed path is present.
        self._set_stroke_width_from_length()

        if has_tip:
            self.add_tip(
                tip=cast(
                    "ArrowTip",
                    old_tips[0],
                )
            )

        if has_start_tip:
            index = 1 if has_tip else 0

            self.add_tip(
                tip=cast(
                    "ArrowTip",
                    old_tips[index],
                ),
                at_start=True,
            )

        return self

    # ------------------------------------------------------------------
    # Other arrow helpers
    # ------------------------------------------------------------------

    def get_normal_vector(self) -> Vector3D:
        p0, p1, p2 = self.tip.get_start_anchors()[:3]

        return normalize(
            np.cross(
                p2 - p1,
                p1 - p0,
            )
        )

    def reset_normal_vector(self) -> Self:
        self.normal_vector = self.get_normal_vector()

        return self

    def get_default_tip_length(
        self,
    ) -> float:
        max_ratio = self.max_tip_length_to_length_ratio

        return min(
            self.tip_length,
            max_ratio * self.get_length(),
        )

    def _set_stroke_width_from_length(
        self,
    ) -> Self:
        max_ratio = self.max_stroke_width_to_length_ratio

        width = min(
            self.initial_stroke_width,
            max_ratio * self.get_arc_length(),
        )

        if config.renderer == RendererType.OPENGL:
            self.set_stroke(
                width=width,
                recurse=False,
            )
        else:
            self.set_stroke(
                width=width,
                family=False,
            )

        return self
