"""Animations that introduce mobjects to scene by growing them from points.

.. manim:: Growing

    class Growing(Scene):
        def construct(self):
            square = Square()
            circle = Circle()
            triangle = Triangle()
            arrow = Arrow(LEFT, RIGHT)
            star = Star()

            VGroup(square, circle, triangle).set_x(0).arrange(buff=1.5).set_y(2)
            VGroup(arrow, star).move_to(DOWN).set_x(0).arrange(buff=1.5).set_y(-2)

            self.play(GrowFromPoint(square, ORIGIN))
            self.play(GrowFromCenter(circle))
            self.play(GrowFromEdge(triangle, DOWN))
            self.play(GrowArrow(arrow))
            self.play(SpinInFromNothing(star))

"""

from __future__ import annotations

__all__ = [
    "GrowFromPoint",
    "GrowFromCenter",
    "GrowFromEdge",
    "GrowArrow",
    "SpinInFromNothing",
]

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from manim.utils.bezier import bezier
from manim.utils.color import ManimColor, interpolate_color
from manim.utils.space_ops import angle_of_vector

from ..animation.transform import Animation, Transform
from ..constants import PI
from ..utils.paths import spiral_path

if TYPE_CHECKING:
    from manim.mobject.geometry.tipable import TipableVMobject
    from manim.mobject.geometry.tips import ArrowTip
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject
    from manim.typing import Point3D, Point3DLike, Vector3DLike
    from manim.utils.color import ManimColor, ParsableManimColor

    from ..mobject.mobject import Mobject


class GrowFromPoint(Transform):
    """Introduce an :class:`~.Mobject` by growing it from a point.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    point
        The point from which the mobject grows.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromPointExample

        class GrowFromPointExample(Scene):
            def construct(self):
                dot = Dot(3 * UR, color=GREEN)
                squares = [Square() for _ in range(4)]
                VGroup(*squares).set_x(0).arrange(buff=1)
                self.add(dot)
                self.play(GrowFromPoint(squares[0], ORIGIN))
                self.play(GrowFromPoint(squares[1], [-2, 2, 0]))
                self.play(GrowFromPoint(squares[2], [3, -2, 0], RED))
                self.play(GrowFromPoint(squares[3], dot, dot.get_color()))

    """

    def __init__(
        self,
        mobject: Mobject,
        point: Point3DLike,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ):
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, introducer=True, **kwargs)

    def create_target(self) -> Mobject | OpenGLMobject:
        return self.mobject

    def create_starting_mobject(self) -> Mobject | OpenGLMobject:
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color:
            start.set_color(self.point_color)
        return start


class GrowFromCenter(GrowFromPoint):
    """Introduce an :class:`~.Mobject` by growing it from its center.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromCenterExample

        class GrowFromCenterExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(2)]
                VGroup(*squares).set_x(0).arrange(buff=2)
                self.play(GrowFromCenter(squares[0]))
                self.play(GrowFromCenter(squares[1], point_color=RED))

    """

    def __init__(
        self,
        mobject: Mobject,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ):
        point = mobject.get_center()
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowFromEdge(GrowFromPoint):
    """Introduce an :class:`~.Mobject` by growing it from one of its bounding box edges.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    edge
        The direction to seek bounding box edge of mobject.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromEdgeExample

        class GrowFromEdgeExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(4)]
                VGroup(*squares).set_x(0).arrange(buff=1)
                self.play(GrowFromEdge(squares[0], DOWN))
                self.play(GrowFromEdge(squares[1], RIGHT))
                self.play(GrowFromEdge(squares[2], UR))
                self.play(GrowFromEdge(squares[3], UP, point_color=RED))


    """

    def __init__(
        self,
        mobject: Mobject,
        edge: Vector3DLike,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ):
        point = mobject.get_critical_point(edge)
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowArrow(Animation):
    """Introduce a :class:`~.TipableVMobject` by growing it from its start toward its tip.

    Parameters
    ----------
    mobject
        The tipable mobject to be introduced.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowArrowExample

        class GrowArrowExample(Scene):
            def construct(self):
                arrows = [
                    Arrow(2 * LEFT, 2 * RIGHT, buff=0),
                    DoubleArrow(2 * LEFT, 2 * RIGHT, buff=0),
                    CurvedArrow(2 * LEFT, 2 * RIGHT),
                    CurvedDoubleArrow(2 * LEFT, 2 * RIGHT),
                ]
                VGroup(*arrows).arrange(DOWN, buff=1)

                self.play(GrowArrow(arrows[0]))
                self.play(GrowArrow(arrows[1]))
                self.play(GrowArrow(arrows[2]))
                self.play(GrowArrow(arrows[3], point_color=RED))

                
    """

    def __init__(
        self,
        mobject: TipableVMobject,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ) -> None:
        self.point_color = point_color
        self.mobject: TipableVMobject
        super().__init__(mobject, **kwargs)

    def begin(self) -> None:
        self._target_color = ManimColor(self.mobject.get_color())

        self._starting_color = ManimColor(
            self.point_color if self.point_color is not None else self._target_color
        )

        self._end_tip = self.mobject.tip.copy() if self.mobject.has_tip() else None
        self._start_tip = (
            self.mobject.start_tip.copy() if self.mobject.has_start_tip() else None
        )

        self._full_path = self.mobject.copy()
        self._full_path.pop_tips()

        super().begin()

    def _remove_current_tips(self) -> None:
        """Remove current tips without altering the shaft."""
        if self.mobject.has_tip():
            self.mobject.remove(self.mobject.tip)

        if self.mobject.has_start_tip():
            self.mobject.remove(self.mobject.start_tip)

    def _point_at_parameter(
        self,
        path: TipableVMobject,
        alpha: float,
    ) -> Point3D | None:
        """Evaluate a VMobject using pointwise_become_partial's parameterisation."""
        n_curves = path.get_num_curves()

        if n_curves == 0:
            return None

        alpha = np.clip(alpha, 0.0, 1.0)

        if alpha >= 1.0:
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
        """Find the closest point before the end at chord distance ``distance``."""
        if distance <= 0:
            return 1.0

        endpoint = path.get_end()

        def chord_error(alpha: float) -> float:
            point = self._point_at_parameter(path, alpha)
            return np.linalg.norm(endpoint - point) - distance

        previous_alpha = 1.0

        for i in range(1, samples + 1):
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
        """Find the closest point after the start at chord distance ``distance``."""
        if distance <= 0:
            return 0.0

        endpoint = path.get_start()

        def chord_error(alpha: float) -> float:
            point = self._point_at_parameter(path, alpha)
            return np.linalg.norm(point - endpoint) - distance

        previous_alpha = 0.0

        for i in range(1, samples + 1):
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

    def _make_scaled_tips(
        self,
        path_length: float,
    ) -> tuple[ArrowTip | None, ArrowTip | None]:
        """Create appropriately scaled copies of the original tips."""
        original_tips = [
            tip
            for tip in (
                self._end_tip,
                self._start_tip,
            )
            if tip is not None
        ]

        if not original_tips:
            return None, None

        original_tip_width = sum(tip.width for tip in original_tips)

        if np.isclose(original_tip_width, 0):
            return None, None

        target_tip_width = min(
            original_tip_width,
            self.mobject.max_tip_length_to_length_ratio * path_length,
        )

        tip_factor = target_tip_width / original_tip_width

        end_tip = (
            self._end_tip.copy().scale(tip_factor)
            if self._end_tip is not None
            else None
        )
        start_tip = (
            self._start_tip.copy().scale(tip_factor)
            if self._start_tip is not None
            else None
        )

        return end_tip, start_tip

    def _position_tip_between(
        self,
        tip: ArrowTip,
        base_point: Point3DLike,
        tip_point: Point3DLike,
    ) -> None:
        """Rigidly position a tip between two prescribed points."""

        current_axis = tip.tip_point - tip.base
        target_axis = tip_point - base_point
        current_length = np.linalg.norm(current_axis)
        target_length = np.linalg.norm(target_axis)

        if np.isclose(current_length, 0) or np.isclose(target_length, 0):
            return

        angle = angle_of_vector(target_axis) - angle_of_vector(current_axis)

        tip.rotate(angle, about_point=tip.base)
        tip.shift(base_point - tip.base)

    def interpolate_mobject(
        self,
        alpha: float,
    ) -> None:
        alpha = float(
            np.clip(
                self.rate_func(alpha),
                0.0,
                1.0,
            )
        )

        self._remove_current_tips()

        current_path = self._full_path.copy()

        current_path.pointwise_become_partial(
            self._full_path,
            0.0,
            alpha,
        )

        if current_path.get_num_curves() == 0:
            self.mobject.set_points(current_path.points.copy())
            return

        path_length = current_path.get_arc_length()

        if path_length <= 1e-12:
            self.mobject.set_points(current_path.points.copy())
            return

        end_tip, start_tip = self._make_scaled_tips(path_length)

        shaft_start = 0.0
        shaft_end = 1.0

        end_tip_point = None
        end_base_point = None

        start_tip_point = None
        start_base_point = None

        if end_tip is not None:
            tip_length = float(np.linalg.norm(end_tip.tip_point - end_tip.base))
            
            shaft_end = self._parameter_before_end_by_chord(
                current_path,
                tip_length,
            )

            end_tip_point = current_path.get_end().copy()

            end_base_point = self._point_at_parameter(
                current_path,
                shaft_end,
            )
            
            self._position_tip_between(
                end_tip,
                end_base_point,
                end_tip_point,
            )

            self.mobject.assign_tip_attr(
                end_tip,
                at_start=False,
            )
            self.mobject.add(end_tip)

        if start_tip is not None:
            tip_length = float(np.linalg.norm(start_tip.tip_point - start_tip.base))
            
            shaft_start = self._parameter_after_start_by_chord(
                current_path,
                tip_length,
            )

            start_tip_point = current_path.get_start().copy()

            start_base_point = self._point_at_parameter(
                current_path,
                shaft_start,
            )
            
            self._position_tip_between(
                start_tip,
                start_base_point,
                start_tip_point,
            )

            self.mobject.assign_tip_attr(
                start_tip,
                at_start=True,
            )
            self.mobject.add(start_tip)

        self.mobject.pointwise_become_partial(
                    current_path,
                    shaft_start,
                    shaft_end,
                )
        
        self.mobject._set_stroke_width_from_length()

        color = interpolate_color(
            self._starting_color,
            self._target_color,
            alpha,
        )

        self.mobject.set_color(color)


class SpinInFromNothing(GrowFromCenter):
    """Introduce an :class:`~.Mobject` spinning and growing it from its center.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    angle
        The amount of spinning before mobject reaches its full size. E.g. 2*PI means
        that the object will do one full spin before being fully introduced.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: SpinInFromNothingExample

        class SpinInFromNothingExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(3)]
                VGroup(*squares).set_x(0).arrange(buff=2)
                self.play(SpinInFromNothing(squares[0]))
                self.play(SpinInFromNothing(squares[1], angle=2 * PI))
                self.play(SpinInFromNothing(squares[2], point_color=RED))

    """

    def __init__(
        self,
        mobject: Mobject,
        angle: float = PI / 2,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ):
        self.angle = angle
        super().__init__(
            mobject, path_func=spiral_path(angle), point_color=point_color, **kwargs
        )
