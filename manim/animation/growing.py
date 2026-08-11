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

from typing import TYPE_CHECKING, Any, cast

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
    """Introduce a :class:`~.TippableVMobject` by growing it from its start toward its tip.

    Parameters
    ----------
    arrow
        The arrow to be introduced.
    point_color
        Initial color of the arrow before growing to its full size. Leave empty to match arrow's color.

    Examples
    --------

    .. manim :: GrowArrowExample

        class GrowArrowExample(Scene):
            def construct(self):
                arrows = [Arrow(2 * LEFT, 2 * RIGHT), Arrow(2 * DR, 2 * UL)]
                VGroup(*arrows).set_x(0).arrange(buff=2)
                self.play(GrowArrow(arrows[0]))
                self.play(GrowArrow(arrows[1], point_color=RED))

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
        self._full_arrow = cast(
            "TipableVMobject",
            self.mobject.copy(),
        )

        self._target_color = ManimColor(self.mobject.get_color())

        self._starting_color = ManimColor(
            self.point_color if self.point_color is not None else self._target_color
        )

        has_end_tip = self._full_arrow.has_tip()
        has_start_tip = self._full_arrow.has_start_tip()

        self._end_tip: ArrowTip | None = None
        self._start_tip: ArrowTip | None = None

        # Store tip identities and the actual intended path endpoints.
        if has_end_tip:
            self._end_tip = cast(
                "ArrowTip",
                self._full_arrow.tip.copy(),
            )
            full_end = self._full_arrow.tip.tip_point.copy()
        else:
            self._end_tip = None
            full_end = self._full_arrow.get_end().copy()

        if has_start_tip:
            self._start_tip = cast(
                "ArrowTip",
                self._full_arrow.start_tip.copy(),
            )
            full_start = self._full_arrow.start_tip.tip_point.copy()
        else:
            self._start_tip = None
            full_start = self._full_arrow.get_start().copy()

        # Remove tips without pop_tips(), so there is no additional
        # modification of the shaft.
        if has_end_tip:
            self._full_arrow.remove(self._full_arrow.tip)

        if has_start_tip:
            self._full_arrow.remove(self._full_arrow.start_tip)

        self._full_arrow.put_start_and_end_on(
            full_start,
            full_end,
        )

        super().begin()

    def _remove_current_tips(self) -> None:
        """Remove current tips without modifying the shaft geometry."""
        if self.mobject.has_tip():
            self.mobject.remove(self.mobject.tip)

        if self.mobject.has_start_tip():
            self.mobject.remove(self.mobject.start_tip)

    def _point_at_alpha(self, alpha: float) -> Point3D | None:
        """Evaluate the pristine path using the same parameterisation
        used by pointwise_become_partial().
        """
        n_curves = self._full_arrow.get_num_curves()

        if n_curves == 0:
            return None

        alpha = np.clip(alpha, 0.0, 1.0)

        if alpha == 1:
            curve_index = n_curves - 1
            local_t = 1.0
        else:
            scaled = alpha * n_curves
            curve_index = int(np.floor(scaled))
            local_t = scaled - curve_index

            curve_index = min(curve_index, n_curves - 1)

        points = self._full_arrow.get_nth_curve_points(curve_index)

        return bezier(points)(local_t)

    def _get_tangent(
        self,
        alpha: float,
        at_start: bool = False,
        eps: float = 1e-4,
        tol: float = 1e-10,
    ) -> Vector3DLike:
        """Estimate the tangent using a short secant in the same
        parameterisation used by pointwise_become_partial().

        This avoids both:
        - the arc-length parameter mismatch of point_from_proportion()
        - unstable/degenerate endpoint handles at Bézier boundaries
        """

        def secant(delta: float) -> Vector3DLike:
            if at_start:
                p0 = self._point_at_alpha(0)
                p1 = self._point_at_alpha(min(1, delta))
            else:
                p0 = self._point_at_alpha(max(0, alpha - delta))
                p1 = self._point_at_alpha(min(1, alpha + delta))

            if p0 is None or p1 is None:
                return np.zeros(3)

            # Start tips point opposite the path direction.
            return p0 - p1 if at_start else p1 - p0

        for scale in (1, 10, 100, 1000):
            tangent = secant(eps * scale)

            if np.linalg.norm(tangent) > tol:
                return tangent

        return np.zeros(3)

    def _position_tip(
        self,
        tip: ArrowTip,
        alpha: float,
        at_start: bool = False,
    ) -> ArrowTip:
        anchor = self.mobject.get_start() if at_start else self.mobject.get_end()

        tangent = self._get_tangent(
            alpha,
            at_start=at_start,
        )

        if np.linalg.norm(tangent) < 1e-10:
            return tip

        tip.rotate(angle_of_vector(tangent) - tip.tip_angle)

        tip.shift(anchor - tip.tip_point)

        return tip

    def _trim_for_tip(
        self,
        tip: ArrowTip,
        at_start: bool = False,
    ) -> None:
        arc_length = self.mobject.get_arc_length()

        if arc_length <= 0:
            return

        # Current intended attachment point:
        # the geometric centre of the tip.
        tip_anchor = tip.get_center()

        direction = tip.tip_point - tip_anchor
        direction_length = np.linalg.norm(direction)

        if direction_length == 0:
            return

        direction /= direction_length

        # Physical distance from tip point back to its anchor,
        # measured along the tip's axis.
        trim_length = np.dot(
            tip.tip_point - tip_anchor,
            direction,
        )

        if trim_length <= 0 or trim_length >= arc_length:
            return

        trim_proportion = trim_length / arc_length

        if at_start:
            self.mobject.pointwise_become_partial(
                self.mobject,
                trim_proportion,
                1,
            )
        else:
            self.mobject.pointwise_become_partial(
                self.mobject,
                0,
                1 - trim_proportion,
            )

    def _attach_tip(
        self,
        tip: ArrowTip,
        alpha: float,
        at_start: bool = False,
    ) -> None:
        # First position against the untrimmed endpoint.
        self._position_tip(
            tip,
            alpha,
            at_start=at_start,
        )

        # Then shorten the shaft underneath the tip.
        self._trim_for_tip(
            tip,
            at_start=at_start,
        )

        self.mobject.assign_tip_attr(
            tip,
            at_start=at_start,
        )

        self.mobject.add(tip)

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(alpha)

        # Do not use pop_tips(); it may modify curved geometry.
        self._remove_current_tips()

        # Reconstruct every frame from the pristine path.
        self.mobject.pointwise_become_partial(
            self._full_arrow,
            0,
            alpha,
        )

        self.mobject._set_stroke_width_from_length()

        length = self.mobject.get_length()

        if length == 0:
            return

        tips = [
            tip
            for tip in (
                self._end_tip,
                self._start_tip,
            )
            if tip is not None
        ]

        if not tips:
            return

        original_tip_width = sum(tip.width for tip in tips)

        if original_tip_width == 0:
            return

        target_tip_width = min(
            original_tip_width,
            self.mobject.max_tip_length_to_length_ratio * length,
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

        if end_tip is not None:
            self._attach_tip(
                end_tip,
                alpha,
                at_start=False,
            )

        if start_tip is not None:
            self._attach_tip(
                start_tip,
                alpha,
                at_start=True,
            )

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
