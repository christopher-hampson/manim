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

from typing import TYPE_CHECKING, Any

import numpy as np

from manim.utils.color import ManimColor, interpolate_color

from ..animation.transform import Animation, Transform
from ..constants import PI
from ..utils.paths import spiral_path

if TYPE_CHECKING:
    from manim.mobject.geometry.tipable import TipableVMobject
    from manim.mobject.geometry.tips import ArrowTip
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject
    from manim.typing import Point3DLike, Vector3DLike
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
    """Introduce a TipableVMobject by growing it along its path."""

    def __init__(
        self,
        mobject: TipableVMobject,
        point_color: ParsableManimColor | None = None,
        **kwargs: Any,
    ) -> None:
        self.point_color = point_color
        self.mobject: TipableVMobject

        super().__init__(
            mobject,
            introducer=True,
            **kwargs,
        )

    def begin(self) -> None:
        self._target_color = ManimColor(self.mobject.get_color())

        self._starting_color = ManimColor(
            self.point_color if self.point_color is not None else self._target_color
        )

        self._end_tip = self.mobject.tip.copy() if self.mobject.has_tip() else None

        self._start_tip = (
            self.mobject.start_tip.copy() if self.mobject.has_start_tip() else None
        )

        self._full_path = self.mobject.get_untipped_copy()

        super().begin()

    def _remove_current_tips(
        self,
    ) -> None:
        """Remove current tips without restoring/changing the shaft."""
        if self.mobject.has_tip():
            self.mobject.remove(self.mobject.tip)

        if self.mobject.has_start_tip():
            self.mobject.remove(self.mobject.start_tip)

    def _make_scaled_tips(
        self,
        path_length: float,
    ) -> tuple[
        ArrowTip | None,
        ArrowTip | None,
    ]:
        """Create scaled copies of the target tips."""
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

        if np.isclose(
            original_tip_width,
            0,
        ):
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

        return (
            end_tip,
            start_tip,
        )

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

        self.mobject.set_points(current_path.points.copy())

        if current_path.get_num_curves() == 0:
            return

        path_length = current_path.get_arc_length()

        if path_length <= 1e-12:
            return

        # Calculate stroke width before trimming for the tips.
        self.mobject._set_stroke_width_from_length()

        end_tip, start_tip = self._make_scaled_tips(path_length)

        # Do NOT call self.mobject.add_tip() here.
        #
        # add_tip() first calls position_tip(), but _trim_for_tip()
        # already positions the tip using _position_tip_between().
        # GrowArrow therefore attaches the animated tip directly.

        if end_tip is not None:
            self.mobject._trim_for_tip(
                end_tip,
                at_start=False,
            )
            self.mobject.assign_tip_attr(
                end_tip,
                at_start=False,
            )
            self.mobject.add(end_tip)

        if start_tip is not None:
            self.mobject._trim_for_tip(
                start_tip,
                at_start=True,
            )
            self.mobject.assign_tip_attr(
                start_tip,
                at_start=True,
            )
            self.mobject.add(start_tip)

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
