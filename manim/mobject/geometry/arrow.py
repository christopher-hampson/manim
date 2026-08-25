from __future__ import annotations

from manim.mobject.geometry.tips import ArrowTriangleFilledTip

# from manim.mobject.matrix import Matrix
from manim.typing import Vector2DLike

__all__ = [
    "Arrow",
    "Vector",
    "DoubleArrow",
    "CurvedArrow",
    "CurvedDoubleArrow",
]


from typing import TYPE_CHECKING, Any

import numpy as np

from manim.constants import *
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.tips import ArrowTip
from manim.utils.color import ParsableManimColor

if TYPE_CHECKING:
    from manim.mobject.geometry.line import Line
    from manim.mobject.geometry.tips import ArrowTip, ArrowTriangleFilledTip
    from manim.typing import (
        Vector3DLike,
    )

    from ..matrix import Matrix  # Avoid circular import


class Arrow(Line):
    """An arrow.

    Parameters
    ----------
    args
        Arguments to be passed to :class:`Line`.
    stroke_width
        The thickness of the arrow. Influenced by :attr:`max_stroke_width_to_length_ratio`.
    buff
        The distance of the arrow from its start and end points.
    max_tip_length_to_length_ratio
        :attr:`tip_length` scales with the length of the arrow. Increasing this ratio raises the max value of :attr:`tip_length`.
    max_stroke_width_to_length_ratio
        :attr:`stroke_width` scales with the length of the arrow. Increasing this ratio ratios the max value of :attr:`stroke_width`.
    kwargs
        Additional arguments to be passed to :class:`Line`.


    .. seealso::
        :class:`ArrowTip`
        :class:`CurvedArrow`

    Examples
    --------
    .. manim:: ArrowExample
        :save_last_frame:

        from manim.mobject.geometry.tips import ArrowSquareTip
        class ArrowExample(Scene):
            def construct(self):
                arrow_1 = Arrow(start=RIGHT, end=LEFT, color=GOLD)
                arrow_2 = Arrow(start=RIGHT, end=LEFT, color=GOLD, tip_shape=ArrowSquareTip).shift(DOWN)
                g1 = Group(arrow_1, arrow_2)

                # the effect of buff
                square = Square(color=MAROON_A)
                arrow_3 = Arrow(start=LEFT, end=RIGHT)
                arrow_4 = Arrow(start=LEFT, end=RIGHT, buff=0).next_to(arrow_1, UP)
                g2 = Group(arrow_3, arrow_4, square)

                # a shorter arrow has a shorter tip and smaller stroke width
                arrow_5 = Arrow(start=ORIGIN, end=config.top).shift(LEFT * 4)
                arrow_6 = Arrow(start=config.top + DOWN, end=config.top).shift(LEFT * 3)
                g3 = Group(arrow_5, arrow_6)

                self.add(Group(g1, g2, g3).arrange(buff=2))


    .. manim:: ArrowExample
        :save_last_frame:

        class ArrowExample(Scene):
            def construct(self):
                left_group = VGroup()
                # As buff increases, the size of the arrow decreases.
                for buff in np.arange(0, 2.2, 0.45):
                    left_group += Arrow(buff=buff, start=2 * LEFT, end=2 * RIGHT)
                # Required to arrange arrows.
                left_group.arrange(DOWN)
                left_group.move_to(4 * LEFT)

                middle_group = VGroup()
                # As max_stroke_width_to_length_ratio gets bigger,
                # the width of stroke increases.
                for i in np.arange(0, 5, 0.5):
                    middle_group += Arrow(max_stroke_width_to_length_ratio=i)
                middle_group.arrange(DOWN)

                UR_group = VGroup()
                # As max_tip_length_to_length_ratio increases,
                # the length of the tip increases.
                for i in np.arange(0, 0.3, 0.1):
                    UR_group += Arrow(max_tip_length_to_length_ratio=i)
                UR_group.arrange(DOWN)
                UR_group.move_to(4 * RIGHT + 2 * UP)

                DR_group = VGroup()
                DR_group += Arrow(start=LEFT, end=RIGHT, color=BLUE, tip_shape=ArrowSquareTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=BLUE, tip_shape=ArrowSquareFilledTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=YELLOW, tip_shape=ArrowCircleTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=YELLOW, tip_shape=ArrowCircleFilledTip)
                DR_group.arrange(DOWN)
                DR_group.move_to(4 * RIGHT + 2 * DOWN)

                self.add(left_group, middle_group, UR_group, DR_group)
    """

    def __init__(
        self,
        *args: Any,
        stroke_width: float = 6,
        buff: float = MED_SMALL_BUFF,
        max_tip_length_to_length_ratio: float = 0.25,
        max_stroke_width_to_length_ratio: float = 5,
        tip_shape: type[ArrowTip] = ArrowTriangleFilledTip,
        **kwargs: Any,
    ) -> None:
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        super().__init__(*args, buff=buff, stroke_width=stroke_width, **kwargs)  # type: ignore[misc]
        # TODO, should this be affected when
        # Arrow.set_stroke is called?
        self.initial_stroke_width = self.stroke_width
        self._set_stroke_width_from_length()
        self.add_tip(tip_shape=tip_shape)


class Vector(Arrow):
    """A vector specialized for use in graphs.

    .. caution::
        Do not confuse with the :class:`~.Vector2D`,
        :class:`~.Vector3D` or :class:`~.VectorND` type aliases,
        which are not Mobjects!

    Parameters
    ----------
    direction
        The direction of the arrow.
    buff
         The distance of the vector from its endpoints.
    kwargs
        Additional arguments to be passed to :class:`Arrow`

    Examples
    --------
    .. manim:: VectorExample
        :save_last_frame:

        class VectorExample(Scene):
            def construct(self):
                plane = NumberPlane()
                vector_1 = Vector([1,2])
                vector_2 = Vector([-5,-2])
                self.add(plane, vector_1, vector_2)
    """

    def __init__(
        self,
        direction: Vector2DLike | Vector3DLike = RIGHT,
        buff: float = 0,
        **kwargs: Any,
    ) -> None:
        self.buff = buff
        if len(direction) == 2:
            direction = np.hstack([direction, 0])

        super().__init__(ORIGIN, direction, buff=buff, **kwargs)

    def coordinate_label(
        self,
        integer_labels: bool = True,
        n_dim: int = 2,
        color: ParsableManimColor | None = None,
        **kwargs: Any,
    ) -> Matrix:
        """Creates a label based on the coordinates of the vector.

        Parameters
        ----------
        integer_labels
            Whether or not to round the coordinates to integers.
        n_dim
            The number of dimensions of the vector.
        color
            Sets the color of label, optional.
        kwargs
            Additional arguments to be passed to :class:`~.Matrix`.

        Returns
        -------
        :class:`~.Matrix`
            The label.

        Examples
        --------
        .. manim:: VectorCoordinateLabel
            :save_last_frame:

            class VectorCoordinateLabel(Scene):
                def construct(self):
                    plane = NumberPlane()

                    vec_1 = Vector([1, 2])
                    vec_2 = Vector([-3, -2])
                    label_1 = vec_1.coordinate_label()
                    label_2 = vec_2.coordinate_label(color=YELLOW)

                    self.add(plane, vec_1, vec_2, label_1, label_2)
        """
        # avoiding circular imports
        from ..matrix import Matrix

        vect = np.array(self.get_end())
        if integer_labels:
            vect = np.round(vect).astype(int)
        vect = vect[:n_dim]
        vect = vect.reshape((n_dim, 1))
        label = Matrix(vect, **kwargs)
        label.scale(LARGE_BUFF - 0.2)

        shift_dir = np.array(self.get_end())
        if shift_dir[0] >= 0:  # Pointing right
            shift_dir -= label.get_left() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * LEFT
        else:  # Pointing left
            shift_dir -= label.get_right() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * RIGHT
        label.shift(shift_dir)
        if color is not None:
            label.set_color(color)
        return label


class DoubleArrow(Arrow):
    """An arrow with tips on both ends.

    Parameters
    ----------
    args
        Arguments to be passed to :class:`Arrow`
    kwargs
        Additional arguments to be passed to :class:`Arrow`


    .. seealso::
        :class:`.~ArrowTip`
        :class:`.~CurvedDoubleArrow`

    Examples
    --------
    .. manim:: DoubleArrowExample
        :save_last_frame:

        from manim.mobject.geometry.tips import ArrowCircleFilledTip
        class DoubleArrowExample(Scene):
            def construct(self):
                circle = Circle(radius=2.0)
                d_arrow = DoubleArrow(start=circle.get_left(), end=circle.get_right())
                d_arrow_2 = DoubleArrow(tip_shape_end=ArrowCircleFilledTip, tip_shape_start=ArrowCircleFilledTip)
                group = Group(Group(circle, d_arrow), d_arrow_2).arrange(UP, buff=1)
                self.add(group)


    .. manim:: DoubleArrowExample2
        :save_last_frame:

        class DoubleArrowExample2(Scene):
            def construct(self):
                box = Square()
                p1 = box.get_left()
                p2 = box.get_right()
                d1 = DoubleArrow(p1, p2, buff=0)
                d2 = DoubleArrow(p1, p2, buff=0, tip_length=0.2, color=YELLOW)
                d3 = DoubleArrow(p1, p2, buff=0, tip_length=0.4, color=BLUE)
                Group(d1, d2, d3).arrange(DOWN)
                self.add(box, d1, d2, d3)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class CurvedArrow(Arrow):
    def __init__(
        self,
        *args: Any,
        buff: float = 0,
        path_arc: float = TAU / 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, buff=buff, path_arc=path_arc, **kwargs)


class CurvedDoubleArrow(CurvedArrow):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)
