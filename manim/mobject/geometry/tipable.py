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
from manim.utils.space_ops import (
    cartesian_to_spherical,
    normalize,
)

if TYPE_CHECKING:
    from manim.mobject.geometry.tips import ArrowTip, ArrowTriangleFilledTip
    from manim.typing import (
        Point3D,
        Vector3DLike,
    )


class TipableVMobject(VMobject, metaclass=ConvertToOpenGL):
    """Meant for shared functionality between Arc and Line.
    Functionality can be classified broadly into these groups:

        * Adding, Creating, Modifying tips
            - add_tip calls create_tip, before pushing the new tip
                into the TipableVMobject's list of submobjects
            - stylistic and positional configuration

        * Checking for tips
            - Boolean checks for whether the TipableVMobject has a tip
                and a starting tip

        * Getters
            - Straightforward accessors, returning information pertaining
                to the TipableVMobject instance's tip(s), its length etc
    """

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
        self.tip_length: float = tip_length
        self.normal_vector = normal_vector
        self.tip_style: dict = tip_style if tip_style is not None else {}
        self.max_tip_length_to_length_ratio: float = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio: float = max_stroke_width_to_length_ratio
        self.initial_stroke_width = kwargs.get("stroke_width", stroke_width)
        super().__init__(stroke_width=stroke_width, **kwargs)

    # Adding, Creating, Modifying tips

    def add_tip(
        self,
        tip: ArrowTip | None = None,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
        at_start: bool = False,
    ) -> Self:
        """Adds a tip to the TipableVMobject instance, recognising
        that the endpoints might need to be switched if it's
        a 'starting tip' or not.
        """
        if tip is None:
            tip = self.create_tip(tip_shape, tip_length, tip_width, at_start)
        else:
            self.position_tip(tip, at_start)
        self.reset_endpoints_based_on_tip(tip, at_start)
        self.assign_tip_attr(tip, at_start)
        self.add(tip)
        return self

    def create_tip(
        self,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
        at_start: bool = False,
    ) -> ArrowTip:
        """Stylises the tip, positions it spatially, and returns
        the newly instantiated tip to the caller.
        """
        tip = self.get_unpositioned_tip(tip_shape, tip_length, tip_width)
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(
        self,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
    ) -> ArrowTip | ArrowTriangleFilledTip:
        """Returns a tip that has been stylistically configured,
        but has not yet been given a position in space.
        """
        from manim.mobject.geometry.tips import ArrowTriangleFilledTip

        style: dict[str, Any] = {}

        if tip_shape is None:
            tip_shape = ArrowTriangleFilledTip

        if tip_shape is ArrowTriangleFilledTip:
            if tip_width is None:
                tip_width = self.get_default_tip_length()
            style.update({"width": tip_width})
        if tip_length is None:
            tip_length = self.get_default_tip_length()

        color = self.get_color()
        style.update({"fill_color": color, "stroke_color": color})
        style.update(self.tip_style)
        tip = tip_shape(length=tip_length, **style)
        return tip

    def position_tip(self, tip: ArrowTip, at_start: bool = False) -> ArrowTip:
        # Last two control points, defining both
        # the end, and the tangency direction
        if at_start:
            anchor = self.get_start()
            handle = self.get_first_handle()
        else:
            handle = self.get_last_handle()
            anchor = self.get_end()
        angles = cartesian_to_spherical(handle - anchor)
        tip.rotate(
            angles[1] - PI - tip.tip_angle,
        )  # Rotates the tip along the azimuthal
        if not hasattr(self, "_init_positioning_axis"):
            axis = np.array(
                [
                    np.sin(angles[1]),
                    -np.cos(angles[1]),
                    0,
                ]
            )  # Obtains the perpendicular of the tip
            tip.rotate(
                -angles[2] + PI / 2,
                axis=axis,
            )  # Rotates the tip along the vertical wrt the axis
            self._init_positioning_axis = axis

        tip.shift(anchor - tip.tip_point)
        return tip

    def reset_endpoints_based_on_tip(self, tip: ArrowTip, at_start: bool) -> Self:
        if self.get_length() == 0:
            # Zero length, put_start_and_end_on wouldn't work
            return self

        if at_start:
            self.put_start_and_end_on(tip.base, self.get_end())
        else:
            self.put_start_and_end_on(self.get_start(), tip.base)
        return self

    def assign_tip_attr(self, tip: ArrowTip, at_start: bool) -> Self:
        if at_start:
            self.start_tip = tip
        else:
            self.tip = tip
        return self

    # Checking for tips

    def has_tip(self) -> bool:
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self) -> bool:
        return hasattr(self, "start_tip") and self.start_tip in self

    # Getters

    def pop_tips(self) -> VGroup:
        start, end = self.get_start_and_end()
        result = self.get_group_class()()
        if self.has_tip():
            result.add(self.tip)
            self.remove(self.tip)
        if self.has_start_tip():
            result.add(self.start_tip)
            self.remove(self.start_tip)
        if result.submobjects:
            self.put_start_and_end_on(start, end)
        return result

    def get_tips(self) -> VGroup:
        """Returns a VGroup (collection of VMobjects) containing
        the TipableVMObject instance's tips.
        """
        result = self.get_group_class()()
        if hasattr(self, "tip"):
            result.add(self.tip)
        if hasattr(self, "start_tip"):
            result.add(self.start_tip)
        return result

    def get_tip(self) -> VMobject:
        """Returns the TipableVMobject instance's (first) tip,
        otherwise throws an exception.
        """
        tips = self.get_tips()
        if len(tips) == 0:
            raise Exception("tip not found")
        else:
            tip: VMobject = tips[0]
            return tip

    def get_first_handle(self) -> Point3D:
        # Type inference of extracting an element from a list, is not
        # supported by numpy, see this numpy issue
        # https://github.com/numpy/numpy/issues/16544
        first_handle: Point3D = self.points[1]
        return first_handle

    def get_last_handle(self) -> Point3D:
        # Type inference of extracting an element from a list, is not
        # supported by numpy, see this numpy issue
        # https://github.com/numpy/numpy/issues/16544
        last_handle: Point3D = self.points[-2]
        return last_handle

    def get_end(self) -> Point3D:
        if self.has_tip():
            return self.tip.get_start()
        else:
            return super().get_end()

    def get_start(self) -> Point3D:
        if self.has_start_tip():
            return self.start_tip.get_start()
        else:
            return super().get_start()

    def get_length(self) -> float:
        return self.get_arc_length()

    def scale(self, factor: float, scale_tips: bool = False, **kwargs: Any) -> Self:  # type: ignore[override]
        r"""Scale an arrow, but keep stroke width and arrow tip size fixed.


        .. seealso::
            :meth:`~.Mobject.scale`

        Examples
        --------
        ::

            >>> arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
            >>> scaled_arrow = arrow.scale(2)
            >>> np.round(scaled_arrow.get_start_and_end(), 8) + 0
            array([[-2., -2.,  0.],
                    [ 2.,  2.,  0.]])
            >>> arrow.tip.length == scaled_arrow.tip.length
            True

        Manually scaling the object using the default method
        :meth:`~.Mobject.scale` does not have the same properties::

            >>> new_arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
            >>> another_scaled_arrow = VMobject.scale(new_arrow, 2)
            >>> another_scaled_arrow.tip.length == arrow.tip.length
            False

        """
        if self.get_length() == 0:
            return self

        if scale_tips:
            super().scale(factor, **kwargs)
            self._set_stroke_width_from_length()
            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()
        if has_tip or has_start_tip:
            old_tips = self.pop_tips()

        super().scale(factor, **kwargs)
        self._set_stroke_width_from_length()

        if has_tip:
            # error: Argument "tip" to "add_tip" of "TipableVMobject" has incompatible type "VMobject"; expected "ArrowTip | None"  [arg-type]
            self.add_tip(tip=cast("ArrowTip", old_tips[0]))
        if has_start_tip:
            # error: Argument "tip" to "add_tip" of "TipableVMobject" has incompatible type "VMobject"; expected "ArrowTip | None"  [arg-type]
            self.add_tip(tip=cast("ArrowTip", old_tips[1]), at_start=True)
        return self

    def get_normal_vector(self) -> Vector3D:
        """Returns the normal of a vector.

        Examples
        --------
        ::

            >>> np.round(Arrow().get_normal_vector()) + 0. # add 0. to avoid negative 0 in output
            array([ 0.,  0., -1.])
        """
        p0, p1, p2 = self.tip.get_start_anchors()[:3]
        return normalize(np.cross(p2 - p1, p1 - p0))

    def reset_normal_vector(self) -> Self:
        """Resets the normal of a vector"""
        self.normal_vector = self.get_normal_vector()
        return self

    def get_default_tip_length(self) -> float:
        """Returns the default tip_length of the arrow.

        Examples
        --------

        ::

            >>> Arrow().get_default_tip_length()
            0.35
        """
        max_ratio = self.max_tip_length_to_length_ratio
        return min(self.tip_length, max_ratio * self.get_length())

    def _set_stroke_width_from_length(self) -> Self:
        """Sets stroke width based on length."""
        max_ratio = self.max_stroke_width_to_length_ratio
        if config.renderer == RendererType.OPENGL:
            # Mypy does not recognize that the self object in this case
            # is a OpenGLVMobject and that the set_stroke method is
            # defined here:
            # mobject/opengl/opengl_vectorized_mobject.py#L248
            self.set_stroke(  # type: ignore[call-arg]
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                recurse=False,
            )
        else:
            self.set_stroke(
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                family=False,
            )
        return self
