"""Scalar easing functions for normalized animation progress.

Inputs are finite scalars with t in [0, 1]. Functions do not validate or clamp
inputs. Default parameters give factors in [0, 1]; custom Bezier control values
can overshoot. These functions define interpolation factors, not motion limits.
"""

import math


class TransitionCurveLibrary:
    """Easing functions evaluated directly from normalized time."""

    @staticmethod
    def linear(t):
        """Linear transition curve"""
        return t

    @staticmethod
    def ease_in(t):
        """Ease-in curve (accelerating from zero velocity)"""
        return t * t

    @staticmethod
    def ease_out(t):
        """Ease-out curve (decelerating to zero velocity)"""
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def s_curve(t):
        """S-curve (ease-in-out) smooth acceleration and deceleration"""
        return 3 * t * t - 2 * t * t * t

    @staticmethod
    def bezier_cubic(t, p1=0.25, p2=0.75):
        """Scalar cubic Bezier with values (0, p1, p2, 1), evaluated at t.

        This does not invert a two-dimensional curve's time coordinate.
        """
        return 3 * (1 - t) * (1 - t) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t

    @staticmethod
    def polynomial_4th(t):
        """Legacy name for cubic smoothstep; retained for compatibility."""
        return t * t * (3 - 2 * t)

    @staticmethod
    def sigmoid(t, steepness=6):
        """Endpoint-normalized sigmoid; near-zero steepness uses its linear limit.

        Steepness must be finite. Its magnitude controls the curve shape;
        normalization gives the same result for positive and negative values.
        """
        # Below this magnitude the nonlinear correction is below float precision.
        if abs(steepness) < 1e-8:
            return t
        scale = math.tanh(steepness / 4)
        return (1 + math.tanh(steepness * (t - 0.5) / 2) / scale) / 2
