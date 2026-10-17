import math
from pathlib import Path
import subprocess
import sys
import unittest

from intempus_motion_tools import curves as curves_module
from intempus_motion_tools import TransitionCurveLibrary


class TransitionCurveTests(unittest.TestCase):
    def test_default_curves_reach_both_endpoints(self):
        curves = TransitionCurveLibrary
        for name in ("linear", "ease_in", "ease_out", "s_curve", "bezier_cubic",
                     "polynomial_4th", "sigmoid"):
            with self.subTest(curve=name):
                self.assertEqual(getattr(curves, name)(0.0), 0.0)
                self.assertEqual(getattr(curves, name)(1.0), 1.0)

    def test_sigmoid_remains_bounded_monotonic_and_symmetric(self):
        for steepness in (0, 1e-12, 1, 6, -6, 1e6):
            with self.subTest(steepness=steepness):
                values = [TransitionCurveLibrary.sigmoid(i / 100, steepness)
                          for i in range(101)]
                self.assertEqual(values[0], 0.0)
                self.assertEqual(values[-1], 1.0)
                self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in values))
                self.assertEqual(values, sorted(values))
                self.assertAlmostEqual(values[25] + values[75], 1.0)
        self.assertEqual(TransitionCurveLibrary.sigmoid(0.25, 0), 0.25)

    def test_legacy_polynomial_keeps_its_cubic_values(self):
        for time in (0, 0.1, 0.25, 0.5, 0.9, 1):
            self.assertAlmostEqual(TransitionCurveLibrary.polynomial_4th(time),
                                   TransitionCurveLibrary.s_curve(time))

    def test_tiny_sigmoid_steepness_approaches_linear_without_underflow_steps(self):
        for steepness in (math.ulp(0.0), 3 * math.ulp(0.0), 1e-322, 1e-320, 1e-12):
            for sign in (-1, 1):
                for time in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
                    with self.subTest(steepness=sign * steepness, time=time):
                        self.assertAlmostEqual(TransitionCurveLibrary.sigmoid(time, sign * steepness),
                                               time, places=15)

    def test_default_curves_are_bounded_and_monotonic(self):
        curves = TransitionCurveLibrary
        for name in ("linear", "ease_in", "ease_out", "s_curve", "bezier_cubic",
                     "polynomial_4th", "sigmoid"):
            with self.subTest(curve=name):
                values = [getattr(curves, name)(i / 100) for i in range(101)]
                self.assertTrue(all(0 <= v <= 1 for v in values))
                self.assertEqual(values, sorted(values))

    def test_bezier_is_a_scalar_polynomial_without_clamping(self):
        self.assertEqual(TransitionCurveLibrary.bezier_cubic(0.25, 0, 1), 0.15625)
        self.assertEqual(TransitionCurveLibrary.bezier_cubic(0.5, 2, 2), 1.625)

    def test_curve_module_imports_without_site_packages(self):
        path = Path(curves_module.__file__)
        code = (
            "import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('curves', {str(path)!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert module.TransitionCurveLibrary.sigmoid(0.5) == 0.5"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
