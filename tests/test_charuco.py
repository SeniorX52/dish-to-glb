"""Self-tests for the ChArUco board spec and, above all, the calibration maths.

The calibration factor multiplies every dimension the pipeline ever reports. If
it is wrong the pipeline does not crash -- it produces confident, plausible,
wrong millimetres. So it gets tested directly.

    python tests/test_charuco.py          (no pytest required)
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import charuco
from charuco import BoardSpec, Calibration, CharucoConfig

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def close(a, b, tol=1e-9):
    return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=tol)


# ------------------------------------------------------------------ geometry
def test_geometry() -> None:
    print("geometry")
    s = BoardSpec()
    check("board mm", (s.width_mm, s.height_mm) == (175.0, 150.0))
    check("marker count", s.n_markers == 21)
    check("interior corners", s.n_corners == 30)
    check("marker inset", close(s.marker_inset_mm, 3.0))
    # Must match OpenCV's own layout, verified against CharucoBoard.generateImage:
    # (0,0) black, ids ascend row-major over the white squares.
    check("parity (0,0) black", s.is_black_square(0, 0))
    check("marker ids row-major",
          s.marker_squares()[:4] == [(0, 1, 0), (1, 3, 0), (2, 5, 0), (3, 0, 1)])
    check("every marker on a white square",
          all(not s.is_black_square(c, r) for _, c, r in s.marker_squares()))
    check("board id stable", s.board_id() == BoardSpec().board_id())
    check("board id tracks square size",
          BoardSpec(square_mm=20.0, marker_mm=15.0).board_id() != s.board_id())


def test_spec_rejects_bad_input() -> None:
    print("spec validation")
    for name, kw in (("marker >= square", dict(square_mm=25.0, marker_mm=25.0)),
                     ("marker negative", dict(marker_mm=-1.0)),
                     ("ratio too fat", dict(square_mm=25.0, marker_mm=23.0)),
                     ("unknown dict", dict(dictionary="DICT_9X9_1")),
                     ("board too small", dict(squares_x=2))):
        try:
            BoardSpec(**kw)
            check(f"rejects {name}", False, "-> constructed anyway")
        except ValueError:
            check(f"rejects {name}", True)


# --------------------------------------------------------------- calibration
def test_uncalibrated_is_not_silently_1() -> None:
    print("calibration: uncalibrated state")
    c = CharucoConfig()
    check("effective is None", c.effective_square_mm is None)
    check("is_calibrated False", c.is_calibrated is False)
    check("warns", len(c.warnings()) == 1 and "no caliper measurement" in c.warnings()[0])


def test_scale_from_bars() -> None:
    print("calibration: scale from reference bars")
    s = BoardSpec()

    # Printer came out exactly right.
    c = CharucoConfig(s, Calibration(measured_span_x_mm=125.0, measured_span_y_mm=75.0))
    check("perfect print -> nominal", close(c.effective_square_mm, 25.0))
    check("perfect print -> 0% error", close(c.calibration.scale_error_pct(s), 0.0))
    check("perfect print -> no warnings", c.warnings() == [])

    # A4 content pushed through a Letter driver: uniform 96.6% shrink.
    c = CharucoConfig(s, Calibration(measured_span_x_mm=125.0 * 0.966,
                                     measured_span_y_mm=75.0 * 0.966))
    check("uniform 96.6% -> 24.15 mm", close(c.effective_square_mm, 24.15, 1e-9))
    check("uniform 96.6% -> -3.4%", close(c.calibration.scale_error_pct(s), -3.4, 1e-9))
    check("uniform shrink is not flagged anisotropic",
          c.calibration.anisotropy_pct() < Calibration.ANISOTROPY_WARN_PCT)
    check("but is flagged as off-nominal",
          any("off nominal" in w for w in c.warnings()))

    # The bug this schema was rewritten to prevent: the bars have different
    # nominals, so equal *readings* on the two bars is a gross anisotropy, and
    # a naive mean-of-readings would have reported a bogus 1.0 scale.
    c = CharucoConfig(s, Calibration(measured_span_x_mm=100.0, measured_span_y_mm=100.0))
    f = c.calibration.scale_factors()
    check("per-axis nominals used", close(f["x"], 0.8) and close(f["y"], 100.0 / 75.0))
    check("gross anisotropy detected", c.calibration.anisotropy_pct() > 40)
    check("anisotropy warns", any("anisotropic" in w for w in c.warnings()))

    # Realistic laser printer: 0.6% wider in X than Y.
    c = CharucoConfig(s, Calibration(measured_span_x_mm=125.0 * 1.006,
                                     measured_span_y_mm=75.0 * 1.000))
    check("0.6% anisotropy flagged",
          close(c.calibration.anisotropy_pct(), 0.6 / 1.003 * 100 / 100, 1e-3)
          and any("anisotropic" in w for w in c.warnings()))


def test_partial_and_fallback() -> None:
    print("calibration: partial input and fallback")
    s = BoardSpec()
    c = CharucoConfig(s, Calibration(measured_span_x_mm=126.0))
    check("one bar still calibrates", close(c.effective_square_mm, 25.0 * 126.0 / 125.0))
    check("one bar warns about anisotropy",
          any("only one reference bar" in w for w in c.warnings()))
    check("one bar -> anisotropy unknown", c.calibration.anisotropy_pct() is None)

    c = CharucoConfig(s, Calibration(measured_square_mm=24.8))
    check("direct square reading used", close(c.effective_square_mm, 24.8))

    # Bars win over a direct reading: 5x the baseline, 5x less reading error.
    c = CharucoConfig(s, Calibration(measured_span_x_mm=125.0, measured_span_y_mm=75.0,
                                     measured_square_mm=99.0))
    check("bars beat direct reading", close(c.effective_square_mm, 25.0))

    for bad in (0.0, -3.0):
        c = CharucoConfig(s, Calibration(measured_span_x_mm=bad))
        check(f"span {bad} ignored, not divided by", c.effective_square_mm is None)


# ---------------------------------------------------------------- round trip
def test_json_round_trip() -> None:
    print("config json")
    s = BoardSpec(squares_x=5, squares_y=7, square_mm=30.0, marker_mm=22.0)
    c = CharucoConfig(s, Calibration(measured_span_x_mm=125.4, measured_span_y_mm=75.2,
                                     measured_by="ofek", instrument="Mitutoyo 500-196"))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "charuco.json"
        charuco.save(c, p)
        back = charuco.load(p)
    check("board survives", back.board == s)
    check("calibration survives", back.calibration == c.calibration)
    check("effective survives", close(back.effective_square_mm, c.effective_square_mm))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "charuco.json"
        p.write_text(json.dumps({"board": {}, "calibration": {
            "span_squares": 5, "nominal_span_mm": 125.0,
            "measured_span_x_mm": 125.4}}))
        try:
            charuco.load(p)
            check("legacy schema rejected", False, "-> loaded silently")
        except ValueError as e:
            check("legacy schema rejected", "old calibration schema" in str(e))


# ------------------------------------------------------- opencv agreement
def test_matches_opencv_board() -> None:
    print("opencv agreement")
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  skip  (no cv2/numpy)")
        return
    s = BoardSpec()
    b = s.cv_board()
    check("cv corner count", len(b.getChessboardCorners()) == s.n_corners)
    check("cv marker count", len(b.getIds()) == s.n_markers)
    # Our marker_squares() must agree with OpenCV about where each id lives,
    # or the drawn sheet and the detector disagree about the board.
    op = np.asarray(b.getObjPoints())
    worst = 0.0
    for k, mid in enumerate(np.asarray(b.getIds()).ravel()):
        col, row = next((c, r) for m, c, r in s.marker_squares() if m == int(mid))
        exp = np.array([col * s.square_mm + s.marker_inset_mm,
                        row * s.square_mm + s.marker_inset_mm])
        worst = max(worst, float(np.linalg.norm(op[k][0][:2] - exp)))
    check("marker placement matches OpenCV", worst < 1e-6, f"worst {worst}")

    # cv_board(measured) must rescale the whole board, keeping the aspect ratio.
    b2 = s.cv_board(24.15)
    c0 = np.asarray(b2.getChessboardCorners())[0]
    check("cv_board honours measured size", close(float(c0[0]), 24.15, 1e-6))
    check("cv_board keeps marker ratio",
          close(float(np.asarray(b2.getObjPoints())[0][1][0]
                      - np.asarray(b2.getObjPoints())[0][0][0]),
                24.15 * (s.marker_mm / s.square_mm), 1e-6))


for t in (test_geometry, test_spec_rejects_bad_input, test_uncalibrated_is_not_silently_1,
          test_scale_from_bars, test_partial_and_fallback, test_json_round_trip,
          test_matches_opencv_board):
    t()

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all passed")
