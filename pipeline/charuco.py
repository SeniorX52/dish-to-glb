"""ChArUco board definition — the single source of truth for the physical board.

Two consumers share this module on purpose:

  * `pipeline/make_charuco_pdf.py` draws the printable sheet from this spec.
  * the metric-scale stage detects the board using this spec.

If the two ever disagree about square size, every scaled measurement is
silently wrong by that ratio, so neither is allowed its own copy of the numbers.

The important field is `measured_square_mm`. Consumer-grade printers do not
reproduce a PDF at exactly 100% -- driver "fit to page" defaults, paper-size
mismatch and toner/paper shrink all show up as a scale error of a fraction of a
percent to several percent. That error propagates 1:1 into every dimension the
pipeline reports. So the *nominal* square size is what gets drawn, and the
*measured* square size is what gets used. Until somebody puts calipers on the
printed sheet and fills in the measurement, `is_calibrated` is False and
downstream stages must say so rather than quietly assuming 1.0.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "charuco.json"

# Kept as strings in the config so the JSON is readable and does not depend on
# OpenCV's integer enum values staying stable across versions.
DICTS = (
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
)


@dataclass
class BoardSpec:
    """Nominal geometry, as drawn on the PDF."""
    squares_x: int = 7
    squares_y: int = 6
    square_mm: float = 25.0
    marker_mm: float = 19.0
    dictionary: str = "DICT_5X5_1000"
    page: str = "a4"

    def __post_init__(self) -> None:
        if self.dictionary not in DICTS:
            raise ValueError(f"unknown dictionary {self.dictionary!r}; known: {list(DICTS)}")
        if not (0 < self.marker_mm < self.square_mm):
            raise ValueError(
                f"marker_mm ({self.marker_mm}) must be >0 and smaller than "
                f"square_mm ({self.square_mm}) -- the marker sits inside the square")
        if self.marker_mm / self.square_mm > 0.85:
            raise ValueError(
                f"marker/square ratio {self.marker_mm / self.square_mm:.2f} leaves "
                f"under 7.5% of white quiet zone on each side; ArUco needs that "
                f"border to find the marker. Keep the ratio at or below 0.85.")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a board smaller than 3x3 has too few interior corners to be useful")

    # ---------------------------------------------------------------- geometry
    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def n_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def n_corners(self) -> int:
        """Interior chessboard corners -- what the scale solve actually uses."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def marker_inset_mm(self) -> float:
        return (self.square_mm - self.marker_mm) / 2.0

    def is_black_square(self, col: int, row: int) -> bool:
        """Matches OpenCV's CharucoBoard layout: (0,0) is black, markers go in
        the white squares, ids increase in row-major order over those."""
        return (col + row) % 2 == 0

    def marker_squares(self) -> list[tuple[int, int, int]]:
        """[(marker_id, col, row)] in the order OpenCV assigns ids.

        Row 0 is the TOP row, matching CharucoBoard.generateImage() and
        getObjPoints(), whose board frame has +y pointing down.
        """
        out, mid = [], 0
        for row in range(self.squares_y):
            for col in range(self.squares_x):
                if not self.is_black_square(col, row):
                    out.append((mid, col, row))
                    mid += 1
        return out

    # ---------------------------------------------------------------- identity
    def board_id(self) -> str:
        """Short stable id printed on the sheet, so a physical sheet can always
        be matched back to the config it was generated from."""
        key = (f"{self.page}|{self.squares_x}x{self.squares_y}|{self.square_mm:.4f}|"
               f"{self.marker_mm:.4f}|{self.dictionary}")
        return hashlib.sha256(key.encode()).hexdigest()[:8].upper()

    def human_id(self) -> str:
        return (f"{self.page.upper()}-{self.squares_x}x{self.squares_y}-"
                f"{self.square_mm:g}mm-{self.dictionary}-{self.board_id()}")

    # ---------------------------------------------------------------- opencv
    def cv_dictionary(self):
        import cv2
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "this OpenCV build has no cv2.aruco -- install opencv-contrib-python")
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def cv_board(self, square_mm: float | None = None):
        """CharucoBoard in millimetres.

        Pass the *measured* square size (see Calibration.effective_square_mm) when
        detecting, so solved geometry comes out in real mm rather than in
        as-drawn mm.
        """
        import cv2
        s = float(self.square_mm if square_mm is None else square_mm)
        m = s * (self.marker_mm / self.square_mm)
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), s, m,
                                      self.cv_dictionary())


@dataclass
class Calibration:
    """What the calipers said about the sheet that actually came out of the printer.

    The two reference bars are deliberately different lengths, so each carries its
    own nominal. X is the long baseline (the primary scale measurement); Y is
    shorter and exists to catch a printer that scales the two axes differently,
    which averaging would otherwise hide.
    """
    nominal_span_x_mm: float = 125.0
    nominal_span_y_mm: float = 75.0
    measured_span_x_mm: float | None = None
    measured_span_y_mm: float | None = None
    # Last resort: a single square measured directly. Much noisier -- the reading
    # error is divided by 1 instead of by 5, so prefer the bars.
    measured_square_mm: float | None = None
    measured_on: str | None = None
    measured_by: str | None = None
    instrument: str | None = None
    printed_on: str | None = None
    paper: str | None = None
    notes: str | None = None

    # Above this the axes disagree enough that averaging them is hiding a fault.
    ANISOTROPY_WARN_PCT = 0.5

    def scale_factors(self) -> dict[str, float]:
        """measured / nominal per axis. 1.0 means the printer got it exactly right."""
        out = {}
        for ax, meas, nom in (("x", self.measured_span_x_mm, self.nominal_span_x_mm),
                              ("y", self.measured_span_y_mm, self.nominal_span_y_mm)):
            if meas is not None and meas > 0 and nom and nom > 0:
                out[ax] = float(meas) / float(nom)
        return out

    def print_scale(self) -> float | None:
        f = self.scale_factors()
        return sum(f.values()) / len(f) if f else None

    def effective_square_mm(self, spec: "BoardSpec") -> float | None:
        """Real-world square size, or None if nobody has measured the sheet."""
        s = self.print_scale()
        if s is not None:
            return spec.square_mm * s
        if self.measured_square_mm and self.measured_square_mm > 0:
            return float(self.measured_square_mm)
        return None

    def scale_error_pct(self, spec: "BoardSpec") -> float | None:
        """How far the print is off nominal. Positive = printed oversize."""
        eff = self.effective_square_mm(spec)
        return None if eff is None else (eff - spec.square_mm) / spec.square_mm * 100.0

    def anisotropy_pct(self) -> float | None:
        f = self.scale_factors()
        if len(f) < 2:
            return None
        return abs(f["x"] - f["y"]) / ((f["x"] + f["y"]) / 2.0) * 100.0


@dataclass
class CharucoConfig:
    board: BoardSpec = field(default_factory=BoardSpec)
    calibration: Calibration = field(default_factory=Calibration)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.effective_square_mm(self.board) is not None

    @property
    def effective_square_mm(self) -> float | None:
        return self.calibration.effective_square_mm(self.board)

    def uncalibrated_reason(self) -> str:
        return (
            f"config/charuco.json has no caliper measurement "
            f"(calibration.measured_span_x_mm / measured_span_y_mm are null), so the "
            f"printed board's real square size is unknown. The nominal "
            f"{self.board.square_mm:g} mm is what was DRAWN, not what was printed; "
            f"consumer printers rescale. Measure the printed sheet and fill the "
            f"config in -- see CALIBRATION_NOTE.md.")

    def warnings(self) -> list[str]:
        w: list[str] = []
        if not self.is_calibrated:
            w.append(self.uncalibrated_reason())
            return w
        err = self.calibration.scale_error_pct(self.board)
        if err is not None and abs(err) > 0.1:
            w.append(f"printed board is {err:+.2f}% off nominal "
                     f"({self.effective_square_mm:.3f} mm vs {self.board.square_mm:g} mm "
                     f"drawn); the measured value is being used.")
        aniso = self.calibration.anisotropy_pct()
        if aniso is not None and aniso > Calibration.ANISOTROPY_WARN_PCT:
            f = self.calibration.scale_factors()
            w.append(f"printed board is anisotropic: X scaled {f['x']:.4f}, Y scaled "
                     f"{f['y']:.4f} -- a {aniso:.2f}% disagreement "
                     f"(>{Calibration.ANISOTROPY_WARN_PCT}%). The printer scaled the "
                     f"axes differently; reprint at 100% rather than averaging it away.")
        if len(self.calibration.scale_factors()) == 1:
            w.append("only one reference bar was measured, so anisotropic printing "
                     "cannot be ruled out. Measure the other bar too.")
        return w

    # ---------------------------------------------------------------- json io
    def to_json(self) -> str:
        return json.dumps({
            "_comment": ("Physical ChArUco reference board. 'board' is what was DRAWN; "
                         "'calibration' is what was MEASURED off the printed sheet and "
                         "is what the pipeline actually uses. See CALIBRATION_NOTE.md."),
            "board": asdict(self.board),
            "calibration": asdict(self.calibration),
        }, indent=2) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoConfig":
        cal_in = d.get("calibration", {})
        if "span_squares" in cal_in or "nominal_span_mm" in cal_in:
            raise ValueError(
                "config/charuco.json uses the old calibration schema "
                "(span_squares / nominal_span_mm), where the spans were measured "
                "across board squares rather than across the printed reference bars. "
                "The two are not interchangeable and reusing the numbers would "
                "mis-scale the Y axis. Regenerate the board and re-measure:\n"
                "    python pipeline/make_charuco_pdf.py --force")
        board = BoardSpec(**{k: v for k, v in d.get("board", {}).items()
                             if k in BoardSpec.__dataclass_fields__})
        cal = Calibration(**{k: v for k, v in cal_in.items()
                             if k in Calibration.__dataclass_fields__})
        return cls(board=board, calibration=cal)


def load(path: Path | None = None) -> CharucoConfig:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- generate the board first:\n"
            f"    python pipeline/make_charuco_pdf.py")
    return CharucoConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save(cfg: CharucoConfig, path: Path | None = None) -> Path:
    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfg.to_json(), encoding="utf-8")
    return p
