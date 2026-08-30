"""Generate the printable ChArUco reference board as a PDF at exact physical size.

    python pipeline/make_charuco_pdf.py                 # A4, 7x6, 25.0 mm squares
    python pipeline/make_charuco_pdf.py --square-mm 20  # different square size
    python pipeline/make_charuco_pdf.py --verify        # rasterise + detect the PDF

Two things this does that a screenshot of an online board generator does not:

1.  **Vector, not raster.** Every square and every marker bit is a filled
    rectangle placed in PDF user space at an exact millimetre coordinate. There
    is no pixel grid to round to and no resampling, so the printer renders the
    edges at its own native resolution. Raster boards printed at 600 dpi put a
    ~0.04 mm quantisation on every marker bit boundary and soften the corners
    that the whole measurement depends on.

2.  **It admits it might be wrong.** The sheet carries reference bars of exactly
    known length and tells the operator to measure them, because the number the
    pipeline needs is the size the board came out of the *printer*, not the size
    it went into the PDF. Nothing downstream trusts the nominal value.

Verification (`--verify`, automatic when `pdftoppm` is on PATH) renders the
finished PDF back to a bitmap at a known dpi, runs OpenCV's ChArUco detector
over it, and checks that every marker and every interior corner is found and
that the corner spacing matches the nominal square size. That tests the actual
PDF, not the intention behind it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from charuco import CONFIG_PATH, BoardSpec, Calibration, CharucoConfig  # noqa: E402
from charuco import DICTS, load, save  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

PAGES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
# Every consumer printer has an unprintable border. 12 mm clears the worst
# inkjets; laser printers are usually happy at 5 mm.
SAFE_MARGIN_MM = 12.0

INK = (0, 0, 0)
GREY = (0.42, 0.42, 0.42)


# --------------------------------------------------------------------- drawing
def _runs(row: list[int]) -> list[tuple[int, int]]:
    """Merge a row of black cells into [start, length) runs.

    One rectangle per run instead of per cell: fewer PDF operators, and no
    hairline seam where two abutting rectangles are anti-aliased.
    """
    out, i, n = [], 0, len(row)
    while i < n:
        if row[i]:
            j = i
            while j < n and row[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def marker_bits(spec: BoardSpec, marker_id: int) -> list[list[int]]:
    """The marker as a (bits+2) square of 1=black cells, straight from OpenCV.

    Asking OpenCV for exactly one pixel per cell means the bit pattern is read
    out of the dictionary rather than reconstructed here, so it cannot drift
    from what the detector expects.
    """
    import cv2
    d = spec.cv_dictionary()
    side = d.markerSize + 2                      # +2 for the 1-cell black border
    img = cv2.aruco.generateImageMarker(d, marker_id, side, 1)
    return [[1 if v < 128 else 0 for v in row] for row in img]


def draw_board(c, spec: BoardSpec, x0_mm: float, y0_mm: float, MM) -> None:
    """Draw the board with its top-left corner at (x0_mm, y0_mm + height).

    Board rows are numbered from the top to match OpenCV's board frame, whose
    +y points down; PDF +y points up, so every row index is flipped here.
    """
    c.setFillColorRGB(*INK)
    c.setStrokeColorRGB(*INK)
    s = spec.square_mm

    for row in range(spec.squares_y):
        for col in range(spec.squares_x):
            if not spec.is_black_square(col, row):
                continue
            x = x0_mm + col * s
            y = y0_mm + (spec.squares_y - 1 - row) * s     # flip
            c.rect(x * MM, y * MM, s * MM, s * MM, stroke=0, fill=1)

    inset = spec.marker_inset_mm
    for mid, col, row in spec.marker_squares():
        bits = marker_bits(spec, mid)
        n = len(bits)
        cell = spec.marker_mm / n
        mx = x0_mm + col * s + inset
        my = y0_mm + (spec.squares_y - 1 - row) * s + inset
        for r, brow in enumerate(bits):
            # bits[0] is the marker's top row; flip into PDF space.
            yy = my + (n - 1 - r) * cell
            for start, length in _runs(brow):
                c.rect((mx + start * cell) * MM, yy * MM,
                       (length * cell) * MM, cell * MM, stroke=0, fill=1)


def fitted_string(c, x_mm, y_mm, text, font, size, colour, max_w_mm, MM) -> None:
    """Draw text, shrinking the font if it would run past max_w_mm.

    The instruction block sits beside the vertical reference bar. A line that
    overruns would print on top of the thing the operator has to measure, so the
    width is enforced rather than eyeballed.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth
    while size > 5.0 and stringWidth(text, font, size) / MM > max_w_mm:
        size -= 0.1
    c.setFont(font, size)
    c.setFillColorRGB(*colour)
    c.drawString(x_mm * MM, y_mm * MM, text)


def draw_ref_bar(c, x_mm, y_mm, length_mm, thick_mm, MM, vertical=False) -> None:
    c.setFillColorRGB(*INK)
    w, h = (thick_mm, length_mm) if vertical else (length_mm, thick_mm)
    c.rect(x_mm * MM, y_mm * MM, w * MM, h * MM, stroke=0, fill=1)


def build_pdf(cfg: CharucoConfig, out_pdf: Path, printed_on: str,
              ref_x_mm: float, ref_y_mm: float) -> dict:
    from reportlab.lib.units import mm as MM
    from reportlab.pdfgen import canvas

    spec = cfg.board
    pw, ph = PAGES_MM[spec.page]

    if spec.width_mm > pw - 2 * SAFE_MARGIN_MM or spec.height_mm > ph - 2 * SAFE_MARGIN_MM:
        raise SystemExit(
            f"board {spec.width_mm:g}x{spec.height_mm:g} mm does not fit on "
            f"{spec.page.upper()} ({pw:g}x{ph:g} mm) inside a {SAFE_MARGIN_MM:g} mm "
            f"printable margin. Reduce --square-mm or the square counts.")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=(pw * MM, ph * MM))
    c.setTitle(f"ChArUco reference board {spec.human_id()}")
    c.setSubject(f"Nominal square {spec.square_mm:g} mm, marker {spec.marker_mm:g} mm, "
                 f"{spec.dictionary}. Print at 100% - do not scale to fit.")
    c.setAuthor("dish-to-glb")
    c.setKeywords(f"charuco {spec.human_id()} square_mm={spec.square_mm:g} "
                  f"marker_mm={spec.marker_mm:g} printed_on={printed_on}")

    # ------------------------------------------------------------- geometry
    board_x = round((pw - spec.width_mm) / 2.0, 4)
    top_block = 35.0                                   # title + warning
    board_y = ph - top_block - spec.height_mm          # PDF y of board bottom
    bottom_h = board_y

    # The vertical bar must sit wholly inside the printable area: a bar clipped by
    # the printer's unprintable border reads short, and reads short *silently*.
    VBAR_Y = SAFE_MARGIN_MM + 2.0
    VBAR_CLEAR = 10.0                 # ... and stay clear of the board above it
    if bottom_h < VBAR_Y + ref_y_mm + VBAR_CLEAR:
        raise SystemExit(
            f"only {bottom_h:.1f} mm left below the board, need "
            f"{VBAR_Y + ref_y_mm + VBAR_CLEAR:.1f} mm for the {ref_y_mm:g} mm vertical "
            f"reference bar. Use fewer rows, a smaller --square-mm, or a shorter "
            f"--ref-y-mm.")

    # ------------------------------------------------------------- top block
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(board_x * MM, (ph - 9.5) * MM, "ChArUco metric reference board")
    c.setFont("Helvetica-Bold", 15)
    c.drawRightString((pw - board_x) * MM, (ph - 9.5) * MM,
                      f"SQUARE {spec.square_mm:.2f} mm")

    fitted_string(c, board_x, ph - 15.5,
                  f"marker {spec.marker_mm:.2f} mm  |  grid {spec.squares_x} x {spec.squares_y}"
                  f"  |  {spec.dictionary}  |  board {spec.width_mm:g} x {spec.height_mm:g} mm"
                  f"  |  {spec.n_markers} markers, {spec.n_corners} interior corners",
                  "Helvetica", 8.5, INK, pw - 2 * board_x, MM)

    # The single instruction that decides whether this sheet is usable at all.
    box_y = ph - 26.0
    c.setFillColorRGB(0.90, 0.90, 0.90)
    c.rect(board_x * MM, box_y * MM, (pw - 2 * board_x) * MM, 7.5 * MM, stroke=0, fill=1)
    fitted_string(c, board_x + 2, box_y + 2.6,
                  'PRINT AT 100% / "ACTUAL SIZE".  Turn OFF "Fit to page", "Shrink to fit" '
                  'and any auto-rotate.  Matte paper, not glossy.',
                  "Helvetica-Bold", 9.5, INK, pw - 2 * board_x - 4, MM)

    from reportlab.pdfbase.pdfmetrics import stringWidth
    id_w = stringWidth(spec.human_id(), "Helvetica-Oblique", 7.5) / MM
    fitted_string(c, board_x, ph - 31.0,
                  "These dimensions are NOMINAL - what was drawn, not what your printer "
                  "produced. Measure the bars below before this sheet is used for anything.",
                  "Helvetica-Oblique", 7.5, GREY, pw - 2 * board_x - id_w - 4, MM)
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColorRGB(*GREY)
    c.drawRightString((pw - board_x) * MM, (ph - 31.0) * MM, spec.human_id())

    # ------------------------------------------------------------- the board
    draw_board(c, spec, board_x, board_y, MM)

    # Corner crop marks: define the board's true outer extent even where a white
    # square meets the edge and leaves no ink to measure against.
    c.setStrokeColorRGB(*GREY)
    c.setLineWidth(0.25)
    gap, arm = 2.0, 6.0
    for cx, sx in ((board_x, -1), (board_x + spec.width_mm, 1)):
        for cy, sy in ((board_y, -1), (board_y + spec.height_mm, 1)):
            c.line((cx + sx * gap) * MM, cy * MM, (cx + sx * (gap + arm)) * MM, cy * MM)
            c.line(cx * MM, (cy + sy * gap) * MM, cx * MM, (cy + sy * (gap + arm)) * MM)

    # ---------------------------------------------------------- bottom block
    bar_t = 6.0
    hbar_y = board_y - 6.0 - bar_t
    draw_ref_bar(c, board_x, hbar_y, ref_x_mm, bar_t, MM)
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9)
    c.drawString((board_x + ref_x_mm + 4) * MM, (hbar_y + 1.6) * MM,
                 f"{ref_x_mm:.2f} mm  (X)")
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(*GREY)
    c.drawString(board_x * MM, (hbar_y - 4.0) * MM,
                 "measure the full length of this black bar, end to end")

    vbar_x = pw - board_x - bar_t
    vbar_y = VBAR_Y
    draw_ref_bar(c, vbar_x, vbar_y, ref_y_mm, bar_t, MM, vertical=True)
    c.saveState()
    c.translate((vbar_x - 2.0) * MM, vbar_y * MM)
    c.rotate(90)
    c.setFillColorRGB(*INK)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(0, 0, f"{ref_y_mm:.2f} mm  (Y)")
    c.restoreState()

    lines = [
        ("Helvetica-Bold", 10, INK,
         "AFTER PRINTING - do this before you shoot anything"),
        ("Helvetica", 8.2, INK,
         "1.  Digital calipers across the full length of the X bar above. Write it down to 0.01 mm."),
        ("Helvetica", 8.2, INK,
         "2.  Same for the Y bar down the right edge. Two axes, because a printer can scale them differently."),
        ("Helvetica", 8.2, INK,
         "3.  Put both readings into config/charuco.json:"),
        ("Courier-Bold", 8.2, INK,
         '        "measured_span_x_mm": <your X reading>,   "measured_span_y_mm": <your Y reading>'),
        ("Helvetica", 8.2, INK,
         "4.  Fill in measured_on / measured_by / paper too, so a future reprint is not confused with this sheet."),
        ("Helvetica-Oblique", 7.6, GREY,
         "Why: a printer that quietly scales to fit puts that error straight into every dimension the pipeline reports."),
        ("Helvetica-Oblique", 7.6, GREY,
         "A reading more than ~0.5% off nominal, or X and Y disagreeing by more than 0.5%, means reprint - do not average it out."),
        ("Helvetica-Oblique", 7.6, GREY,
         "Toner spreads roughly 0.1 mm on a bar edge. That is far below the print-scaling error this is correcting; ignore it."),
        ("Helvetica-Oblique", 7.6, GREY,
         "Keep the sheet flat - tape it down or mount it on board. A curl is a scale error the calipers will not catch."),
        ("Helvetica-Oblique", 7.6, GREY,
         "Full instructions: CALIBRATION_NOTE.md"),
    ]
    text_w = (vbar_x - 4.0) - board_x        # stop short of the vertical bar
    y = hbar_y - 12.0
    for font, size, colour, text in lines:
        fitted_string(c, board_x, y, text, font, size, colour, text_w, MM)
        y -= (size * 0.62)

    fitted_string(c, board_x, 6.0,
                  f"dish-to-glb  |  generated {printed_on}  |  {spec.human_id()}  |  "
                  f"regenerate: python pipeline/make_charuco_pdf.py "
                  f"--squares-x {spec.squares_x} --squares-y {spec.squares_y} "
                  f"--square-mm {spec.square_mm:g} --marker-mm {spec.marker_mm:g} "
                  f"--dict {spec.dictionary}",
                  "Helvetica", 6.5, GREY, pw - 2 * board_x, MM)

    c.showPage()
    c.save()
    return {"board_x_mm": board_x, "board_y_mm": board_y,
            "page_mm": [pw, ph], "bottom_block_mm": round(bottom_h, 2)}


# ---------------------------------------------------------------- verification
def verify_pdf(pdf: Path, spec: BoardSpec, geom: dict, dpi: int = 600) -> dict:
    """Rasterise the finished PDF and check the artefact, not the intention.

    Three independent questions, because each can fail without the others:

      1. Is the drawn geometry actually OpenCV's board? Crop the board out of the
         page raster and diff it against `CharucoBoard.generateImage()`. Any
         difference that is not on a rendered edge (i.e. not anti-aliasing) means
         a square or a marker bit is in the wrong place.
      2. Are the marker ids right, and in the right squares? A board that decodes
         to the correct ids in the wrong order still detects, and then reports
         confidently wrong geometry.
      3. Does the printed scale come out right? Measure corner-to-corner spacing
         off the raster and compare to the nominal square size.
    """
    import cv2
    import numpy as np

    if not shutil.which("pdftoppm"):
        return {"ran": False, "reason": "pdftoppm not on PATH (poppler-utils); "
                                        "skipped rasterised self-check"}

    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "page"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png",
                        "-singlefile", str(pdf), str(stem)], check=True,
                       capture_output=True)
        img = cv2.imread(str(stem.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"ran": False, "reason": "pdftoppm produced no readable image"}

    ppm = dpi / 25.4
    pw, ph = PAGES_MM[spec.page]
    res: dict = {
        "ran": True, "dpi": dpi,
        "raster_px": [int(img.shape[1]), int(img.shape[0])],
        "page_px_expected": [round(pw * ppm), round(ph * ppm)],
    }
    # pdftoppm ceilings the page dimension, and a page height that is an exact
    # integer number of pixels can land a float epsilon above it, so 1 px of slack.
    res["page_size_ok"] = all(
        abs(a - b) <= 1 for a, b in zip(res["raster_px"], res["page_px_expected"]))

    # ---- 1. pixel diff against OpenCV's own render of the same board
    top_mm = ph - (geom["board_y_mm"] + spec.height_mm)
    x0, y0 = round(geom["board_x_mm"] * ppm), round(top_mm * ppm)
    w, h = round(spec.width_mm * ppm), round(spec.height_mm * ppm)
    crop = img[y0:y0 + h, x0:x0 + w]
    if crop.shape != (h, w):
        res["pass"] = False
        res["error"] = f"board crop {crop.shape} != expected {(h, w)}"
        return res

    ref = spec.cv_board().generateImage((w, h), marginSize=0, borderBits=1)
    a = (crop < 128).astype(np.uint8)
    b = (ref < 128).astype(np.uint8)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

    # Two places the two renders may legitimately disagree, neither of them a
    # geometry error:
    #
    #   * within a pixel or two of any edge, where both are anti-aliasing;
    #   * on the outermost ring, because the board's true extent is a fractional
    #     number of pixels. generateImage() stretches the board to fill an integer
    #     raster exactly, whereas the vector render leaves the boundary row/column
    #     partly covered -- the vector one is the truthful of the two, and the
    #     difference is unmeasurable at this resolution either way.
    #
    # Everything further in is a real mismatch: a square or a marker bit is in the
    # wrong place.
    AA_PX = 2
    edge = cv2.dilate(cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
                      np.ones((2 * AA_PX + 1, 2 * AA_PX + 1), np.uint8))
    edge[:AA_PX, :] = 1
    edge[-AA_PX:, :] = 1
    edge[:, :AA_PX] = 1
    edge[:, -AA_PX:] = 1
    res["px_differing_pct"] = round(float(diff.sum()) / diff.size * 100.0, 4)
    res["px_differing_off_edge"] = int((diff & (1 - edge)).sum())

    # ---- 2. marker ids and their positions
    mc, mi, _ = cv2.aruco.ArucoDetector(spec.cv_dictionary()).detectMarkers(crop)
    found = sorted(int(i) for i in mi.ravel()) if mi is not None else []
    res["markers_found"] = len(found)
    res["markers_expected"] = spec.n_markers
    res["marker_ids_correct"] = (found == list(range(spec.n_markers)))
    worst_pos = 0.0
    where = {m: (col, row) for m, col, row in spec.marker_squares()}
    for quad, i in zip(mc or [], mi.ravel() if mi is not None else []):
        col, row = where[int(i)]
        ctr = np.asarray(quad).reshape(4, 2).mean(0) / ppm
        exp = np.array([(col + 0.5) * spec.square_mm, (row + 0.5) * spec.square_mm])
        worst_pos = max(worst_pos, float(np.linalg.norm(ctr - exp)))
    res["worst_marker_pos_error_mm"] = round(worst_pos, 5)

    # ---- 3. scale, from the interior corners the solve will actually use
    ch_corners, ch_ids, _, _ = cv2.aruco.CharucoDetector(spec.cv_board()).detectBoard(crop)
    n_ch = 0 if ch_ids is None else len(ch_ids)
    res["corners_found"] = n_ch
    res["corners_expected"] = spec.n_corners
    res["one_raster_pixel_mm"] = round(1.0 / ppm, 5)

    if n_ch == spec.n_corners:
        nx = spec.squares_x - 1
        pos = {int(i): np.asarray(q).ravel() for i, q in zip(ch_ids.ravel(), ch_corners)}
        d = []
        for cid, q in pos.items():
            if (cid % nx) + 1 < nx and cid + 1 in pos:
                d.append(float(np.linalg.norm(pos[cid + 1] - q)) / ppm)
            if cid + nx in pos:
                d.append(float(np.linalg.norm(pos[cid + nx] - q)) / ppm)
        res["square_mm_measured_mean"] = round(float(np.mean(d)), 5)
        res["square_mm_measured_std"] = round(float(np.std(d)), 5)
        worst = max(abs(v - spec.square_mm) for v in d)
        res["max_abs_error_mm"] = round(worst, 5)
        res["max_abs_error_pct"] = round(worst / spec.square_mm * 100.0, 4)
    else:
        worst = float("inf")

    res["pass"] = bool(
        res["page_size_ok"]
        and res["px_differing_off_edge"] == 0
        and res["marker_ids_correct"]
        and n_ch == spec.n_corners
        and worst <= 2.0 / ppm                 # within two raster pixels
        and worst_pos <= 2.0 / ppm)
    return res


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a printable ChArUco board PDF at exact physical size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = BoardSpec()
    ap.add_argument("--squares-x", type=int, default=d.squares_x)
    ap.add_argument("--squares-y", type=int, default=d.squares_y)
    ap.add_argument("--square-mm", type=float, default=d.square_mm)
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="default: 0.76 x square size")
    ap.add_argument("--dict", dest="dictionary", default=d.dictionary, choices=list(DICTS))
    ap.add_argument("--page", default=d.page, choices=list(PAGES_MM))
    ap.add_argument("--ref-x-mm", type=float, default=125.0,
                    help="length of the horizontal caliper reference bar")
    ap.add_argument("--ref-y-mm", type=float, default=75.0,
                    help="length of the vertical caliper reference bar (shorter than "
                         "X: it only has to catch anisotropic scaling)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--force", action="store_true",
                    help="overwrite a config that already holds caliper measurements")
    ap.add_argument("--verify", dest="verify", action="store_true", default=None)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--verify-dpi", type=int, default=600)
    args = ap.parse_args()

    spec = BoardSpec(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm if args.marker_mm else round(args.square_mm * 0.76, 3),
        dictionary=args.dictionary, page=args.page)

    # A config that already carries measurements describes a sheet somebody has
    # physically calibrated. Silently replacing it would leave the pipeline
    # scaling by a number that belongs to a different piece of paper.
    cal = Calibration()
    if args.config.exists():
        old = load(args.config)
        if old.is_calibrated and not args.force:
            raise SystemExit(
                f"{args.config} already holds caliper measurements for board "
                f"{old.board.human_id()} (effective square "
                f"{old.effective_square_mm:.4f} mm).\n"
                f"Refusing to overwrite. Pass --force if you really are replacing the "
                f"printed sheet -- and re-measure the new one afterwards.")
        if old.board == spec:
            cal = old.calibration                       # same board, keep any notes

    printed_on = time.strftime("%Y-%m-%d")
    # The nominals must describe the bars on THIS sheet, or the scale solve reads
    # the right number off the wrong ruler.
    cal.nominal_span_x_mm = args.ref_x_mm
    cal.nominal_span_y_mm = args.ref_y_mm

    cfg = CharucoConfig(board=spec, calibration=cal)

    out = args.out or (ROOT / "outputs" / "charuco" /
                       f"charuco_{spec.page}_{spec.squares_x}x{spec.squares_y}_"
                       f"{spec.square_mm:g}mm.pdf")
    geom = build_pdf(cfg, out, printed_on, args.ref_x_mm, args.ref_y_mm)

    cfg_path = save(cfg, args.config)

    print(f"board      {spec.human_id()}")
    print(f"page       {spec.page.upper()}  {geom['page_mm'][0]:g} x {geom['page_mm'][1]:g} mm")
    print(f"grid       {spec.squares_x} x {spec.squares_y}  "
          f"({spec.n_markers} markers, {spec.n_corners} interior corners)")
    print(f"nominal    square {spec.square_mm:g} mm   marker {spec.marker_mm:g} mm   "
          f"board {spec.width_mm:g} x {spec.height_mm:g} mm")
    print(f"ref bars   X {args.ref_x_mm:g} mm   Y {args.ref_y_mm:g} mm")
    print(f"pdf        {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print(f"config     {cfg_path}")

    run_verify = args.verify if args.verify is not None else bool(shutil.which("pdftoppm"))
    if run_verify:
        v = verify_pdf(out, spec, geom, args.verify_dpi)
        print("verify     " + json.dumps(v, indent=11)[1:-1].strip())
        if v.get("ran") and not v.get("pass"):
            raise SystemExit("PDF self-check FAILED - do not print this sheet")
    else:
        print("verify     skipped")

    print()
    print("NEXT: print at 100% (not 'fit to page'), then measure the two black bars "
          "with calipers and\n      put the readings in config/charuco.json. "
          "See CALIBRATION_NOTE.md.")


if __name__ == "__main__":
    main()
