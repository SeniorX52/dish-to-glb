# Calibrate the printed board before you shoot anything

**Read this the moment the board comes out of the printer, not later.**

---

## Why this exists

`pipeline/make_charuco_pdf.py` draws the board at exactly 25.00 mm per square.
Your printer will not print it at exactly 25.00 mm per square.

Printer drivers apply a scale factor almost by default. "Fit to page" and
"Shrink to fit oversized pages" are ON by default in Adobe Reader, Chrome's PDF
viewer and most macOS print dialogs. A4 artwork pushed through a driver whose
default paper is US Letter comes out at **96.6%**. On top of that, laser fusing
shrinks paper by a tenth of a percent or so, and humidity moves it further.

That scale factor lands *directly* on every millimetre the pipeline reports. It
does not degrade gracefully and it does not look wrong:

| what the printer did | plate you measure at 260 mm is reported as | looks plausible? |
|---|---|---|
| exactly 100% | 260.0 mm | — |
| 96.6% ("fit to page" A4→Letter) | **269.2 mm** | yes, and it is 9 mm wrong |
| 98.5% | 264.0 mm | yes |
| 101.2% | 257.0 mm | yes |

There is no way to detect this from the photographs. The only fix is to measure
the paper.

---

## What to do

**1. Print it.**

```
python pipeline/make_charuco_pdf.py
# -> outputs/charuco/charuco_a4_7x6_25mm.pdf
```

- **Scale: 100% / "Actual size".** Turn OFF *Fit to page*, *Shrink to fit*,
  *Auto-rotate and centre*.
- **Matte paper.** Gloss and lamination throw specular highlights straight into
  the camera and kill marker detection at exactly the oblique angles you need.
- Heavier stock if you have it (160 gsm+). Plain 80 gsm is fine if you mount it.
- Check the black is solid. A streaky or grey-banded print loses markers.

**2. Measure the two black bars with digital calipers.**

The sheet carries two solid black reference bars, drawn at exactly:

| bar | where | nominal |
|---|---|---|
| **X** | horizontal, under the board | **125.00 mm** |
| **Y** | vertical, down the right edge | **75.00 mm** |

Measure the **full length of each bar, end to end**. Record to 0.01 mm.

Two bars, not one, because a printer can scale the axes differently — and if it
did, a single measurement would hide it inside a plausible-looking average.

> 125 mm is chosen to fit inside a standard 150 mm caliper. Toner spreads a bar
> edge by roughly 0.1 mm; that is ~0.08% here, an order of magnitude below the
> print scaling this is correcting. Ignore it. Do not try to correct for it.

**3. Put the readings in `config/charuco.json`.**

```json
"calibration": {
  "nominal_span_x_mm": 125.0,
  "nominal_span_y_mm": 75.0,
  "measured_span_x_mm": 124.62,
  "measured_span_y_mm": 74.78,
  "measured_on": "2026-08-28",
  "measured_by": "ofek",
  "instrument": "Mitutoyo 500-196 digital caliper",
  "paper": "160 gsm matte A4, laser",
  "notes": "taped flat to 5 mm foamboard"
}
```

Only the two `measured_span_*_mm` values change the maths. The rest is so that a
reprint six weeks from now is not silently confused with this sheet.

**4. Check what it derived.**

```
python -c "import sys; sys.path.insert(0,'pipeline'); import charuco; \
c=charuco.load(); print('effective square:', c.effective_square_mm); \
[print('WARN:', w) for w in c.warnings()]"
```

**5. Keep the sheet flat.** Tape it to foamboard, hardboard or glass. A curled
sheet is a scale error your calipers will not catch, because it measures correct
lying flat on the desk and short standing curled on the table.

---

## Reading the result

| reading | means | do |
|---|---|---|
| within ±0.1% (X: 124.88–125.13) | printer is honest | proceed |
| ±0.1% to ±0.5% | normal | proceed — the measured value is used |
| beyond ±0.5% | driver almost certainly scaled | check the print dialog, reprint |
| X and Y disagree by >0.5% | printer scaled the axes differently | **reprint** — do not average |
| beyond ±3% | "fit to page" was on | reprint |

Anisotropy is the one case where you must not just record the numbers and move
on. An averaged anisotropic scale is wrong on both axes at once, and it is wrong
in a way that looks like a good measurement.

---

## If you skip this

`config/charuco.json` reports itself uncalibrated rather than guessing:
`is_calibrated` is `False`, `effective_square_mm` is `None` (not `25.0`), and
`warnings()` says why. There is no code path that quietly substitutes the
nominal value.

> **Status:** that guarantee is enforced today in `pipeline/charuco.py`, which is
> what the metric-scale stage will read. The scale stage itself and
> `report.json` are priorities 3 and 4 and are **not built yet** — so right now
> nothing consumes this config. The contract those stages will honour: refuse to
> scale, record the stage as skipped with this reason, and surface it in
> `report.json`'s `warnings[]`. Verify that when they land.

A pipeline that silently assumes 1.0 produces numbers you would send to a client.

---

## Reprinting

Regenerating with different geometry changes the board id printed on the sheet
(`A4-7x6-25mm-DICT_5X5_1000-7D0B57C9`), so a physical sheet can always be matched
back to the config it came from.

`make_charuco_pdf.py` **refuses to overwrite a config that already holds caliper
measurements** — those numbers belong to a specific piece of paper. Use `--force`
when you genuinely are replacing the sheet, then re-measure the new one.

```
python pipeline/make_charuco_pdf.py --square-mm 30 --marker-mm 22 --force
```

---

## Verifying the PDF itself

The generator rasterises its own output at 600 dpi, runs OpenCV's ChArUco
detector over it, diffs it against `CharucoBoard.generateImage()`, and refuses to
report success unless the page is exactly A4, every marker decodes to the right
id in the right square, all interior corners are found, and the measured square
spacing is within two raster pixels of nominal.

Measured on the shipped board (`--verify`, 600 dpi, one raster pixel = 0.0423 mm):

```
page raster            4961 x 7016 px  (= A4 exactly)
markers                21 / 21, ids correct
interior corners       30 / 30
pixels differing from OpenCV's reference, off-edge     0
worst marker centre position error                     0.048 mm
mean measured square                                   25.005 mm
worst square spacing error                             0.024 mm  (0.094%)
```

That validates the **PDF**. It says nothing about your printer — which is what
the calipers are for.
