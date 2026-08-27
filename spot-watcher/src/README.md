# Spot Watcher

Camera watches the 16 parking bays (6 top / 4 middle / 6 bottom = spot ids
0-15) and reports occupancy to the web server via `POST /update_spots`.

- `cal.py` - a small web app for placing the 16 bay boxes and capturing the
  empty-lot references (the Pi is headless; you drive it from a laptop browser).
- `main.py` - compare each bay to its reference every second and report.

## Setup

```bash
cd spot-watcher/src
uv sync
```

## Calibrate (once per physical setup)

```bash
uv run cal.py
```

It serves a page on port **8000**. Open `http://<pi-ip>:8000/` from a laptop on
the same network. Above the live camera view is a row of 16 numbered pads
(ordered top row 0-5, middle 6-9, bottom 10-15).

To place a bay:

1. click its pad (or it's armed for you)
2. click one corner of the bay in the image, then the opposite corner
3. it advances to the next unplaced pad automatically

Right-click restarts the current bay; **Esc** / **Stop placing** leaves
placement mode. Then, in edit mode: **drag** a box to move it, **drag the
yellow corner** to resize, **click** to select, **arrow keys** nudge
(Shift = x10), or edit the selected box's native-pixel `x y w h` in the readout.

- **Refresh view** re-grabs the frame (or tick `auto 2s`)
- **Clear all** wipes every box and re-arms pad 0
- **Save layout** writes `spots.json` (partial is fine - resume later)
- **Capture empty references** - all 16 placed, lot **empty** - writes
  `refs/00.png` .. `15.png` (also saves `spots.json`)

## Run

```bash
uv run main.py            # report to the web server
uv run main.py --dry-run  # print per-bay scores + write dryrun_preview.jpg, no server
```

Use `--dry-run` to tune `DIFF_THRESHOLD` (and `PIXEL_DELTA`) in `main.py`:
watch the scores with the lot empty vs. with cars, pick a threshold between
the two. Set `SERVER_URL` in `main.py`, and keep `CAMERA_INDEX` / `RESOLUTION`
matching between `cal.py` and `main.py`.

## Files

`spots.json` is committed (site layout). `refs/` and the `*preview*.jpg`
images are generated and gitignored.
