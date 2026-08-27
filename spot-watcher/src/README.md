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
the same network. You get the live camera view with 16 boxes on top:

- **drag** a box to move it, **drag the yellow corner** to resize
- **click** to select; **arrow keys** nudge (Shift = x10); the selected box's
  native-pixel `x y w h` are editable in the readout
- **Refresh view** re-grabs the frame (or tick `auto 2s`)
- **Auto-detect** fits the 6/4/6 layout to the bright vertical lane markers
- **Reset grid** drops back to an even grid
- **Save layout** writes `spots.json`
- **Capture empty references** - with the lot **empty** - writes
  `refs/00.png` .. `15.png` (also saves `spots.json`)

Auto-detect needs 3 rows of at least 2 bright markers each. When a row shows
exactly `count + 1` markers the bay edges come straight from them; otherwise
that row is split evenly between its outermost markers. Tune `MARKER_THRESH`
and the `MARKER_*` filters at the top of `cal.py` if it misses or over-detects.

Boxes are ordered top row 0-5, middle 6-9, bottom 10-15.

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
