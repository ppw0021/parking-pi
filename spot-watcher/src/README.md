# Spot Watcher

Camera watches the 16 parking bays (6 top / 4 middle / 6 bottom = spot ids
0-15) and reports occupancy to the web server via `POST /update_spots`.

Two scripts, both headless (SSH, no GUI):

- `cal.py` - place the 16 bay boxes and capture the empty-lot references.
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

First run tries to **auto-detect** the bays from the bright vertical lane
markers and fit the fixed 6/4/6 layout; if that fails it falls back to an even
grid. It writes `spots.json`, `cal_preview.jpg`, and `cal_auto_debug.jpg` (red =
detected markers, green = resulting boxes). Open `cal_preview.jpg`, then at the
`cal>` prompt adjust boxes until each sits on its bay (rewrites `spots.json` +
`cal_preview.jpg` after every change):

| command | effect |
| --- | --- |
| `auto` | re-run marker auto-detection |
| `5 120 340 90 160` | set box 5 to `x y w h` |
| `row0 60 90 1180 260` | set the top row's outer rectangle, split evenly into 6 |
| `mv 5 -4 0` | nudge box 5 |
| `grid` | reset to the default 6/4/6 grid |
| `show` | print the current boxes |
| `shot` | re-grab the camera frame and redraw the preview |
| `save` | with the lot **empty**, capture `refs/00.png` .. `15.png` |
| `q` | quit |

Auto-detect needs 3 rows of at least 2 bright markers each. When a row shows
exactly `count + 1` markers the bay edges come straight from them; otherwise
that row is split evenly between its outermost markers. Tune `MARKER_THRESH`
and the `MARKER_*` filters at the top of `cal.py` if it misses or over-detects.

Rows map to ids: `row0` = 0-5, `row1` = 6-9, `row2` = 10-15.

## Run

```bash
uv run main.py            # report to the web server
uv run main.py --dry-run  # print per-bay scores + write dryrun_preview.jpg, no server
```

Use `--dry-run` to tune `DIFF_THRESHOLD` (and `PIXEL_DELTA`) in `main.py`:
watch the scores with the lot empty vs. with cars, pick a threshold between
the two. Set `SERVER_URL`, `CAMERA_INDEX` and `RESOLUTION` at the top of both
scripts for the site.

## Files

`spots.json` is committed (site layout). `refs/` and the `*preview*.jpg`
images are generated and gitignored.
