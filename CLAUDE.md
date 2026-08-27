# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A physical parking-lot automation demo built from three **independent** Python sub-projects that
run on separate machines (mostly Raspberry Pis) and talk to each other over HTTP on port 5000:

- `web-server/src/` — Flask app: the source of truth for parked vehicles, payments, and live spot
  occupancy. Serves the customer page (`/`) and admin page (`/admin`).
- `gate-watcher/src/` — runs on a Pi wired to boom-gate servos, RGB LEDs and buttons. Reads number
  plates from a camera with Tesseract OCR and drives the entry/exit gates.
- `spot-watcher/src/` — runs on a headless (SSH) Pi with a camera overlooking the bays. Compares
  each of the 16 bays against an empty-lot reference crop and pushes occupancy to the web server.

There is no repo-level build, test, or lint tooling. `Tests/TestStream.py` is an unrelated scratch
script, not a test suite.

## Commands

Each sub-project has its own `uv` environment and lockfile in its `src/` directory. Work from
inside that directory.

```bash
# one-time / after changing dependencies
cd <sub-project>/src && uv sync

# web-server — serves on 0.0.0.0:5000 with debug=True
cd web-server/src && uv run main.py

# gate-watcher — needs Pi GPIO hardware, a camera, and the tesseract binary installed
cd gate-watcher/src && uv run main.py

# spot-watcher — calibrate FIRST (see below), then:
cd spot-watcher/src && uv run main.py            # report to web server
cd spot-watcher/src && uv run main.py --dry-run  # print bay scores, no server
```

The web-server / gate-watcher `README.md` files say `uv run main`; the real entrypoint is
`main.py` in every sub-project.

spot-watcher is two scripts, both headless (no GUI — you inspect JPGs written to disk):
- `uv run cal.py` — interactive prompt to place the 16 bay boxes (`spots.json`) and, with the lot
  empty, capture reference crops (`refs/00.png`..`15.png`). Writes `cal_preview.jpg` after every
  change. Rows map to ids: `row0`→0-5, `row1`→6-9, `row2`→10-15. `main.py` exits if `spots.json`
  or `refs/` is missing.
- `uv run main.py` — every `INTERVAL_SEC`, crop each bay, score it against its reference, POST all
  16 to `/update_spots`. `--dry-run` prints per-bay scores and writes `dryrun_preview.jpg` instead
  of contacting the server — use it to tune `DIFF_THRESHOLD`.

`spots.json` is committed (site layout); `refs/` and `*preview*.jpg` are generated and gitignored.

## Configuration lives in source

There is no config file or env-var layer. Deployment settings are constants at the top of each
script and must be edited by hand when hardware or the network changes:

- `gate-watcher/src/main.py`: `WEB_PI_IP` (web server address — several old values are commented
  out), GPIO pin maps (`ENTRY_LED_PINS`, `EXIT_BUTTON_PIN`, `SERVO_ENTRY_PIN`, …), `CAMERA_INDEX`,
  `EXIT_ON_RIGHT` (which half of the frame is the exit lane), OCR/area thresholds.
- `spot-watcher/src/main.py`: `SERVER_URL` (web server, `/update_spots` is appended), `CAMERA_INDEX`,
  `RESOLUTION`, `INTERVAL_SEC`, `DIFF_THRESHOLD` / `PIXEL_DELTA` / `BRIGHT_TOLERANCE` (occupancy
  tuning). `CAMERA_INDEX` / `RESOLUTION` are repeated in `cal.py` — keep them in sync.

These addresses are frequently out of sync with each other — check them before assuming a
connectivity bug is in the code.

## Inter-service contract

Everything below is duplicated as literals across the three code bases; changing one side means
changing the others.

**Plate format:** `[A-Za-z]{3}\d{3}` (e.g. `abc123`), stored lower-case. The web server rejects
anything else on `/enter`.

**gate-watcher → web-server**, using non-standard HTTP status codes as the signal:
- `GET /enter/<plate>` → `210` added / `211` already parked / `213` error or bad format
- `GET /exit/<plate>`  → `210` paid, gate opens / `211` not paid / `212` plate not found / `213` error

On `210` gate-watcher pulses the servo open for `GATE_DELAY` s then closes; `211` lights red;
`212`/`213` blink red. LED policy is centralised in `gate-watcher/src/leds.py` (`LedControl`).

**spot-watcher → web-server:**
- `POST /update_spots` with a JSON array `[{"id": <int 0-15>, "taken": <bool>}]`. The server keeps
  occupancy only in the in-memory `parkingSpots` dict (16 slots); `GET /spots` returns it and the
  web pages poll it every second.

**Payment model (web-server):** SQLite table `parkingLot(plate, timeIn, paidToTime)`. A car may
exit while `paidToTime > now`. `/enter` grants `entryGracePeriod` (10 s) free; `/pay/<plate>` sets
`paidToTime = now + exitGracePeriod` (300 s) — note it does **not** currently charge by
`hourly_rate` or elapsed time. `hourly_rate` is an in-memory global, settable via
`POST /hourly-rate`.

## Things that will surprise you

- **`web-server/src/main.py` wipes the database on every startup** — it drops all tables then
  recreates `parkingLot`. `parkinglot.db` is gitignored. Restarting the server clears all state,
  including spot occupancy (in-memory) and `hourly_rate`.
- `app.run(...)` is called at module top level in `main.py`, so importing it starts the server.
- gate-watcher imports Raspberry-Pi-only libraries (`RPi.GPIO` via `rpi-lgpio`) and opens an OpenCV
  GUI window — it only runs meaningfully on the target Pi with hardware attached, not on a dev
  laptop. spot-watcher uses `opencv-python-headless` and never opens a window (it writes JPGs), but
  still needs a real camera.
- gate-watcher's OpenCV window is the operator UI: `e`/`x` toggle enter/exit-only scanning, `a`
  processes the bbox nearest the mouse, `[` / `]` adjust the detection area filter, and there is a
  "fine tune" trackbar mode for the plate-detection masks. The long changelog docstring at the top
  of `main.py` is the most detailed record of its behaviour.
- spot-watcher occupancy is a raw-pixel comparison: each bay crop is greyscaled, resized to
  128×128, blurred, brightness-matched to its reference within `BRIGHT_TOLERANCE`, then `taken` iff
  more than `DIFF_THRESHOLD` of pixels differ by more than `PIXEL_DELTA`. No lens correction, no
  CLAHE. Static references mean large lighting swings can still trip it — recalibrate (`cal.py` →
  `save`) if the camera or lighting changes.
