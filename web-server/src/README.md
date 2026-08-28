# Web Server

Source of truth for parked vehicles, payments and live spot occupancy, and
(since gate-watcher was retired) the thing that drives the boom-gate servos.

## Prerequisites

- Python
- uv
- On the Pi wired to the gate servos only: `rpi-lgpio`
  (`uv pip install rpi-lgpio`). Off a Pi, `gate.py` falls back to
  `fake_gpio` and just prints what the servos/LEDs would have done.

## Run

```bash
# from the web-server/src directory
uv run main.py        # serves on 0.0.0.0:5000, debug=True
```

## Customer pages

A three-step flow, all sharing `templates/base.html`:

| Page    | URL     | What it does                                                            |
|---------|---------|-----------------------------------------------------------------------|
| Entry   | `/`     | Type your plate → `GET /enter/<plate>` → on 210 the entry gate opens. Also shows the live spots board. |
| Pay     | `/pay`  | `GET /check_plate/<plate>` for what's owed, `GET /pay/<plate>` to pay. |
| Exit    | `/exit` | Type your plate → `GET /exit/<plate>` → on 210 the exit gate opens and the car is removed. |

`/admin` (rate + current vehicles) is unchanged.

## Gate hardware

`gate.py` holds the servo/LED wiring (`SERVO_ENTRY_PIN`, `SERVO_EXIT_PIN`,
`SERVO_OPEN_ANGLE`, `GATE_DELAY`, `ENTRY_LED_PINS`, `EXIT_LED_PINS`) — the
values that used to live at the top of `gate-watcher/src/main.py`. Gate
pulses run on a background thread so requests return immediately.
