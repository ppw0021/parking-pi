# Parking PI
<img width="1476" height="787" alt="setup" src="https://github.com/user-attachments/assets/7e0459dd-d5fa-44ae-8c07-372fa182c060" />

Prerequisites
- Python
- UV
- CV2

Web cameras need to be connected to spot-watcher, and gate-watcher.

The boom arms need to be wired to gate-watcher, same with the buttons and LEDs.
![gates](https://github.com/user-attachments/assets/a7892f02-e3d5-40d4-a079-32cae406e872)

Run the app

```powershell
# from each `/src` directory under each folder (spot-watcher, gate-watcher, web-server)
uv run main
```
