#!/usr/bin/env python3
"""
Reads the saved bay status JSON and (for now) prints it in compact multi-line format.
When --dry is not set, it will send to the server.
"""

import json, argparse, requests

def format_json_multiline(data):
    lines = ["["]
    for i, item in enumerate(data):
        comma = "," if i < len(data) - 1 else ""
        lines.append(f'  {json.dumps(item, separators=(", ", ": "))}{comma}')
    lines.append("]")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="latest_spots.json")
    ap.add_argument("--url", default="http://127.0.0.1:5000/update_spots")
    ap.add_argument("--dry", action="store_true", help="Print only, don't send")
    args = ap.parse_args()

    with open(args.file, "r") as f:
        spots_data = json.load(f)

    print(format_json_multiline(spots_data))

    if args.dry:
        print("\n💡 Dry mode: not sending to server.")
        return

    try:
        response = requests.post(args.url, json=spots_data)
        print("Sent to server, response:", response.status_code, response.text)
    except Exception as e:
        print("Failed to send:", e)

if __name__ == "__main__":
    main()
