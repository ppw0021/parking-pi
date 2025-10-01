from flask import Flask, jsonify, render_template_string
import numpy as np

app = Flask(__name__)

# Example NumPy array
arr = np.array([1, 2, 3, 4, 5])

def compute_stats(a: np.ndarray) -> dict:
    """Return dictionary of simple NumPy computations."""
    return {
        "original": a.tolist(),
        "add10": (a + 10).tolist(),
        "times2": (a * 2).tolist(),
        "square": (a ** 2).tolist(),
        "sum": int(np.sum(a)),
        "mean": float(np.mean(a)),
        "max": int(np.max(a))
    }

@app.route("/api/array")
def api_array():
    """Return NumPy computations as JSON."""
    return jsonify(compute_stats(arr))

# Minimal HTML page that fetches /api/array and renders results
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>NumPy Example Server</title>
  <style>
    body { font-family: system-ui, -apple-system, Roboto, Arial; padding: 20px; max-width: 720px; }
    pre { background:#f7f7f7; padding: 12px; border-radius: 8px; }
    button { padding: 8px 12px; border-radius:6px; cursor:pointer; }
  </style>
</head>
<body>
  <h1>NumPy example (Flask)</h1>
  <p>Click the button to fetch computed results from <code>/api/array</code>.</p>
  <button id="fetchBtn">Fetch results</button>
  <div id="output" style="margin-top:16px"></div>

  <script>
    async function fetchResults() {
      const out = document.getElementById('output');
      out.innerHTML = '<em>Loading…</em>';
      try {
        const resp = await fetch('/api/array');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        out.innerHTML = `
          <h3>Results</h3>
          <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
      } catch (err) {
        out.innerHTML = '<strong>Error:</strong> ' + err;
      }
    }

    document.getElementById('fetchBtn').addEventListener('click', fetchResults);

    // optional: fetch on page load
    // fetchResults();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    # For development only. Use a production server (gunicorn/uvicorn) for deployment.
    app.run(host="0.0.0.0", port=5000, debug=True)
