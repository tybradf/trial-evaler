"""
Export the dashboard and explore pages as static HTML for GitHub Pages.
Deliberately does NOT export /live -- that page needs a running server with
a secret API key, which GitHub Pages structurally cannot provide. Instead,
the static site's "Live demo" nav link points to the real deployed app
(Render, see render.yaml) as an external link.

Renders templates directly with Jinja2 (not through Flask's routing), so
this has no dependency on a running server and no risk of accidentally
baking in Flask-specific paths.

    python app/build_static.py --live-url https://your-app.onrender.com

Output: app/static_build/{index.html, explore.html, static/*}
Point your GitHub Pages workflow at app/static_build/ as the publish dir.
"""
import argparse
import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "static_build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-url", required=True,
                     help="Full URL of the deployed live app, e.g. https://trial-evaler.onrender.com/live")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(APP_DIR))
    from config import MODEL_CONFIGS, PROJECT_BLURB, GITHUB_REPO_URL

    data = json.loads((APP_DIR / "data" / "dashboard_data.json").read_text())

    env = Environment(loader=FileSystemLoader(str(APP_DIR / "templates")))

    shared = {
        "nav": {"dashboard": "index.html", "explore": "explore.html", "live": args.live_url},
        "asset_prefix": "static/",
        "is_static_build": True,
        "project_blurb": PROJECT_BLURB,
        "github_url": GITHUB_REPO_URL,
    }

    OUT_DIR.mkdir(exist_ok=True)

    dashboard_html = env.get_template("dashboard.html").render(
        data=data, models=MODEL_CONFIGS, active_page="dashboard", **shared
    )
    (OUT_DIR / "index.html").write_text(dashboard_html)

    explore_html = env.get_template("explore.html").render(
        pairs=data["explorer_pairs"], models=MODEL_CONFIGS,
        active_page="explore", **shared
    )
    (OUT_DIR / "explore.html").write_text(explore_html)

    static_out = OUT_DIR / "static"
    if static_out.exists():
        shutil.rmtree(static_out)
    shutil.copytree(APP_DIR / "static", static_out, ignore=shutil.ignore_patterns("live.js"))

    print(f"Static site built -> {OUT_DIR}")
    print(f"  index.html, explore.html, static/ (style.css, site.js, explore.js)")
    print(f"  Live demo link points to: {args.live_url}")


if __name__ == "__main__":
    main()
