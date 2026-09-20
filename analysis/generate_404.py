#!/usr/bin/env python3
"""Generate reports/404.html.

Cloudflare Pages answers every unknown path with index.html and HTTP 200 unless a
404.html exists at the site root. This page makes a wrong link say so, and lists
every report so the reader can recover in one click.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import REPORT_REGISTRY, make_section, wrap_html  # noqa: E402


def report_list() -> str:
    items = "\n".join(
        f'<li><a href="{escape(r["file"])}">{escape(r["title"])}</a> <span style="color:#5B616E">{escape(r.get("desc", ""))}</span></li>'
        for r in REPORT_REGISTRY
    )
    return f'<ul style="line-height:1.7;padding-left:20px;">{items}</ul>'


def main() -> None:
    body = make_section(
        "This page does not exist",
        "<p>The address may have changed when the site was rebuilt. Every report on the site is listed below.</p>"
        + report_list(),
        section_id="missing",
    )
    html = wrap_html(
        title="Page not found",
        body_content=body,
        report_id="home",
        subtitle="404",
        header_meta="",
    )
    out = REPORTS_DIR / "404.html"
    out.write_text(html, encoding="utf-8")
    print(f"404 page written to {out}")


if __name__ == "__main__":
    main()
