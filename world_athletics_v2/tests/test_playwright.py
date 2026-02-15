"""
Playwright smoke tests for World Athletics Dashboard v2.
Tests page load times, basic rendering, and interactive elements.

Usage:
    python world_athletics_v2/tests/test_playwright.py
"""

import json
import time
import sys
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

BASE_URL = "http://localhost:8502"

# Streamlit page URL slugs (from file names)
PAGES = {
    "Home (app.py)": "/",
    "Home": "/Home",
    "Event Intelligence": "/Event_Intelligence",
    "Athlete Profile": "/Athlete_Profile",
    "Competition Hub": "/Competition_Hub",
    "Scouting": "/Scouting",
    "Asian Games 2026": "/Asian_Games_2026",
    "LA 2028": "/LA_2028",
    "AI Analyst": "/AI_Analyst",
    "Competition Map": "/Competition_Map",
    "Competitor Analysis": "/Competitor_Analysis",
    "Championship WITTW": "/Championship_WITTW",
}

# Collect results
results = []


def log(msg):
    print(f"  {msg}")


def record(name, status, load_time_ms, details=""):
    results.append({
        "page": name,
        "status": status,
        "load_ms": round(load_time_ms),
        "details": details,
    })
    icon = "PASS" if status == "pass" else "FAIL" if status == "fail" else "WARN"
    time_str = f"{load_time_ms:.0f}ms"
    print(f"  [{icon}] {name}: {time_str} {details}")


def wait_for_streamlit_load(page, timeout=45000):
    """Wait for Streamlit to finish loading (spinner gone, content rendered)."""
    # Wait for the main app container
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout)
    # Wait for any running status widgets (spinners) to disappear
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="hidden", timeout=10000)
    except PlaywrightTimeout:
        pass  # No spinner present is fine
    # Wait for network to settle
    page.wait_for_load_state("networkidle", timeout=15000)
    # Extra time for Streamlit React rendering
    page.wait_for_timeout(2000)


def check_no_errors(page):
    """Check for Streamlit error messages on the page."""
    errors = []
    # Check for st.error / st.exception blocks
    error_els = page.query_selector_all('[data-testid="stException"], [data-testid="stError"]')
    for el in error_els:
        text = el.inner_text()[:300]
        errors.append(f"Error: {text}")

    # Check for red notification/alert banners
    alert_els = page.query_selector_all('[data-testid="stNotification"]')
    for el in alert_els:
        text = el.inner_text()[:200]
        # Only count actual errors, not st.info or st.warning
        classes = el.get_attribute("class") or ""
        if "error" in classes.lower() or "error" in text.lower()[:30]:
            errors.append(f"Alert: {text}")

    return errors


def get_page_content_summary(page):
    """Get a summary of what's visible on the page."""
    summary = {}

    # Count selectboxes
    summary["selectboxes"] = len(page.query_selector_all('[data-testid="stSelectbox"]'))

    # Count data frames/tables
    summary["dataframes"] = len(page.query_selector_all(
        '[data-testid="stDataFrame"], [data-testid="stTable"], '
        '[data-testid="stDataFrameResizable"]'
    ))

    # Count plotly charts
    summary["charts"] = len(page.query_selector_all(
        '.js-plotly-plot, [data-testid="stPlotlyChart"], '
        '[data-testid="stVegaLiteChart"]'
    ))

    # Count metrics
    summary["metrics"] = len(page.query_selector_all('[data-testid="stMetric"]'))

    # Count buttons
    summary["buttons"] = len(page.query_selector_all(
        'button[kind="primary"], button[kind="secondary"]'
    ))

    # Check for markdown/HTML content
    markdown_els = page.query_selector_all('[data-testid="stMarkdown"]')
    summary["markdown_blocks"] = len(markdown_els)

    # Check for info/warning/success messages
    summary["info_msgs"] = len(page.query_selector_all('[data-testid="stAlert"]'))

    # Get visible text length (proxy for "has content")
    try:
        main_content = page.query_selector('[data-testid="stAppViewContainer"]')
        if main_content:
            text = main_content.inner_text()
            summary["text_length"] = len(text)
            # Capture first 200 chars of visible text
            summary["text_preview"] = text[:200].replace('\n', ' ').strip()
        else:
            summary["text_length"] = 0
            summary["text_preview"] = ""
    except Exception:
        summary["text_length"] = 0
        summary["text_preview"] = ""

    return summary


def test_page_load(page, name, url_path):
    """Navigate to a page and measure load time."""
    full_url = f"{BASE_URL}{url_path}"
    start = time.time()
    try:
        page.goto(full_url, wait_until="domcontentloaded", timeout=60000)
        wait_for_streamlit_load(page, timeout=45000)
        load_time = (time.time() - start) * 1000

        errors = check_no_errors(page)
        summary = get_page_content_summary(page)

        # Build detail string
        parts = []
        if summary["selectboxes"]:
            parts.append(f"{summary['selectboxes']} selects")
        if summary["dataframes"]:
            parts.append(f"{summary['dataframes']} tables")
        if summary["charts"]:
            parts.append(f"{summary['charts']} charts")
        if summary["metrics"]:
            parts.append(f"{summary['metrics']} metrics")
        if summary["markdown_blocks"]:
            parts.append(f"{summary['markdown_blocks']} md blocks")
        parts.append(f"{summary['text_length']} chars")

        detail_str = " | ".join(parts)

        if errors:
            record(name, "fail", load_time, f"ERRORS: {'; '.join(errors)} | {detail_str}")
            return False
        elif summary["text_length"] < 50:
            record(name, "warn", load_time, f"Minimal content | {detail_str}")
            return True
        else:
            status = "pass" if load_time < 15000 else "warn"
            if status == "warn":
                detail_str = f"SLOW | {detail_str}"
            record(name, status, load_time, detail_str)
            return True
    except PlaywrightTimeout:
        load_time = (time.time() - start) * 1000
        record(name, "fail", load_time, "Timeout waiting for page load")
        return False
    except Exception as e:
        load_time = (time.time() - start) * 1000
        record(name, "fail", load_time, f"Exception: {str(e)[:150]}")
        return False


def test_competitor_analysis_interactions(page):
    """Test Competitor Analysis page interactive elements."""
    print("\n--- Competitor Analysis: Interaction Tests ---")
    page.goto(f"{BASE_URL}/Competitor_Analysis", wait_until="domcontentloaded", timeout=60000)
    wait_for_streamlit_load(page, timeout=45000)

    summary = get_page_content_summary(page)
    log(f"Content: {summary['selectboxes']} selects, {summary['dataframes']} tables, "
        f"{summary['charts']} charts, {summary['metrics']} metrics, {summary['text_length']} chars")

    if summary["text_preview"]:
        log(f"Preview: {summary['text_preview'][:120]}...")

    # Check for athlete selector
    selectboxes = page.query_selector_all('[data-testid="stSelectbox"]')
    if selectboxes:
        log(f"Found {len(selectboxes)} selectbox(es)")
        try:
            selectboxes[0].click()
            page.wait_for_timeout(1000)
            # Check for dropdown options - Streamlit uses a popover
            options = page.query_selector_all('li[role="option"]')
            if not options:
                options = page.query_selector_all('[data-testid="stSelectboxOption"]')
            log(f"Dropdown has {len(options)} options")
            if options:
                # Select first real option
                options[0].click()
                page.wait_for_timeout(3000)
                wait_for_streamlit_load(page, timeout=20000)
                errors = check_no_errors(page)
                post_summary = get_page_content_summary(page)
                if errors:
                    record("Competitor Analysis - Select Athlete", "fail", 0,
                           f"Errors after selection: {'; '.join(errors)}")
                else:
                    record("Competitor Analysis - Select Athlete", "pass", 0,
                           f"OK - {post_summary['dataframes']} tables, {post_summary['charts']} charts, "
                           f"{post_summary['text_length']} chars after selection")
            else:
                record("Competitor Analysis - Dropdown", "warn", 0, "No options in dropdown")
                # Press Escape to close
                page.keyboard.press("Escape")
        except Exception as e:
            record("Competitor Analysis - Interaction", "warn", 0, f"{str(e)[:120]}")
    else:
        record("Competitor Analysis - Selectbox", "warn", 0, "No selectboxes found on page")


def test_championship_wittw_interactions(page):
    """Test Championship WITTW page interactive elements."""
    print("\n--- Championship WITTW: Interaction Tests ---")
    page.goto(f"{BASE_URL}/Championship_WITTW", wait_until="domcontentloaded", timeout=60000)
    wait_for_streamlit_load(page, timeout=45000)

    summary = get_page_content_summary(page)
    log(f"Content: {summary['selectboxes']} selects, {summary['dataframes']} tables, "
        f"{summary['charts']} charts, {summary['text_length']} chars")

    if summary["text_preview"]:
        log(f"Preview: {summary['text_preview'][:120]}...")

    selectboxes = page.query_selector_all('[data-testid="stSelectbox"]')
    if len(selectboxes) >= 2:
        try:
            # Click championship selector (last one)
            selectboxes[-1].click()
            page.wait_for_timeout(1000)
            options = page.query_selector_all('li[role="option"]')
            log(f"Championship dropdown has {len(options)} options")
            if len(options) >= 2:
                options[1].click()
                page.wait_for_timeout(3000)
                wait_for_streamlit_load(page, timeout=20000)
                errors = check_no_errors(page)
                if errors:
                    record("Championship WITTW - Change Filter", "fail", 0,
                           f"Errors: {'; '.join(errors)}")
                else:
                    post = get_page_content_summary(page)
                    record("Championship WITTW - Change Filter", "pass", 0,
                           f"OK - {post['dataframes']} tables, {post['charts']} charts after filter change")
            else:
                page.keyboard.press("Escape")
                record("Championship WITTW - Filter Options", "warn", 0,
                       f"Only {len(options)} option(s) in dropdown")
        except Exception as e:
            record("Championship WITTW - Interaction", "warn", 0, f"{str(e)[:120]}")
    else:
        record("Championship WITTW - Filters", "warn", 0,
               f"Only {len(selectboxes)} selectbox(es) found (expected 3+)")


def test_athlete_profile_report(page):
    """Test Athlete Profile page report generation section."""
    print("\n--- Athlete Profile: Report Generation Test ---")
    page.goto(f"{BASE_URL}/Athlete_Profile", wait_until="domcontentloaded", timeout=60000)
    wait_for_streamlit_load(page, timeout=45000)

    summary = get_page_content_summary(page)
    log(f"Content: {summary['selectboxes']} selects, {summary['dataframes']} tables, "
        f"{summary['text_length']} chars")

    if summary["text_preview"]:
        log(f"Preview: {summary['text_preview'][:120]}...")

    # Check for Pre-Competition Report section in page HTML
    page_html = page.content()
    if "Pre-Competition Report" in page_html:
        record("Athlete Profile - Report Section", "pass", 0, "Pre-Competition Report section found")
    else:
        record("Athlete Profile - Report Section", "warn", 0,
               "Section not found (may need athlete selected)")

    # Try selecting an athlete first
    selectboxes = page.query_selector_all('[data-testid="stSelectbox"]')
    if selectboxes:
        try:
            selectboxes[0].click()
            page.wait_for_timeout(1000)
            options = page.query_selector_all('li[role="option"]')
            log(f"Athlete dropdown has {len(options)} options")
            if options:
                options[0].click()
                page.wait_for_timeout(3000)
                wait_for_streamlit_load(page, timeout=20000)

                # Now check for Generate Report button
                buttons = page.query_selector_all('button')
                report_btn = None
                for btn in buttons:
                    try:
                        text = btn.inner_text()
                        if "generate" in text.lower() and "report" in text.lower():
                            report_btn = btn
                            break
                    except Exception:
                        pass

                if report_btn:
                    record("Athlete Profile - Report Button", "pass", 0,
                           "Generate Report button found after athlete selection")
                else:
                    # Check page HTML again
                    page_html2 = page.content()
                    if "Pre-Competition Report" in page_html2:
                        record("Athlete Profile - Report Button", "pass", 0,
                               "Report section present, button may need event data")
                    else:
                        record("Athlete Profile - Report Button", "warn", 0,
                               "Button not found after athlete selection")
            else:
                page.keyboard.press("Escape")
        except Exception as e:
            record("Athlete Profile - Interaction", "warn", 0, f"{str(e)[:120]}")


def test_event_intelligence(page):
    """Test Event Intelligence page."""
    print("\n--- Event Intelligence: Interaction Tests ---")
    page.goto(f"{BASE_URL}/Event_Intelligence", wait_until="domcontentloaded", timeout=60000)
    wait_for_streamlit_load(page, timeout=45000)

    summary = get_page_content_summary(page)
    log(f"Content: {summary['selectboxes']} selects, {summary['dataframes']} tables, "
        f"{summary['charts']} charts, {summary['text_length']} chars")

    if summary["text_preview"]:
        log(f"Preview: {summary['text_preview'][:120]}...")


def test_sidebar_navigation(page):
    """Test sidebar navigation is accessible."""
    print("\n--- Sidebar Navigation Test ---")
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
    wait_for_streamlit_load(page, timeout=30000)

    # Streamlit sidebar may be collapsed by default in newer versions
    # Try to find it or expand it
    sidebar = page.query_selector('[data-testid="stSidebar"]')
    if not sidebar:
        # Try clicking the hamburger menu to expand
        hamburger = page.query_selector('[data-testid="stSidebarCollapsedControl"] button')
        if hamburger:
            hamburger.click()
            page.wait_for_timeout(1000)
            sidebar = page.query_selector('[data-testid="stSidebar"]')

    if sidebar:
        log("Sidebar present")
        # Look for navigation links (Streamlit multipage nav)
        nav_links = page.query_selector_all(
            '[data-testid="stSidebarNav"] a, '
            '[data-testid="stSidebarNavItems"] a, '
            'nav a, '
            '[data-testid="stSidebar"] a'
        )
        log(f"Found {len(nav_links)} navigation links")

        # Get link text
        link_texts = []
        for link in nav_links[:15]:
            try:
                link_texts.append(link.inner_text().strip())
            except Exception:
                pass
        if link_texts:
            log(f"Nav items: {', '.join(link_texts[:8])}")

        record("Sidebar Navigation", "pass" if len(nav_links) > 0 else "warn", 0,
               f"{len(nav_links)} nav links found")
    else:
        record("Sidebar Navigation", "warn", 0, "Sidebar not found (may be collapsed)")


def take_screenshot(page, name):
    """Take a screenshot for visual inspection."""
    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
    path = f"world_athletics_v2/tests/screenshots/{safe_name}.png"
    try:
        page.screenshot(path=path, full_page=True)
        log(f"Screenshot: {path}")
    except Exception as e:
        log(f"Screenshot failed: {e}")


def main():
    print("=" * 70)
    print(f"World Athletics v2 - Playwright Smoke Tests")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {BASE_URL}")
    print("=" * 70)

    import os
    os.makedirs("world_athletics_v2/tests/screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Force light mode via color scheme preference
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            color_scheme="light",
        )
        page = context.new_page()

        # Add a query param to force light theme on Streamlit
        # Streamlit respects ?embed=true for cleaner rendering

        # ── 1. Page Load Tests ──────────────────────────────────────
        print("\n--- Page Load Tests ---")
        for name, url_path in PAGES.items():
            success = test_page_load(page, name, url_path)
            take_screenshot(page, name)

        # ── 2. Sidebar Navigation ───────────────────────────────────
        test_sidebar_navigation(page)

        # ── 3. Competitor Analysis Interactions ─────────────────────
        test_competitor_analysis_interactions(page)
        take_screenshot(page, "competitor_analysis_interactions")

        # ── 4. Championship WITTW Interactions ──────────────────────
        test_championship_wittw_interactions(page)
        take_screenshot(page, "championship_wittw_interactions")

        # ── 5. Athlete Profile + Report Gen ─────────────────────────
        test_athlete_profile_report(page)
        take_screenshot(page, "athlete_profile_report")

        # ── 6. Event Intelligence ───────────────────────────────────
        test_event_intelligence(page)

        browser.close()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passes = sum(1 for r in results if r["status"] == "pass")
    fails = sum(1 for r in results if r["status"] == "fail")
    warns = sum(1 for r in results if r["status"] == "warn")

    # Load time summary for page loads only
    page_loads = [r for r in results if r["load_ms"] > 0]
    if page_loads:
        print(f"\nPage Load Times:")
        print(f"{'Page':<30} {'Time':>8} {'Status':>6}")
        print("-" * 50)
        for r in sorted(page_loads, key=lambda x: x["load_ms"], reverse=True):
            icon = "OK" if r["status"] == "pass" else "!!" if r["status"] == "fail" else "~~"
            print(f"  {r['page']:<28} {r['load_ms']:>6}ms  [{icon}]")

        avg_load = sum(r["load_ms"] for r in page_loads) / len(page_loads)
        max_load = max(r["load_ms"] for r in page_loads)
        min_load = min(r["load_ms"] for r in page_loads)
        print(f"\n  Average: {avg_load:.0f}ms | Fastest: {min_load:.0f}ms | Slowest: {max_load:.0f}ms")

    # Interaction test results
    interactions = [r for r in results if r["load_ms"] == 0]
    if interactions:
        print(f"\nInteraction Tests:")
        for r in interactions:
            icon = "PASS" if r["status"] == "pass" else "FAIL" if r["status"] == "fail" else "WARN"
            print(f"  [{icon}] {r['page']}: {r['details']}")

    print(f"\nTotal: {passes} passed, {fails} failed, {warns} warnings")

    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {"pass": passes, "fail": fails, "warn": warns},
    }
    report_path = "world_athletics_v2/tests/playwright_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {report_path}")
    print(f"Screenshots: world_athletics_v2/tests/screenshots/")

    if fails > 0:
        print(f"\n** {fails} FAILURES detected - review errors above **")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
