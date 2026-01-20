import os
import json
import time
import requests
import subprocess
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Auto-refresh logic
# ─────────────────────────────────────────────────────────────────────
HEADERS_FILE = "refreshed_headers.json"
MAX_HEADER_AGE_DAYS = 2

def is_headers_stale(path=HEADERS_FILE, max_age_days=MAX_HEADER_AGE_DAYS):
    if not os.path.exists(path):
        return True
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds > (max_age_days * 86400)

def refresh_headers():
    print("🔄 Refreshing headers via Selenium...")
    subprocess.run(["python", "refresh_headers.py"], check=True)

def load_headers():
    if is_headers_stale():
        refresh_headers()
    with open(HEADERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

HEADERS = load_headers()

# ✅ Force known-good API key
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Referer": "https://worldathletics.org/",
    "Origin": "https://worldathletics.org",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "x-api-key": "da2-qmxd4dl6zfbihixs5ik7uhwor4",
    "Cookie": "_fbp=fb.1.1748159855701.107672299468395212; _ga_7FE9YV46NW=GS2.1.s1748159855$o1$g0$t1748159855$j0$l0$h0; _gcl_au=1.1.1317388703.1748159854; _hjSession_3465481=eyJpZCI6Ijg4MzQ4NWJmLTgzMWMtNDllZS05NmFkLTMzNWM0YjVmYTFkMiIsImMiOjE3NDgxNTk4NTE1MjQsInMiOjAsInIiOjAsInNiIjowLCJzciI6MCwic2UiOjAsImZzIjoxLCJzcCI6MX0=; _hjSessionUser_3465481=eyJpZCI6IjkyYzRlMDRhLTg3MGUtNWEwYi05MGZiLWVmODJjOWE0OGRlNCIsImNyZWF0ZWQiOjE3NDgxNTk4NTE1MjAsImV4aXN0aW5nIjpmYWxzZX0=; _ga=GA1.1.569950542.1748159855; NEXT_LOCALE=en; __exponea_time2__=59.99706292152405; CookieConsent={stamp:'zgCoCx/Vz20mwQCAO4z1O1ph0u7W0qKmVjpNxo/+gpc+PDuTUGddgw==',necessary:true,preferences:true,statistics:true,marketing:true,method:'explicit',ver:1,utc:1748159914084,region:'kr'}; __exponea_etc__=5e7d00e7-c576-4a95-9511-3c3ff8c316eb"
}


GRAPHQL_URL = "https://graphql-prod-4752.prod.aws.worldathletics.org/graphql"

# ─────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────
QUERIES = {
    "Profile": """
    query GetSingleCompetitorBasic($id: Int!) {
      getSingleCompetitor(id: $id) {
        id fullName gender birthDate country { name code }
        disciplines { discipline disciplineName indoor }
      }
    }
    """,
    "Rankings": """
    query GetSingleCompetitorWorldRanking($id: Int!) {
      getSingleCompetitorWorldRanking(id: $id) {
        disciplineName eventType indoor place score date rankingCategory
      }
    }
    """,
    "PBs": """
    query GetSingleCompetitorPB($id: Int!) {
      getSingleCompetitorPB(id: $id) {
        disciplineName indoor mark date competition venue country
      }
    }
    """,
    "SBs": """
    query GetSingleCompetitorSeasonBests($id: Int!) {
      getSingleCompetitorSeasonBests(id: $id) {
        disciplineName indoor mark date competition venue country
      }
    }
    """,
    "Results": """
    query GetSingleCompetitorResultsDate($id: Int) {
      getSingleCompetitorResultsDate(id: $id) {
        resultsByDate {
          date competition venue indoor discipline country category
          typeNameUrlSlug mark place wind notLegal remark
        }
      }
    }
    """,
    "Progression": """
    query GetSingleCompetitorProgression($id: Int!) {
      getSingleCompetitorProgression(id: $id) {
        disciplineName indoor data {
          season mark place competition venue country
        }
      }
    }
    """,
    "Honours": """
    query GetSingleCompetitorHonours($id: Int!) {
      getSingleCompetitorHonours(id: $id) {
        competition date event discipline mark place country category
      }
    }
    """
}

def fetch_query(name, query, athlete_id):
    payload = {
        "operationName": name,
        "variables": {"id": int(athlete_id)},
        "query": query
    }
    try:
        resp = requests.post(GRAPHQL_URL, headers=HEADERS, json=payload)
        if resp.status_code != 200:
            print(f"❌ {name} failed ({resp.status_code})")
            return None
        return resp.json().get("data")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None

def fetch_all_sections(athlete_id):
    print(f"\n🧾 Fetching data for athlete {athlete_id}...")
    results = {}
    for name, query in QUERIES.items():
        result = fetch_query(name, query, athlete_id)
        results[name] = result
        time.sleep(0.5)  # reduce server load
    return results

def save_data(athlete_id, data, output_dir="profiles"):
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, f"{athlete_id}_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    if "Rankings" in data:
        pd.DataFrame(data["Rankings"]["getSingleCompetitorWorldRanking"]).to_csv(f"{output_dir}/{athlete_id}_rankings.csv", index=False)
    if "PBs" in data:
        pd.DataFrame(data["PBs"]["getSingleCompetitorPB"]).to_csv(f"{output_dir}/{athlete_id}_pbs.csv", index=False)
    if "SBs" in data:
        pd.DataFrame(data["SBs"]["getSingleCompetitorSeasonBests"]).to_csv(f"{output_dir}/{athlete_id}_sbs.csv", index=False)
    if "Progression" in data:
        for p in data["Progression"]["getSingleCompetitorProgression"]:
            name = p["disciplineName"].replace(" ", "_")
            pd.DataFrame(p["data"]).to_csv(f"{output_dir}/{athlete_id}_progression_{name}.csv", index=False)
    if "Honours" in data:
        pd.DataFrame(data["Honours"]["getSingleCompetitorHonours"]).to_csv(f"{output_dir}/{athlete_id}_honours.csv", index=False)
    if "Results" in data:
        pd.DataFrame(data["Results"]["getSingleCompetitorResultsDate"]["resultsByDate"]).to_csv(f"{output_dir}/{athlete_id}_results.csv", index=False)

    print(f"✅ Saved data for {athlete_id}")

# ─────────────────────────────────────────────────────────────────────
# Run for all IDs in athlete_ids.txt
# ─────────────────────────────────────────────────────────────────────
def load_athlete_ids(file="athlete_ids.txt"):
    with open(file, "r") as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    athlete_ids = load_athlete_ids()
    for aid in athlete_ids:
        try:
            data = fetch_all_sections(aid)
            save_data(aid, data)
        except Exception as e:
            print(f"⚠️ Failed to process athlete {aid}: {e}")
