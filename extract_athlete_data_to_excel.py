import json
import pandas as pd
from pathlib import Path

# === Step 1: Load JSON ===
athlete_id = "15017843"
input_path = Path(f"athlete_{athlete_id}_graphql_live.json")
output_excel = Path(f"athlete_{athlete_id}_data.xlsx")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

# === Step 2: Parse Sections ===

# Bio
bio_raw = data.get("getSingleCompetitor", {})
bio = {
    "full_name": bio_raw.get("fullName"),
    "gender": bio_raw.get("gender"),
    "birth_date": bio_raw.get("birthDate"),
    "country": bio_raw.get("country", {}).get("name"),
}
bio_df = pd.DataFrame([bio])

# PBs
pbs = data.get("getSingleCompetitorPB", [])
pb_df = pd.DataFrame(pbs) if pbs else pd.DataFrame(columns=["disciplineName", "mark", "date", "competition", "venue", "country"])

# SBs
sbs = data.get("getSingleCompetitorSeasonBests", [])
sb_df = pd.DataFrame(sbs) if sbs else pd.DataFrame(columns=["disciplineName", "mark", "date", "competition", "venue", "country"])

# Results by distribution
results_by_limit = data.get("getSingleCompetitorResultsByLimit", {}).get("results", [])
results_df = pd.DataFrame(results_by_limit) if results_by_limit else pd.DataFrame(columns=["result", "count", "totalCount"])

# Progression
progressions = data.get("getSingleCompetitorProgression", [])
prog_rows = []
for p in progressions:
    for row in p.get("data", []):
        row["discipline"] = p.get("disciplineName")
        prog_rows.append(row)
progression_df = pd.DataFrame(prog_rows) if prog_rows else pd.DataFrame(columns=["season", "mark", "place", "discipline"])

# Honours
honours = data.get("getSingleCompetitorHonours", [])
honours_df = pd.DataFrame(honours) if honours else pd.DataFrame(columns=["competition", "date", "event", "discipline", "mark", "place", "country", "category"])

# Rankings
rankings = data.get("getSingleCompetitorWorldRanking", [])
rankings_df = pd.DataFrame(rankings) if rankings else pd.DataFrame(columns=["disciplineName", "place", "score", "date", "rankingCategory"])

# === Step 3: Save to Excel ===
with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    bio_df.to_excel(writer, index=False, sheet_name="Bio")
    pb_df.to_excel(writer, index=False, sheet_name="PBs")
    sb_df.to_excel(writer, index=False, sheet_name="SBs")
    results_df.to_excel(writer, index=False, sheet_name="Results Dist")
    progression_df.to_excel(writer, index=False, sheet_name="Progression")
    honours_df.to_excel(writer, index=False, sheet_name="Honours")
    rankings_df.to_excel(writer, index=False, sheet_name="Rankings")

print(f"✅ Athlete data saved to: {output_excel}")
