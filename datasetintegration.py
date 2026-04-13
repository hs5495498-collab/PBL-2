#  CSV COLUMN STRUCTURE (example)
#  │ Weather │ Occasion │   Style    │               Outfit                  │
#  │ Cold    │ Casual   │ Everyday   │ Chunky knit + jeans + Chelsea boots  │
#  │ Hot     │ Formal   │ Elegant    │ Linen blazer + trousers + loafers    │
#  │ Rainy   │ Party    │ Streetwear │ Leather jacket + slip dress + boots  │


import pandas as pd          
import random                
import os                    
from nlp_module import parse_query, print_parsed  


# BUILT-IN SAMPLE DATASET
# Used when no external CSV is provided (perfect for first-time runs)

SAMPLE_DATA = {
    "Weather":  ["Hot",   "Hot",     "Cold",   "Cold",    "Rainy",   "Rainy",
                 "Windy", "Windy",   "Cloudy", "Cloudy",  "Hot",     "Cold",
                 "Rainy", "Hot",     "Cold"],
    "Occasion": ["Casual","Formal",  "Casual", "Formal",  "Casual",  "Party",
                 "Travel","Formal",  "Casual", "Sports",  "Date Night","Sports",
                 "Travel","Party",   "Date Night"],
    "Style":    ["Everyday","Elegant","Everyday","Elegant","Everyday","Streetwear",
                 "Everyday","Elegant","Minimalist","Sporty","Elegant","Sporty",
                 "Everyday","Elegant","Elegant"],
    "Outfit":   [
        "White linen tee + chino shorts + slip-on sneakers + sunglasses",
        "Light linen blazer + white shirt + slim trousers + loafers",
        "Chunky knit sweater + straight jeans + Chelsea boots + beige scarf",
        "Wool suit + silk tie + Oxford shoes + cashmere pocket square",
        "Waterproof anorak + dark slim jeans + white sneakers + mini backpack",
        "Sequin mini skirt + black bodysuit + block-heel boots + statement earrings",
        "Packable windbreaker + straight jeans + sturdy boots + crossbody bag",
        "Double-breasted blazer + pressed shirt + wool trousers + Oxfords",
        "Oversized hoodie + high-waist jeans + clean white sneakers + cap",
        "Moisture-wicking tee + running shorts + trail runners + sports cap",
        "Off-shoulder top + linen trousers + strappy sandals + delicate necklace",
        "Thermal base layer + fleece joggers + running jacket + beanie",
        "Quick-dry trousers + waterproof sneakers + packable rain jacket + tote",
        "Floral midi dress + block-heel sandals + mini clutch + gold hoops",
        "Fitted turtleneck + leather trousers + block-heel boots + mini bag",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the wardrobe dataset.

    If csv_path is provided and the file exists, load from CSV.
    Otherwise, fall back to the built-in SAMPLE_DATA dictionary.

    Parameters
    ----------
    csv_path : str or None
        Path to your wardrobe CSV file.

    Returns
    -------
    pd.DataFrame with columns: Weather, Occasion, Style, Outfit
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[✔] Dataset loaded from '{csv_path}'")
    else:
        if csv_path:
            print(f"[!] '{csv_path}' not found — using built-in sample dataset.")
        else:
            print("[INFO] No CSV path provided — using built-in sample dataset.")
        df = pd.DataFrame(SAMPLE_DATA)

    # Normalize column names: strip whitespace, title-case
    df.columns = [c.strip().title() for c in df.columns]

    # Strip whitespace from all string cells
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    print(f"[✔] Dataset ready: {len(df)} outfit records\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Filter dataset by parsed NLP output
# ─────────────────────────────────────────────────────────────────────────────
def filter_outfits(
    df: pd.DataFrame,
    weather: str,
    occasion: str,
    style: str = None,
) -> pd.DataFrame:
    """
    Filter the dataset in priority order:
      1. Exact match on all three fields (Weather + Occasion + Style)
      2. If no result → relax Style (match Weather + Occasion only)
      3. If still no result → relax Occasion too (match Weather only)
      4. If still nothing → return the full dataset as a last resort

    This cascading approach ensures we always return SOMETHING useful.

    Parameters
    ----------
    df       : full wardrobe DataFrame
    weather  : detected weather label (e.g. "Cold")
    occasion : detected occasion label (e.g. "Casual")
    style    : detected style label (e.g. "Everyday") — optional

    Returns
    -------
    Filtered DataFrame (may contain 1 or more matching rows)
    """

    print("── Filtering Dataset ───────────────────────────────────")
    print(f"   Searching for: Weather={weather} | Occasion={occasion} | Style={style}")

    # ── Level 1: Exact match (all three columns) ──
    mask = (
        (df["Weather"].str.lower()  == weather.lower()) &
        (df["Occasion"].str.lower() == occasion.lower())
    )
    if style:
        mask &= (df["Style"].str.lower() == style.lower())

    filtered = df[mask]

    if not filtered.empty:
        print(f"   [✔] Exact match — {len(filtered)} outfit(s) found.")
        return filtered

    # ── Level 2: Relax style constraint ──
    print("   [~] No exact match. Relaxing Style filter…")
    mask2 = (
        (df["Weather"].str.lower()  == weather.lower()) &
        (df["Occasion"].str.lower() == occasion.lower())
    )
    filtered = df[mask2]

    if not filtered.empty:
        print(f"   [✔] Partial match (Weather + Occasion) — {len(filtered)} outfit(s) found.")
        return filtered

    # ── Level 3: Relax occasion too ──
    print("   [~] Still no match. Relaxing Occasion filter…")
    mask3 = df["Weather"].str.lower() == weather.lower()
    filtered = df[mask3]

    if not filtered.empty:
        print(f"   [✔] Loose match (Weather only) — {len(filtered)} outfit(s) found.")
        return filtered

    # ── Level 4: Return entire dataset as fallback ──
    print("   [!] No weather match either. Returning random outfit from full dataset.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Pick the best outfit from filtered results
# ─────────────────────────────────────────────────────────────────────────────
def pick_outfit(filtered: pd.DataFrame) -> pd.Series:
    """
    From the filtered results, pick one outfit.
    If multiple matches, choose randomly (adds variety for repeat users).
    """
    return filtered.sample(1).iloc[0]      # .sample(1) picks 1 random row


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Display recommendation nicely
# ─────────────────────────────────────────────────────────────────────────────
def display_recommendation(outfit: pd.Series, parsed: dict) -> None:
    """
    Print a styled recommendation summary.

    Parameters
    ----------
    outfit : one row from the DataFrame (pd.Series)
    parsed : the NLP-parsed dict for context
    """
    print("\n" + "=" * 55)
    print("  ✦  YOUR OUTFIT RECOMMENDATION")
    print("=" * 55)
    print(f"  Weather  : {outfit.get('Weather', 'N/A')}")
    print(f"  Occasion : {outfit.get('Occasion', 'N/A')}")
    print(f"  Style    : {outfit.get('Style', 'N/A')}")
    print()
    print("  OUTFIT:")
    # Split outfit string on '+' to print each piece on its own line
    pieces = [p.strip() for p in outfit.get("Outfit", "").split("+")]
    for i, piece in enumerate(pieces, 1):
        print(f"    {i}. {piece}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API — one function to rule them all
# ─────────────────────────────────────────────────────────────────────────────
def recommend_from_query(user_query: str, df: pd.DataFrame) -> dict:
    """
    End-to-end recommendation pipeline:
      user text → NLP parse → filter dataset → return outfit

    Parameters
    ----------
    user_query : str        — raw text from the user
    df         : DataFrame  — the loaded wardrobe dataset

    Returns
    -------
    dict with keys: weather, occasion, style, outfit
    """
    print("\n── NLP PARSING ─────────────────────────────────────────")
    # Parse the query (verbose=False to keep output clean here)
    parsed = parse_query(user_query, verbose=False)

    # Print what was extracted
    print(f"   Query    : \"{user_query}\"")
    print(f"   Weather  : {parsed['weather']}")
    print(f"   Occasion : {parsed['occasion']}")
    print(f"   Style    : {parsed['style']}")
    defaults = parsed.get("_defaults_applied", {})
    defaulted = [k for k, v in defaults.items() if v]
    if defaulted:
        print(f"   Defaults : {', '.join(defaulted)} (not found in query, default used)")

    # Filter dataset
    filtered = filter_outfits(df, parsed["weather"], parsed["occasion"], parsed["style"])

    # Pick one outfit
    outfit = pick_outfit(filtered)

    # Display result
    display_recommendation(outfit, parsed)

    # Return structured result for programmatic use
    return {
        "weather":  outfit.get("Weather"),
        "occasion": outfit.get("Occasion"),
        "style":    outfit.get("Style"),
        "outfit":   outfit.get("Outfit"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    # Change "wardrobe_dataset.csv" to your actual file path.
    # If the file doesn't exist the built-in sample dataset is used automatically.
    df = load_dataset("wardrobe_dataset.csv")

    # ── 2. Preview the dataset ───────────────────────────────────────────────
    print("── Dataset Preview (first 5 rows) ──────────────────────")
    print(df.head().to_string(index=False))
    print(f"\nTotal records : {len(df)}")
    print(f"Columns       : {list(df.columns)}\n")

    # ── 3. Single query recommendation ───────────────────────────────────────
    query_1 = "I need a casual outfit for cold weather"
    result_1 = recommend_from_query(query_1, df)

    # ── 4. More test queries ──────────────────────────────────────────────────
    test_queries = [
        "formal look for a hot day at the office",
        "something for a rainy party night, streetwear vibes",
        "gym clothes for cold winter morning",
        "date night elegant look for hot weather",
        "just give me something to wear",       # all defaults triggered
    ]

    print("\n\n── ADDITIONAL RECOMMENDATIONS ──────────────────────────\n")
    for q in test_queries:
        recommend_from_query(q, df)
        print()

    # ── 5. Direct filter demo (without NLP — using known values) ─────────────
    print("\n── DIRECT FILTER DEMO ──────────────────────────────────")
    print("   (Passing known values directly — no NLP parsing)")
    direct_filtered = filter_outfits(df, weather="Cold", occasion="Casual", style=None)
    print("\n   Matching outfits:")
    print(direct_filtered[["Weather", "Occasion", "Style", "Outfit"]].to_string(index=False))
