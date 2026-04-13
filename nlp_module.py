#  nlp_module.py


import json     # To pretty-print and return structured output
import re       # Regular expressions for clean text splitting


WEATHER_KEYWORDS: dict[str, list[str]] = {
    "Hot":    ["hot", "warm", "sunny", "summer", "heat", "scorching", "blazing"],
    "Cold":   ["cold", "winter", "chilly", "freezing", "frost", "icy", "cool", "snow", "snowy"],
    "Rainy":  ["rainy", "rain", "wet", "drizzle", "monsoon", "shower", "umbrella"],
    "Windy":  ["windy", "wind", "breezy", "gusty", "storm", "stormy"],
    "Cloudy": ["cloudy", "overcast", "grey", "gray", "dull", "foggy", "misty"],
}

OCCASION_KEYWORDS: dict[str, list[str]] = {
    "Casual":     ["casual", "everyday", "relaxed", "chill", "lounge", "weekend", "daily"],
    "Formal":     ["formal", "office", "business", "professional", "meeting", "interview",
                   "corporate", "work", "job", "presentation"],
    "Party":      ["party", "celebration", "festive", "disco", "club", "night out",
                   "birthday", "wedding", "event", "gala"],
    "Sports":     ["sports", "gym", "workout", "exercise", "running", "jogging", "athletic",
                   "fitness", "training", "outdoor activity"],
    "Travel":     ["travel", "trip", "vacation", "holiday", "journey", "airport",
                   "backpacking", "tour", "explore"],
    "Date Night": ["date", "romantic", "dinner", "anniversary", "special occasion"],
}

STYLE_KEYWORDS: dict[str, list[str]] = {
    "Minimalist": ["minimal", "minimalist", "simple", "clean", "basic", "understated"],
    "Streetwear": ["streetwear", "street", "urban", "hype", "sneaker", "hoodie", "baggy"],
    "Elegant":    ["elegant", "classy", "sophisticated", "chic", "luxe", "luxury",
                   "refined", "tailored"],
    "Sporty":     ["sporty", "athletic", "active", "performance", "technical"],
    "Bohemian":   ["boho", "bohemian", "flowy", "earthy", "vintage", "retro", "hippie"],
    "Everyday":   ["everyday", "regular", "normal", "standard", "comfortable", "comfy"],
}

# Default values — used when the query doesn't mention a category
DEFAULTS = {
    "weather":  "Hot",
    "occasion": "Casual",
    "style":    "Everyday",
}



def preprocess_query(query: str) -> str:
    """
    Lowercase the text and remove punctuation so keyword matching is reliable.

    Example:
      "Formal! outfit for COLD weather?" → "formal outfit for cold weather"
    """
    query = query.lower().strip()                        # To lowercase
    query = re.sub(r"[^\w\s]", " ", query)              # Remove punctuation
    query = re.sub(r"\s+", " ", query)                  # Collapse whitespace
    return query
def find_match(text: str, keyword_dict: dict[str, list[str]]) -> str | None:
    """
    Scan `text` for any keyword in `keyword_dict`.
    Returns the first matching canonical label, or None.

    We sort the keyword list so longer phrases (e.g. "night out") are
    checked before single words to avoid partial matches.
    """
    for label, keywords in keyword_dict.items():
        # Sort by length descending — check longer phrases first
        for kw in sorted(keywords, key=len, reverse=True):
            if kw in text:
                return label
    return None     # No match found


def parse_query(user_query: str, verbose: bool = True) -> dict:
    """
    Parse free-form user text into a structured dict with three keys:
      weather, occasion, style

    Parameters
    ----------
    user_query : str
        Raw text from the user (e.g. "casual outfit for cold weather")
    verbose    : bool
        If True, prints step-by-step parsing details (useful for learning)

    Returns
    -------
    dict
        { "weather": str, "occasion": str, "style": str }
        All values are guaranteed non-None (defaults applied).
    """
    if verbose:
        print("=" * 55)
        print("  NLP QUERY PARSER")
        print("=" * 55)
        print(f"  Input query : \"{user_query}\"")

    # 1. Pre-process
    clean_text = preprocess_query(user_query)
    if verbose:
        print(f"  Cleaned text: \"{clean_text}\"")
        print()

    # 2. Extract each field
    weather  = find_match(clean_text, WEATHER_KEYWORDS)
    occasion = find_match(clean_text, OCCASION_KEYWORDS)
    style    = find_match(clean_text, STYLE_KEYWORDS)

    if verbose:
        print(f"  Weather  detected : {weather  or '(none found)'}")
        print(f"  Occasion detected : {occasion or '(none found)'}")
        print(f"  Style    detected : {style    or '(none found)'}")

    # 3. Apply defaults for undetected fields
    result = {
        "weather":  weather  or DEFAULTS["weather"],
        "occasion": occasion or DEFAULTS["occasion"],
        "style":    style    or DEFAULTS["style"],
    }

    # Track which fields were defaulted (nice for UI feedback)
    result["_defaults_applied"] = {
        "weather":  weather  is None,
        "occasion": occasion is None,
        "style":    style    is None,
    }

    if verbose:
        print()
        print("  ── Final Parsed Result ─────────────────────────")
        for key in ["weather", "occasion", "style"]:
            default_tag = " ← (default)" if result["_defaults_applied"][key] else ""
            print(f"   {key:<10}: {result[key]}{default_tag}")
        print("=" * 55)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PARSING — useful for testing multiple queries at once
# ─────────────────────────────────────────────────────────────────────────────
def batch_parse(queries: list[str]) -> list[dict]:
    """
    Run parse_query() over a list of user queries.
    Returns a list of result dicts (verbose=False to reduce noise).
    """
    results = []
    for q in queries:
        res = parse_query(q, verbose=False)
        results.append({"query": q, "parsed": res})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT helper
# ─────────────────────────────────────────────────────────────────────────────
def print_parsed(result: dict) -> None:
    """Print only the three main fields as clean JSON."""
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    print(json.dumps(clean, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Single query demo ────────────────────────────────────────────────────
    test_query = "I need a casual outfit for cold and rainy weather"
    result = parse_query(test_query, verbose=True)

    print("\n  Structured JSON output:")
    print_parsed(result)

    # ── Batch test — shows how the parser handles different inputs ───────────
    print("\n\n── BATCH TEST ──────────────────────────────────────────")
    sample_queries = [
        "formal look for a job interview",
        "party outfit on a hot summer night",
        "something sporty for the gym",
        "warm clothes for travel in winter",
        "date night elegant dress",
        "just give me something to wear",       # → all defaults applied
        "RAINY DAY casual streetwear vibes",    # → tests uppercase + slang
    ]

    batch_results = batch_parse(sample_queries)

    print(f"{'Query':<45} {'Weather':<10} {'Occasion':<12} {'Style'}")
    print("-" * 85)
    for item in batch_results:
        p = item["parsed"]
        print(f"{item['query'][:44]:<45} {p['weather']:<10} {p['occasion']:<12} {p['style']}")

    # ── Edge case: empty query ───────────────────────────────────────────────
    print("\n── Edge Case: Empty Query ──────────────────────────────")
    empty_result = parse_query("", verbose=True)
    print("  Result:", {k: v for k, v in empty_result.items() if not k.startswith("_")})
