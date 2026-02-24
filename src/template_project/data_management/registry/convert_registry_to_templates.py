"""One-off script: Convert existing series_registry.csv to series_templates.csv.

This script reads the 148-row Phase A registry (DE + U2 only) and produces
a template CSV with placeholders for multi-country expansion.

Transformations:
- DE Eurostat: "geo": "DE" → "geo": "{COUNTRY_EUROSTAT}"
- DE ECB (country-level): .DE. in key → .{COUNTRY_ISO2}.
- EA ECB (EA-level): scope=ea, no placeholders
- DE OECD: DEU. in key → {COUNTRY_OECD}.
- DE BIS: .DE at end of key → .{COUNTRY_ISO2}

Run once, then delete this script.
"""

import csv
import json
import re
from pathlib import Path


def main():
    registry_path = Path(__file__).parent / "series_registry.csv"
    templates_path = Path(__file__).parent / "series_templates.csv"

    with open(registry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from series_registry.csv")

    templates = []
    for row in rows:
        template = convert_row(row)
        templates.append(template)

    # Write templates
    fieldnames = [
        "template_id",
        "source",
        "category",
        "category_name",
        "variable_name",
        "dataset",
        "key_template",
        "filters_json_template",
        "scope",
        "unit_measure_filter",
        "frequency",
        "start_period",
    ]

    with open(templates_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(templates)

    print(f"Wrote {len(templates)} templates to series_templates.csv")

    # Print summary
    scopes = {}
    for t in templates:
        scopes[t["scope"]] = scopes.get(t["scope"], 0) + 1
    print(f"Scope distribution: {scopes}")

    # Verify: count expected expanded rows
    n_country = scopes.get("country", 0)
    n_ea = scopes.get("ea", 0)
    expected = n_country * 19 + n_ea
    print(f"Expected expanded rows: {n_country} × 19 + {n_ea} = {expected}")


def convert_row(row: dict) -> dict:
    """Convert a single registry row to a template row."""
    series_id = row["series_id"]
    source = row["source"]
    country = row["country_iso2"]

    # Determine template_id and scope
    template_id, scope = extract_template_id(series_id, source, country)

    # Convert key and filters_json to templates with placeholders
    key_template = templatize_key(row.get("key", ""), source, country, scope)
    filters_template = templatize_filters(
        row.get("filters_json", ""), source, country, scope
    )

    return {
        "template_id": template_id,
        "source": source,
        "category": row["category"],
        "category_name": row["category_name"],
        "variable_name": row["variable_name"],
        "dataset": row["dataset"],
        "key_template": key_template,
        "filters_json_template": filters_template,
        "scope": scope,
        "unit_measure_filter": row.get("unit_measure_filter", ""),
        "frequency": row.get("frequency", ""),
        "start_period": row.get("start_period", ""),
    }


def extract_template_id(series_id: str, source: str, country: str) -> tuple[str, str]:
    """Extract template_id and scope from series_id.

    Returns:
        (template_id, scope) tuple.
    """
    # BIS special case: BIS_NEER_001_DE → NEER_001, country
    if source == "bis":
        # Strip BIS_ prefix and _XX country suffix
        match = re.match(r"BIS_(.+)_([A-Z]{2})$", series_id)
        if match:
            return match.group(1), "country"

    # EA-level ECB series: EA_XXX → XXX, ea
    if country == "U2" and series_id.startswith("EA_"):
        return series_id[3:], "ea"

    # Country-level: DE_XXX → XXX, country
    if series_id.startswith(f"{country}_"):
        return series_id[len(country) + 1 :], "country"

    # Fallback
    msg = f"Cannot extract template_id from {series_id} (source={source}, country={country})"
    raise ValueError(msg)


def templatize_key(key: str, source: str, country: str, scope: str) -> str:
    """Replace country codes in SDMX key with placeholders."""
    if not key or key.strip() == "":
        return ""

    if scope == "ea":
        # EA-level: keep key as-is (no country expansion needed)
        return key

    if source == "ecb":
        # ECB keys have country code as dot-separated segment: M.DE.N... → M.{COUNTRY_ISO2}.N...
        # Replace .DE. with .{COUNTRY_ISO2}. (only first occurrence after M.)
        return key.replace(f".{country}.", ".{COUNTRY_ISO2}.", 1)

    if source == "oecd":
        # OECD keys start with ISO3 code: DEU.M... → {COUNTRY_OECD}.M...
        iso3_map = {"DE": "DEU"}  # Only DE in Phase A
        iso3 = iso3_map.get(country, country)
        if key.startswith(f"{iso3}."):
            return "{COUNTRY_OECD}." + key[len(iso3) + 1 :]
        return key

    if source == "bis":
        # BIS key ends with country: M.N.B.DE → M.N.B.{COUNTRY_ISO2}
        if key.endswith(f".{country}"):
            return key[: -len(country)] + "{COUNTRY_ISO2}"
        return key

    return key


def templatize_filters(
    filters_json: str, source: str, country: str, scope: str
) -> str:
    """Replace country codes in Eurostat filters_json with placeholders."""
    if not filters_json or filters_json.strip() == "":
        return ""

    if scope == "ea":
        return filters_json

    if source != "eurostat":
        return filters_json

    # Eurostat: Replace "geo": "DE" with "geo": "{COUNTRY_EUROSTAT}"
    # The JSON uses double-escaped quotes in CSV: ""geo"": ""DE""
    # We need to handle the raw CSV value
    result = filters_json.replace(f'"geo": "{country}"', '"geo": "{COUNTRY_EUROSTAT}"')
    return result


if __name__ == "__main__":
    main()
