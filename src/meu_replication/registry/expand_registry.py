"""Expand series templates into the committed multi-country registry.

Usage:
    pixi run python -m meu_replication.registry.expand_registry
"""

import pandas as pd

from meu_replication.config import SRC, load_countries

TEMPLATES_PATH = SRC / "registry" / "series_templates.csv"
REGISTRY_PATH = SRC / "registry" / "series_registry.csv"


def expand_registry(
    templates: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Expand template rows into the full series registry.

    Args:
        templates: DataFrame with template definitions (148 rows).
        countries: DataFrame with country code mappings (20 rows).

    Returns:
        Expanded registry DataFrame with one row per series-country combination.
        Schema matches series_registry.csv:
            series_id, source, category, category_name, country_iso2,
            variable_name, dataset, key, filters_json, unit_measure_filter,
            frequency, start_period, transformationcode
    """
    ea_members = countries[countries.ea_member == True]  # noqa: E712

    expanded_rows = []
    for _, template in templates.iterrows():
        scope = template["scope"]

        if scope == "ea":
            row = _expand_template_for_ea(template)
            expanded_rows.append(row)

        elif scope == "country":
            for _, country_row in ea_members.iterrows():
                row = _expand_template_for_country(template, country_row)
                expanded_rows.append(row)

        else:
            msg = f"Unknown scope '{scope}' in template '{template['template_id']}'"
            raise ValueError(msg)

    expanded_registry = pd.DataFrame(expanded_rows)
    return expanded_registry


def _expand_template_for_ea(template: pd.Series) -> dict:
    """Expand EA-aggregate template to a single U2 series."""
    key_raw = template.get("key_template", "")
    filters_raw = template.get("filters_json_template", "")

    row = {
        "series_id": f"U2_{template['template_id']}",
        "source": template["source"],
        "category": template["category"],
        "category_name": template["category_name"],
        "country_iso2": "U2",
        "variable_name": template["variable_name"],
        "dataset": template["dataset"],
        "key": _nan_to_empty(key_raw),
        "filters_json": _nan_to_empty(filters_raw),
        "unit_measure_filter": _nan_to_empty(template.get("unit_measure_filter", "")),
        "frequency": template.get("frequency", "M"),
        "start_period": template.get("start_period", "2003-01"),
        "transformationcode": int(template["transformationcode"]),
    }
    return row


def _expand_template_for_country(
    template: pd.Series,
    country: pd.Series,
) -> dict:
    """Expand country-level template for one specific country."""
    key_raw = _nan_to_empty(template.get("key_template", ""))
    filters_raw = _nan_to_empty(template.get("filters_json_template", ""))

    key = _expand_placeholders(key_raw, country)
    filters_json = _expand_placeholders(filters_raw, country)

    row = {
        "series_id": f"{country['country_iso2']}_{template['template_id']}",
        "source": template["source"],
        "category": template["category"],
        "category_name": template["category_name"],
        "country_iso2": country["country_iso2"],
        "variable_name": template["variable_name"],
        "dataset": template["dataset"],
        "key": key,
        "filters_json": filters_json,
        "unit_measure_filter": _nan_to_empty(template.get("unit_measure_filter", "")),
        "frequency": template.get("frequency", "M"),
        "start_period": template.get("start_period", "2003-01"),
        "transformationcode": int(template["transformationcode"]),
    }
    return row


def _expand_placeholders(template_str: str, country: pd.Series) -> str:
    """Replace country placeholders in a registry template string.

    Placeholders:
        {COUNTRY_ISO2}     → "DE", "FR", etc. (used for most sources)
        {COUNTRY_ISO3}     → "DEU", "FRA", etc. (not used directly, see OECD)
        {COUNTRY_EUROSTAT} → Handles Greece: "EL" for Eurostat, "GR" elsewhere
        {COUNTRY_OECD}     → ISO3 for OECD API keys
        {COUNTRY_BIS}      → BIS country code (same as ISO2 for EA countries)
    """
    if template_str == "":
        return ""

    replacements = {
        "{COUNTRY_ISO2}": str(country["country_iso2"]),
        "{COUNTRY_ISO3}": str(country["country_iso3"]),
        "{COUNTRY_EUROSTAT}": str(country["source_eurostat"]),
        "{COUNTRY_OECD}": str(country["source_oecd"]),
        "{COUNTRY_BIS}": str(country["source_bis"]),
    }

    result = template_str
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def _nan_to_empty(value) -> str:
    """Convert NaN or None to empty string."""
    if pd.isna(value):
        return ""
    return str(value)


def load_templates() -> pd.DataFrame:
    """Load series templates from CSV."""
    return pd.read_csv(TEMPLATES_PATH)


def main() -> None:
    """Generate series_registry.csv from templates + countries (CLI entry point)."""
    templates = load_templates()
    countries = load_countries()

    print(f"Loaded {len(templates)} templates and {len(countries)} countries")

    registry = expand_registry(templates, countries)

    print(f"Expanded to {len(registry)} series")
    print(f"  country-scope templates: {len(templates[templates.scope == 'country'])}")
    print(f"  ea-scope templates: {len(templates[templates.scope == 'ea'])}")
    print(f"  unique countries: {registry.country_iso2.nunique()}")

    registry.to_csv(REGISTRY_PATH, index=False)
    print(f"Wrote {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
