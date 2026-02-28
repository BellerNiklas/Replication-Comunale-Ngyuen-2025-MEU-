"""Unit tests for temporal coverage filtering functions."""

import pandas as pd

from meu_replication.cleaning.temporal_coverage import (
    _build_expected_months,
    filter_by_temporal_coverage,
    run_all_filter_variants,
)

# Small 3-month sample period for compact tests
SAMPLE_START = "2020-01"
SAMPLE_END = "2020-03"


def _make_panel(
    series_months: dict[str, list[str]], country: str = "DE"
) -> pd.DataFrame:
    """Build a tiny panel for testing.

    Args:
        series_months: Maps series_id -> list of "YYYY-MM" date strings.
        country: Country code for all rows.

    Returns:
        DataFrame with the canonical clean panel schema.
    """
    rows = [
        {
            "date": m,
            "value": 100.0,
            "series_id": sid,
            "country_iso2": country,
            "variable_name": f"var_{sid}",
            "category": 1,
            "category_name": "test_category",
            "source": "test",
        }
        for sid, months in series_months.items()
        for m in months
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for _build_expected_months
# ---------------------------------------------------------------------------


def test_expected_months_count():
    """Sanity check: 2003-01 to 2022-12 yields exactly 240 months."""
    expected_count = 20 * 12  # 20 years * 12 months
    months = _build_expected_months("2003-01", "2022-12")
    assert len(months) == expected_count


def test_expected_months_format():
    """All month strings are zero-padded YYYY-MM."""
    n_months_in_year = 12
    months = _build_expected_months("2020-01", "2020-12")
    assert len(months) == n_months_in_year
    for m in months:
        year, month = m.split("-")
        assert len(year) == 4  # noqa: PLR2004
        assert len(month) == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Tests for filter_by_temporal_coverage
# ---------------------------------------------------------------------------


def test_filter_keeps_full_coverage_series():
    """Series with all 3 months kept, incomplete series dropped."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],  # Full
            "B": ["2020-01", "2020-03"],  # Missing 2020-02
        }
    )

    filtered, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    assert set(filtered["series_id"].unique()) == {"A"}
    assert len(drop_info) == 1
    assert drop_info.iloc[0]["series_id"] == "B"
    assert drop_info.iloc[0]["n_missing"] == 1


def test_filter_all_series_pass():
    """When all series have full coverage, drop_info is empty."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],
            "B": ["2020-01", "2020-02", "2020-03"],
        }
    )

    filtered, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    n_expected = 2
    assert filtered["series_id"].nunique() == n_expected
    assert drop_info.empty


def test_filter_all_series_dropped():
    """When no series has full coverage, filtered panel is empty."""
    panel = _make_panel(
        {
            "A": ["2020-01"],
            "B": ["2020-02", "2020-03"],
        }
    )

    filtered, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    n_expected_drops = 2
    assert filtered.empty
    assert len(drop_info) == n_expected_drops


def test_filter_restricts_to_sample_period():
    """Rows outside the sample period are excluded from filtered output."""
    panel = _make_panel(
        {
            "A": ["2019-12", "2020-01", "2020-02", "2020-03", "2020-04"],
        }
    )

    filtered, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    n_sample_months = 3
    assert drop_info.empty
    assert set(filtered["date"].unique()) == {
        "2020-01",
        "2020-02",
        "2020-03",
    }
    assert len(filtered) == n_sample_months


def test_filter_drop_info_reports_correct_missing():
    """Drop info correctly reports months present and missing."""
    panel = _make_panel(
        {
            "A": ["2020-01"],  # Missing 2020-02, 2020-03
        }
    )

    _, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    n_expected_missing = 2
    assert len(drop_info) == 1
    assert drop_info.iloc[0]["n_months"] == 1
    assert drop_info.iloc[0]["n_missing"] == n_expected_missing


def test_filter_empty_input():
    """Empty input produces empty outputs without crashing."""
    panel = _make_panel({})

    filtered, drop_info = filter_by_temporal_coverage(panel, SAMPLE_START, SAMPLE_END)

    assert filtered.empty
    assert drop_info.empty


# ---------------------------------------------------------------------------
# Tests for allowed_missing parameter
# ---------------------------------------------------------------------------


def test_filter_with_allowed_missing_keeps_near_complete():
    """Series missing 1 month kept when allowed_missing=1, dropped when 0."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02"],  # Missing 2020-03
        }
    )

    # Strict: should drop
    _, drop_strict = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END, allowed_missing=0
    )
    assert len(drop_strict) == 1

    # Relaxed: should keep
    filtered_relaxed, drop_relaxed = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END, allowed_missing=1
    )
    assert drop_relaxed.empty
    assert set(filtered_relaxed["series_id"].unique()) == {"A"}


def test_filter_with_allowed_missing_still_drops_beyond_threshold():
    """Series missing 2 months dropped when allowed_missing=1."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],  # Full
            "B": ["2020-01"],  # Missing 2 months
        }
    )

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END, allowed_missing=1
    )
    assert set(filtered["series_id"].unique()) == {"A"}
    assert len(drop_info) == 1
    assert drop_info.iloc[0]["series_id"] == "B"


def test_filter_allowed_missing_zero_matches_strict():
    """Verify allowed_missing=0 behaves identically to original strict."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],
            "B": ["2020-01", "2020-03"],
            "C": ["2020-01"],
        }
    )

    filtered_default, drop_default = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )
    filtered_explicit, drop_explicit = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END, allowed_missing=0
    )

    assert set(filtered_default["series_id"]) == set(filtered_explicit["series_id"])
    assert set(drop_default["series_id"]) == set(drop_explicit["series_id"])


# ---------------------------------------------------------------------------
# Tests for run_all_filter_variants
# ---------------------------------------------------------------------------


def test_run_all_filter_variants_returns_panels():
    """run_all_filter_variants returns dict of filtered DataFrames."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],
            "B": ["2020-01", "2020-03"],
        }
    )
    variants = [
        {
            "label": "strict",
            "key": "panel_strict",
            "start": SAMPLE_START,
            "end": SAMPLE_END,
            "allowed_missing": 0,
        },
        {
            "label": "relaxed",
            "key": "panel_relaxed",
            "start": SAMPLE_START,
            "end": SAMPLE_END,
            "allowed_missing": 1,
        },
    ]

    panels = run_all_filter_variants(panel, variants)

    assert set(panels) == {"panel_strict", "panel_relaxed"}
    # Strict keeps only A, relaxed keeps both
    assert set(panels["panel_strict"]["series_id"].unique()) == {"A"}
    assert set(panels["panel_relaxed"]["series_id"].unique()) == {"A", "B"}
