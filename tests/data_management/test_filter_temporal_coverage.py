"""Unit tests for temporal coverage filtering functions."""

import pandas as pd

from meu_replication.data_management.task_filter_temporal_coverage import (
    _build_expected_months,
    filter_by_temporal_coverage,
    generate_comparative_report,
    generate_filter_report,
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
    """Series missing 3 months dropped when allowed_missing=2."""
    panel = _make_panel(
        {
            "A": ["2020-01", "2020-02", "2020-03"],  # Full
            "B": ["2020-01"],  # Missing 2 months
        }
    )

    # allowed_missing=1 should still drop B (missing 2 > 1)
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
# Tests for generate_filter_report
# ---------------------------------------------------------------------------


def _empty_drop_info():
    """Create an empty drop_info DataFrame with correct schema."""
    return pd.DataFrame(
        columns=[
            "series_id",
            "country_iso2",
            "variable_name",
            "category_name",
            "n_months",
            "n_missing",
        ]
    )


def test_report_contains_summary_counts():
    """Report includes total, kept, and dropped counts."""
    report = generate_filter_report(
        n_total=100,
        n_kept=90,
        n_dropped=10,
        country_totals={"DE": 100},
        drop_info=_empty_drop_info(),
        sample_start="2003-01",
        sample_end="2022-12",
        expected_months=240,
    )

    assert "100" in report
    assert "90" in report
    assert "10" in report
    assert "2003-01" in report
    assert "2022-12" in report


def test_report_contains_country_table():
    """Report shows per-country survival rows."""
    drop_info = pd.DataFrame(
        {
            "series_id": ["DE_001"],
            "country_iso2": ["DE"],
            "variable_name": ["test"],
            "category_name": ["test_cat"],
            "n_months": [200],
            "n_missing": [40],
        }
    )

    report = generate_filter_report(
        n_total=10,
        n_kept=9,
        n_dropped=1,
        country_totals={"DE": 10},
        drop_info=drop_info,
        sample_start="2003-01",
        sample_end="2022-12",
        expected_months=240,
    )

    assert "| DE |" in report
    assert "90.0" in report  # 9/10 = 90%


def test_report_empty_drops():
    """Report renders correctly when no series are dropped."""
    report = generate_filter_report(
        n_total=50,
        n_kept=50,
        n_dropped=0,
        country_totals={"DE": 50},
        drop_info=_empty_drop_info(),
        sample_start="2020-01",
        sample_end="2020-03",
        expected_months=3,
    )

    assert "Dropped" in report
    assert "0" in report


def test_report_contains_variant_label():
    """Report includes variant label when provided."""
    report = generate_filter_report(
        n_total=50,
        n_kept=50,
        n_dropped=0,
        country_totals={"DE": 50},
        drop_info=_empty_drop_info(),
        sample_start="2020-01",
        sample_end="2020-03",
        expected_months=3,
        variant_label="2022 Strict",
    )

    assert "2022 Strict" in report


def test_report_contains_allowed_missing():
    """Report includes allowed missing count."""
    n_allowed = 4
    report = generate_filter_report(
        n_total=50,
        n_kept=50,
        n_dropped=0,
        country_totals={"DE": 50},
        drop_info=_empty_drop_info(),
        sample_start="2020-01",
        sample_end="2020-03",
        expected_months=3,
        allowed_missing=n_allowed,
    )

    assert f"{n_allowed} months" in report


# ---------------------------------------------------------------------------
# Tests for generate_comparative_report
# ---------------------------------------------------------------------------


def _make_variant_result(label, start, end, n_kept, n_dropped, allowed_missing=0):
    """Build a minimal variant result dict for testing."""
    return {
        "label": label,
        "start": start,
        "end": end,
        "allowed_missing": allowed_missing,
        "n_total": n_kept + n_dropped,
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "country_totals": {"DE": n_kept + n_dropped},
        "drop_info": _empty_drop_info(),
    }


def test_comparative_report_contains_all_variants():
    """Comparative report shows all 4 variant labels."""
    variants = [
        _make_variant_result("2022_strict", "2003-01", "2022-12", 90, 10),
        _make_variant_result("2022_cov98", "2003-01", "2022-12", 95, 5, 4),
        _make_variant_result("2021_strict", "2003-01", "2021-12", 92, 8),
        _make_variant_result("2021_cov98", "2003-01", "2021-12", 97, 3, 4),
    ]

    report = generate_comparative_report(variants)

    for v in variants:
        assert v["label"] in report


def test_comparative_report_car_registration_note():
    """Comparative report includes note about CARS_002-004."""
    variants = [
        _make_variant_result("2022_strict", "2003-01", "2022-12", 90, 10),
    ]

    report = generate_comparative_report(variants)

    assert "CARS_002" in report
    assert "2021-12" in report
