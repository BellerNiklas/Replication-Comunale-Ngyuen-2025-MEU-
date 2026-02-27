"""Unit tests for temporal coverage filtering functions."""

import pandas as pd

from meu_replication.data_management.task_filter_temporal_coverage import (
    _build_expected_months,
    filter_by_temporal_coverage,
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
    panel = _make_panel({
        "A": ["2020-01", "2020-02", "2020-03"],  # Full
        "B": ["2020-01", "2020-03"],  # Missing 2020-02
    })

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    assert set(filtered["series_id"].unique()) == {"A"}
    assert len(drop_info) == 1
    assert drop_info.iloc[0]["series_id"] == "B"
    assert drop_info.iloc[0]["n_missing"] == 1


def test_filter_all_series_pass():
    """When all series have full coverage, drop_info is empty."""
    panel = _make_panel({
        "A": ["2020-01", "2020-02", "2020-03"],
        "B": ["2020-01", "2020-02", "2020-03"],
    })

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    n_expected = 2
    assert filtered["series_id"].nunique() == n_expected
    assert drop_info.empty


def test_filter_all_series_dropped():
    """When no series has full coverage, filtered panel is empty."""
    panel = _make_panel({
        "A": ["2020-01"],
        "B": ["2020-02", "2020-03"],
    })

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    n_expected_drops = 2
    assert filtered.empty
    assert len(drop_info) == n_expected_drops


def test_filter_restricts_to_sample_period():
    """Rows outside the sample period are excluded from filtered output."""
    panel = _make_panel({
        "A": ["2019-12", "2020-01", "2020-02", "2020-03", "2020-04"],
    })

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

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
    panel = _make_panel({
        "A": ["2020-01"],  # Missing 2020-02, 2020-03
    })

    _, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    n_expected_missing = 2
    assert len(drop_info) == 1
    assert drop_info.iloc[0]["n_months"] == 1
    assert drop_info.iloc[0]["n_missing"] == n_expected_missing


def test_filter_empty_input():
    """Empty input produces empty outputs without crashing."""
    panel = _make_panel({})

    filtered, drop_info = filter_by_temporal_coverage(
        panel, SAMPLE_START, SAMPLE_END
    )

    assert filtered.empty
    assert drop_info.empty


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
    drop_info = pd.DataFrame({
        "series_id": ["DE_001"],
        "country_iso2": ["DE"],
        "variable_name": ["test"],
        "category_name": ["test_cat"],
        "n_months": [200],
        "n_missing": [40],
    })

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
