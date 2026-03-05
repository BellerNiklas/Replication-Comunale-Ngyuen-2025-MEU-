import pandas as pd

from meu_replication.cleaning.high_correlation import (
    _correlated_pairs,
    _find_redundant_series,
    _pivot_to_wide,
    remove_high_correlation,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

SAMPLE_DATES = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"]


def _make_panel(
    series_values: dict[str, list[float]], country: str = "DE"
) -> pd.DataFrame:
    """Build a long-format panel for one country.

    Args:
        series_values: Maps series_id -> list of monthly values.
        country: Country code for all rows.

    Returns:
        DataFrame with the canonical clean panel schema.
    """
    rows = []
    for sid, vals in series_values.items():
        for date, val in zip(SAMPLE_DATES[: len(vals)], vals):
            rows.append(
                {
                    "date": date,
                    "value": val,
                    "series_id": sid,
                    "country_iso2": country,
                    "variable_name": f"var_{sid}",
                    "category": 1,
                    "category_name": "test_category",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for _pivot_to_wide
# ---------------------------------------------------------------------------


def test_pivot_to_wide_shape():
    panel = _make_panel({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    wide = _pivot_to_wide(panel)
    n_dates = 3
    n_series = 2
    assert wide.shape == (n_dates, n_series)
    assert set(wide.columns) == {"A", "B"}


# ---------------------------------------------------------------------------
# Tests for _correlated_pairs
# ---------------------------------------------------------------------------


def test_correlated_pairs_finds_perfect_correlation():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    wide = _pivot_to_wide(panel)
    corr = wide.corr()
    pairs = _correlated_pairs(corr, threshold=0.95)

    pair_ids = [(a, b) for a, b, _ in pairs]
    assert ("A", "B") in pair_ids
    assert all(not (a == "C" or b == "C") for a, b, _ in pairs)


def test_correlated_pairs_sorted_descending():
    corr = pd.DataFrame(
        [[1.0, 0.99, 0.96], [0.99, 1.0, 0.97], [0.96, 0.97, 1.0]],
        index=pd.Index(("A", "B", "C")),
        columns=pd.Index(("A", "B", "C")),
    )
    pairs = _correlated_pairs(corr, threshold=0.95)
    correlations = [c for _, _, c in pairs]
    assert correlations == sorted(correlations, reverse=True)


# ---------------------------------------------------------------------------
# Tests for _find_redundant_series
# ---------------------------------------------------------------------------


def test_alphabetical_tiebreak():
    panel = _make_panel(
        {
            "AT_A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "AT_B": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    dropped, records = _find_redundant_series(panel, 0.95, "AT")

    assert "AT_B" in dropped
    assert "AT_A" not in dropped
    assert len(records) == 1
    assert records[0]["kept_series"] == "AT_A"
    assert records[0]["dropped_series"] == "AT_B"


# ---------------------------------------------------------------------------
# Tests for remove_high_correlation (integration)
# ---------------------------------------------------------------------------


def test_drops_correlated_keeps_uncorrelated():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    filtered, drop_info = remove_high_correlation(panel, threshold=0.95)

    remaining = set(filtered["series_id"].unique())
    assert "A" in remaining
    assert "C" in remaining
    assert "B" not in remaining
    assert len(drop_info) == 1


def test_per_country_independence():
    at = _make_panel(
        {"AT_X": [1.0, 2.0, 3.0], "AT_Y": [10.0, 20.0, 30.0]}, country="AT"
    )
    de = _make_panel(
        {"DE_X": [1.0, 2.0, 3.0], "DE_Y": [10.0, 20.0, 30.0]}, country="DE"
    )
    panel = pd.concat([at, de], ignore_index=True)

    filtered, drop_info = remove_high_correlation(panel, threshold=0.95)

    assert set(drop_info["dropped_series"]) == {"AT_Y", "DE_Y"}
    assert set(drop_info["country_iso2"]) == {"AT", "DE"}


def test_no_drop_below_threshold():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    unreachable_threshold = 1.01
    filtered, drop_info = remove_high_correlation(
        panel, threshold=unreachable_threshold
    )

    assert len(filtered) == len(panel)
    assert len(drop_info) == 0


def test_schema_preserved():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    filtered, _ = remove_high_correlation(panel, threshold=0.95)

    assert list(filtered.columns) == list(panel.columns)


def test_empty_input():
    panel = _make_panel({})
    filtered, drop_info = remove_high_correlation(panel, threshold=0.95)

    assert filtered.empty
    assert drop_info.empty


def test_negative_correlation_detected():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [-1.0, -2.0, -3.0, -4.0, -5.0],
        }
    )
    filtered, drop_info = remove_high_correlation(panel, threshold=0.95)

    assert len(drop_info) == 1
    assert drop_info.iloc[0]["dropped_series"] == "B"
    remaining = set(filtered["series_id"].unique())
    assert remaining == {"A"}


def test_greedy_chain():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [3.0, 6.0, 9.0, 12.0, 15.0],
        }
    )
    filtered, drop_info = remove_high_correlation(panel, threshold=0.95)

    remaining = set(filtered["series_id"].unique())
    assert "A" in remaining
    assert "B" not in remaining
    assert "C" not in remaining


def test_no_cascading_drops():
    panel = _make_panel(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    _, drop_info = remove_high_correlation(panel, threshold=0.95)

    kept_set = set(drop_info["kept_series"])
    dropped_set = set(drop_info["dropped_series"])
    assert not kept_set & dropped_set
