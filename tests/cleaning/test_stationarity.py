import numpy as np
import pandas as pd
import pytest

from meu_replication.cleaning.stationarity import (
    _transform_series,
    apply_stationarity_transforms,
    build_transform_map,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DATES = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"]


def _make_panel(
    series_values: dict[str, list[float]], country: str = "DE"
) -> pd.DataFrame:
    rows = []
    for sid, vals in series_values.items():
        for date, val in zip(DATES[: len(vals)], vals):
            rows.append(
                {
                    "date": date,
                    "value": val,
                    "series_id": sid,
                    "country_iso2": country,
                    "variable_name": f"var_{sid}",
                    "category": 1,
                    "category_name": "test",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def _make_transform_map(mapping: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"series_id": k, "transformationcode": v} for k, v in mapping.items()]
    )


# ---------------------------------------------------------------------------
# Tests for _transform_series
# ---------------------------------------------------------------------------


def test_code_1_identity():
    values = pd.Series([10.0, 20.0, 30.0])
    result = _transform_series(values, 1)
    pd.testing.assert_series_equal(result, values)


def test_code_2_first_diff():
    values = pd.Series([10.0, 12.0, 15.0])
    result = _transform_series(values, 2)
    expected = pd.Series([np.nan, 2.0, 3.0])
    pd.testing.assert_series_equal(result, expected)


def test_code_5_log_diff():
    values = pd.Series([100.0, 110.0])
    result = _transform_series(values, 5)
    expected_val = np.log(110.0) - np.log(100.0)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(expected_val)


def test_code_5_nonpositive_produces_nan():
    values = pd.Series([100.0, 0.0, 50.0])
    result = _transform_series(values, 5)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])


def test_unsupported_code_raises():
    values = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported transformation code: 3"):
        _transform_series(values, 3)


# ---------------------------------------------------------------------------
# Tests for build_transform_map
# ---------------------------------------------------------------------------


def test_build_transform_map():
    registry = pd.DataFrame(
        {
            "series_id": ["A", "B", "C"],
            "transformationcode": [1, 2, 5],
            "source": ["test"] * 3,
        }
    )
    result = build_transform_map(registry)
    assert list(result.columns) == ["series_id", "transformationcode"]
    assert len(result) == 3
    assert set(result["transformationcode"]) == {1, 2, 5}


def test_build_transform_map_bad_code_raises():
    registry = pd.DataFrame(
        {"series_id": ["A"], "transformationcode": [99]}
    )
    with pytest.raises(ValueError, match="Unsupported transformation codes"):
        build_transform_map(registry)


def test_build_transform_map_missing_column_raises():
    registry = pd.DataFrame({"series_id": ["A"]})
    with pytest.raises(ValueError, match="missing columns"):
        build_transform_map(registry)


# ---------------------------------------------------------------------------
# Tests for apply_stationarity_transforms
# ---------------------------------------------------------------------------


def test_apply_transforms_code_1_keeps_all_rows():
    panel = _make_panel({"A": [10.0, 20.0, 30.0]})
    tmap = _make_transform_map({"A": 1})
    result = apply_stationarity_transforms(panel, tmap)
    assert result[result.series_id == "A"]["value"].tolist() == [10.0, 20.0, 30.0]


def test_apply_transforms_code_2_computes_diff():
    panel = _make_panel({"B": [5.0, 8.0, 12.0]})
    tmap = _make_transform_map({"B": 2})
    result = apply_stationarity_transforms(panel, tmap)
    assert result[result.series_id == "B"]["value"].tolist() == pytest.approx([3.0, 4.0])


def test_apply_transforms_code_5_computes_log_diff():
    panel = _make_panel({"C": [100.0, 200.0, 400.0]})
    tmap = _make_transform_map({"C": 5})
    result = apply_stationarity_transforms(panel, tmap)
    c_vals = result[result.series_id == "C"]["value"].tolist()
    assert len(c_vals) == 2
    assert c_vals[0] == pytest.approx(np.log(200.0) - np.log(100.0))


def test_nan_rows_dropped():
    panel = _make_panel({"A": [10.0, 20.0, 30.0]})
    tmap = _make_transform_map({"A": 2})
    result = apply_stationarity_transforms(panel, tmap)
    assert len(result) == 2
    assert result["date"].iloc[0] == "2020-02"


def test_output_has_transformationcode_column():
    panel = _make_panel({"A": [10.0, 20.0]})
    tmap = _make_transform_map({"A": 1})
    result = apply_stationarity_transforms(panel, tmap)
    assert "transformationcode" in result.columns
    assert result["transformationcode"].iloc[0] == 1


def test_empty_panel():
    panel = _make_panel({})
    tmap = _make_transform_map({})
    result = apply_stationarity_transforms(panel, tmap)
    assert result.empty
    assert "transformationcode" in result.columns


def test_schema_preserved():
    panel = _make_panel({"A": [10.0, 20.0, 30.0]})
    tmap = _make_transform_map({"A": 1})
    result = apply_stationarity_transforms(panel, tmap)
    expected_cols = [
        "date", "value", "series_id", "country_iso2",
        "variable_name", "category", "category_name",
        "source", "transformationcode",
    ]
    assert list(result.columns) == expected_cols


def test_code_1_no_rows_lost():
    panel = _make_panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    tmap = _make_transform_map({"A": 1})
    result = apply_stationarity_transforms(panel, tmap)
    assert len(result) == 5
