"""Integration test: verify the cleaning pipeline stages wire together correctly.

Runs clean → stationarity → temporal filter → correlation filter on a small
synthetic dataset (2 countries, 3 series each, 3 transformation codes).
No network calls. Catches schema mismatches between pipeline stages.
"""

import pandas as pd
import pytest

from meu_replication.cleaning.high_correlation import remove_high_correlation
from meu_replication.cleaning.stationarity import (
    apply_stationarity_transforms,
    build_transform_map,
)
from meu_replication.cleaning.temporal_coverage import filter_by_temporal_coverage
from meu_replication.data_management.task_build_clean import _clean_macro_panel

# --- Synthetic data ---

MONTHS = [f"2020-{m:02d}" for m in range(1, 13)]  # 12 months


def _build_raw_panel() -> pd.DataFrame:
    """Build a minimal raw panel: 2 countries x 3 series x 12 months."""
    rows = []
    for country in ("DE", "FR"):
        for sid, base in [("IP_001", 100.0), ("RATE_001", 2.5), ("SENT_001", 50.0)]:
            for i, month in enumerate(MONTHS):
                rows.append(
                    {
                        "date": month,
                        "value": base + i * 1.1 + (0.5 if country == "FR" else 0),
                        "series_id": f"{country}_{sid}",
                        "country_iso2": country,
                        "variable_name": f"var_{sid}",
                        "category": 1,
                        "category_name": "test",
                        "source": "test",
                    }
                )
    return pd.DataFrame(rows)


def _build_registry() -> pd.DataFrame:
    """Build a minimal registry matching the raw panel."""
    rows = []
    codes = {"IP_001": 5, "RATE_001": 2, "SENT_001": 1}
    for country in ("DE", "FR"):
        for sid, code in codes.items():
            rows.append({"series_id": f"{country}_{sid}", "transformationcode": code})
    return pd.DataFrame(rows)


# --- Integration test ---


@pytest.mark.integration
def test_pipeline_stages_wire_together():
    """Clean → stationarity → temporal filter → correlation filter.

    Verifies that each stage accepts the output of the previous stage
    without schema errors, and that the final output has the expected
    properties.
    """
    # Stage 0: Build raw panel (simulates raw snapshot concatenation)
    raw = _build_raw_panel()
    assert len(raw) == 2 * 3 * 12  # 72 rows

    # Stage 1: Clean (EPP functional rules — new DataFrame, no mutation)
    cleaned = _clean_macro_panel(raw)
    assert len(cleaned) == 72
    assert list(cleaned.columns) == [
        "date", "value", "series_id", "country_iso2",
        "variable_name", "category", "category_name", "source",
    ]

    # Stage 2: Stationarity transforms
    registry = _build_registry()
    tmap = build_transform_map(registry)
    transformed = apply_stationarity_transforms(cleaned, tmap)

    # Code 1 (SENT) keeps all 12 rows; codes 2,5 lose first → 11 each
    # Per country: 12 + 11 + 11 = 34 rows; × 2 countries = 68
    assert len(transformed) == 68
    assert "transformationcode" in transformed.columns
    assert set(transformed["transformationcode"]) == {1, 2, 5}

    # Stage 3: Temporal coverage filter (strict: 0 missing allowed)
    filtered, drop_info = filter_by_temporal_coverage(
        transformed,
        sample_start="2020-02",  # after differencing
        sample_end="2020-12",
        allowed_missing=0,
    )

    # All series should survive (all have 11 months in 2020-02..2020-12,
    # except code 1 which has 11 months in that window too)
    assert filtered["series_id"].nunique() == 6
    assert len(drop_info) == 0

    # Stage 4: High-correlation filter
    final, corr_drop_info = remove_high_correlation(filtered, threshold=0.95)

    # IP_001 (code 5, log-diff) and RATE_001 (code 2, diff) have very similar
    # linear trends so may be correlated. SENT_001 (code 1, identity) has a
    # different pattern. Exact drop count depends on data; just verify schema.
    assert set(final.columns) == set(filtered.columns)
    assert final["series_id"].nunique() <= 6
    assert final["series_id"].nunique() > 0

    # Verify drop_info schema
    expected_drop_cols = {"country_iso2", "dropped_series", "kept_series", "abs_correlation"}
    assert expected_drop_cols.issubset(set(corr_drop_info.columns))
