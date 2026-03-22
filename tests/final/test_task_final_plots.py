from __future__ import annotations

from pathlib import Path

import pandas as pd

from meu_replication.final.task_final_template import (
    run_appendix_availability_comparison_plot,
    run_availability_overview_plot,
    run_country_availability_plot,
    run_country_vs_ea_plot,
    run_ea_meu_h3_plot,
)


def _make_long_panel(
    dates: list[str],
    series_specs: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates, start=1):
        for series_id, country_iso2 in series_specs:
            rows.append(
                {
                    "date": date,
                    "value": float(date_idx),
                    "series_id": series_id,
                    "country_iso2": country_iso2,
                    "variable_name": series_id,
                    "category": 1,
                    "category_name": "Test",
                    "source": "synthetic",
                    "transformationcode": 1,
                }
            )
    return pd.DataFrame(rows)


def _write_availability_inputs(tmp_path: Path) -> dict[str, Path]:
    transformed = pd.concat(
        [
            _make_long_panel(
                ["2020-01", "2020-02", "2020-03"],
                [
                    ("AT_A", "AT"),
                    ("AT_B", "AT"),
                    ("BE_A", "BE"),
                    ("U2_FX_001", "U2"),
                ],
            ),
            _make_long_panel(["2019-12"], [("OLD_ONLY", "AT")]),
        ],
        ignore_index=True,
    )
    transformed_path = tmp_path / "transformed_panel.parquet"
    transformed.to_parquet(transformed_path, index=False)

    depends_on: dict[str, Path] = {"transformed_panel": transformed_path}
    for year in (2021, 2022, 2025):
        strict = _make_long_panel(
            ["2020-02", "2020-03"],
            [("AT_A", "AT"), ("BE_A", "BE"), ("U2_FX_001", "U2")],
        )
        clean = _make_long_panel(
            ["2020-02", "2020-03"],
            [("AT_A", "AT"), ("U2_FX_001", "U2")],
        )
        strict_path = tmp_path / f"strict_{year}.parquet"
        clean_path = tmp_path / f"clean_{year}.parquet"
        strict.to_parquet(strict_path, index=False)
        clean.to_parquet(clean_path, index=False)
        depends_on[f"strict_{year}"] = strict_path
        depends_on[f"clean_{year}"] = clean_path

    return depends_on


def _write_meu_inputs(tmp_path: Path) -> dict[str, Path]:
    depends_on: dict[str, Path] = {}
    for year, dates in (
        (2021, ["2020-01", "2020-02"]),
        (2022, ["2021-01", "2021-02"]),
        (2025, ["2024-01", "2024-02"]),
    ):
        ea = pd.DataFrame(
            {
                "date": [dates[0], dates[0], dates[1], dates[1]],
                "horizon": [1, 3, 1, 3],
                "meu": [0.20, 0.35, 0.25, 0.40],
            }
        )
        country = pd.DataFrame(
            {
                "country_iso2": [
                    "AT",
                    "BE",
                    "AT",
                    "BE",
                    "AT",
                    "BE",
                    "AT",
                    "BE",
                ],
                "date": [
                    dates[0],
                    dates[0],
                    dates[0],
                    dates[0],
                    dates[1],
                    dates[1],
                    dates[1],
                    dates[1],
                ],
                "horizon": [1, 1, 3, 3, 1, 1, 3, 3],
                "meu": [0.30, 0.32, 0.42, 0.38, 0.31, 0.33, 0.44, 0.39],
            }
        )
        ea_path = tmp_path / f"ea_{year}.parquet"
        country_path = tmp_path / f"country_{year}.parquet"
        ea.to_parquet(ea_path, index=False)
        country.to_parquet(country_path, index=False)
        depends_on[f"ea_{year}"] = ea_path
        depends_on[f"country_{year}"] = country_path

    return depends_on


def test_final_plot_tasks_write_expected_html_and_png_outputs(tmp_path: Path):
    availability_inputs = _write_availability_inputs(tmp_path)
    meu_inputs = _write_meu_inputs(tmp_path)

    availability_overview_outputs = {
        "html": tmp_path / "plots" / "availability" / "availability_overview.html",
        "png": tmp_path / "plots" / "availability" / "availability_overview.png",
    }
    run_availability_overview_plot(
        depends_on=availability_inputs,
        produces=availability_overview_outputs,
    )

    appendix_comparison_outputs = {
        "html": tmp_path / "plots" / "availability" / "availability_vs_appendix_2021.html",
        "png": tmp_path / "plots" / "availability" / "availability_vs_appendix_2021.png",
    }
    run_appendix_availability_comparison_plot(
        depends_on={"strict_2021": availability_inputs["strict_2021"]},
        produces=appendix_comparison_outputs,
    )

    country_availability_outputs = {
        "html": tmp_path / "plots" / "availability" / "availability_by_country_2021.html",
        "png": tmp_path / "plots" / "availability" / "availability_by_country_2021.png",
    }
    run_country_availability_plot(
        depends_on={
            "transformed_panel": availability_inputs["transformed_panel"],
            "strict": availability_inputs["strict_2021"],
            "clean": availability_inputs["clean_2021"],
        },
        produces=country_availability_outputs,
        year=2021,
    )

    ea_meu_outputs = {
        "html": tmp_path / "plots" / "meu" / "ea_meu_h3_by_panel.html",
        "png": tmp_path / "plots" / "meu" / "ea_meu_h3_by_panel.png",
    }
    run_ea_meu_h3_plot(
        depends_on={
            "ea_2021": meu_inputs["ea_2021"],
            "ea_2022": meu_inputs["ea_2022"],
            "ea_2025": meu_inputs["ea_2025"],
        },
        produces=ea_meu_outputs,
    )

    country_vs_ea_outputs = {
        "html": tmp_path / "plots" / "meu" / "country_vs_ea_h3_2021.html",
        "png": tmp_path / "plots" / "meu" / "country_vs_ea_h3_2021.png",
    }
    run_country_vs_ea_plot(
        depends_on={
            "ea_meu": meu_inputs["ea_2021"],
            "country_meu": meu_inputs["country_2021"],
        },
        produces=country_vs_ea_outputs,
        year=2021,
    )

    expected_files = [
        *availability_overview_outputs.values(),
        *appendix_comparison_outputs.values(),
        *country_availability_outputs.values(),
        *ea_meu_outputs.values(),
        *country_vs_ea_outputs.values(),
    ]
    for path in expected_files:
        assert path.exists()
        assert path.stat().st_size > 0
