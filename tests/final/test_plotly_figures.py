from __future__ import annotations

import pandas as pd

from meu_replication.final.plotly_figures import (
    build_appendix_availability_comparison_data,
    build_appendix_availability_comparison_figure,
    build_availability_overview_data,
    build_country_availability_data,
    build_country_availability_figure,
    build_country_vs_ea_figure,
    build_ea_meu_h3_2025_website_figure,
    prepare_country_vs_ea_plot_data,
    prepare_ea_meu_plot_data,
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


def test_build_availability_overview_data_filters_preclean_to_panel_window():
    transformed = pd.concat(
        [
            _make_long_panel(
                ["2020-01", "2020-02", "2020-03"],
                [
                    ("AT_A", "AT"),
                    ("BE_A", "BE"),
                    ("U2_FX_001", "U2"),
                ],
            ),
            _make_long_panel(["2019-12"], [("OLD_ONLY", "AT")]),
        ],
        ignore_index=True,
    )
    strict_2021 = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("BE_A", "BE"), ("U2_FX_001", "U2")],
    )
    clean_2021 = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("U2_FX_001", "U2")],
    )

    data = build_availability_overview_data(
        transformed_panel=transformed,
        strict_panels={2021: strict_2021},
        clean_panels={2021: clean_2021},
    )

    counts = dict(zip(data["stage"], data["series_count"], strict=True))
    assert counts["Before cleaning"] == 3
    assert counts["After incomplete-drop"] == 3
    assert counts["Fully cleaned"] == 2


def test_build_country_availability_data_excludes_u2_and_respects_country_order():
    transformed = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("AT_B", "AT"), ("BE_A", "BE"), ("U2_FX_001", "U2")],
    )
    strict = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("BE_A", "BE"), ("U2_FX_001", "U2")],
    )
    clean = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("U2_FX_001", "U2")],
    )

    data = build_country_availability_data(
        transformed_panel=transformed,
        strict_panel=strict,
        clean_panel=clean,
        year=2021,
        country_order=("BE", "AT"),
        country_names={"AT": "Austria", "BE": "Belgium"},
    )

    assert "U2" not in data["country_iso2"].tolist()
    before = data.loc[data["stage"] == "Before cleaning"].set_index("country_iso2")
    strict_stage = data.loc[data["stage"] == "After incomplete-drop"].set_index(
        "country_iso2"
    )
    clean_stage = data.loc[data["stage"] == "Fully cleaned"].set_index("country_iso2")

    assert int(before.loc["AT", "series_count"]) == 2
    assert int(before.loc["BE", "series_count"]) == 1
    assert int(strict_stage.loc["AT", "series_count"]) == 1
    assert int(clean_stage.loc["BE", "series_count"]) == 0


def test_build_country_availability_data_defaults_to_preclean_order():
    transformed = _make_long_panel(
        ["2020-02", "2020-03"],
        [
            ("AT_A", "AT"),
            ("AT_B", "AT"),
            ("AT_C", "AT"),
            ("BE_A", "BE"),
            ("BE_B", "BE"),
            ("CY_A", "CY"),
            ("U2_FX_001", "U2"),
        ],
    )
    strict = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("BE_A", "BE"), ("CY_A", "CY"), ("U2_FX_001", "U2")],
    )
    clean = _make_long_panel(
        ["2020-02", "2020-03"],
        [("AT_A", "AT"), ("CY_A", "CY"), ("U2_FX_001", "U2")],
    )

    data = build_country_availability_data(
        transformed_panel=transformed,
        strict_panel=strict,
        clean_panel=clean,
        year=2021,
        country_names={"AT": "Austria", "BE": "Belgium", "CY": "Cyprus"},
    )

    ordered = (
        data.loc[:, ["country_iso2", "country_rank"]]
        .drop_duplicates()
        .sort_values("country_rank")["country_iso2"]
        .tolist()
    )
    assert ordered[:3] == ["AT", "BE", "CY"]


def test_prepare_ea_meu_plot_data_filters_to_horizon_three():
    ea_frames = {
        2021: pd.DataFrame(
            {
                "date": ["2020-01", "2020-01", "2020-02", "2020-02"],
                "horizon": [1, 3, 1, 3],
                "meu": [0.1, 0.3, 0.2, 0.4],
            }
        )
    }

    data = prepare_ea_meu_plot_data(ea_frames, horizon=3)

    assert data["panel_end_year"].tolist() == [2021, 2021]
    assert data["date"].tolist() == ["2020-01", "2020-02"]
    assert data["meu"].tolist() == [0.3, 0.4]


def test_ea_meu_h3_2025_website_figure_uses_only_the_2025_panel():
    expected_recession_windows = 3
    data = pd.DataFrame(
        {
            "panel_end_year": [2021, 2025, 2025, 2025],
            "date_ts": pd.to_datetime(
                ["2020-01-01", "2003-01-01", "2020-04-01", "2025-12-01"]
            ),
            "meu": [9.0, 0.3, 1.2, 0.5],
        }
    )

    fig = build_ea_meu_h3_2025_website_figure(data)

    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.3, 1.2, 0.5]
    assert len(fig.layout.shapes) == expected_recession_windows
    assert "2003-2025" in fig.layout.title.text
    assert any("Peak: Apr 2020" in item.text for item in fig.layout.annotations)


def test_prepare_country_vs_ea_plot_data_merges_and_preserves_order():
    ea_meu = pd.DataFrame(
        {
            "date": ["2020-01", "2020-01", "2020-02", "2020-02"],
            "horizon": [1, 3, 1, 3],
            "meu": [0.1, 0.3, 0.2, 0.4],
        }
    )
    country_meu = pd.DataFrame(
        {
            "country_iso2": ["BE", "AT", "BE", "AT", "BE", "AT", "BE", "AT"],
            "date": ["2020-01", "2020-01", "2020-01", "2020-01", "2020-02", "2020-02", "2020-02", "2020-02"],
            "horizon": [1, 1, 3, 3, 1, 1, 3, 3],
            "meu": [0.5, 0.6, 0.7, 0.8, 0.55, 0.65, 0.75, 0.85],
        }
    )

    data = prepare_country_vs_ea_plot_data(
        ea_meu=ea_meu,
        country_meu=country_meu,
        horizon=3,
        country_order=("BE", "AT"),
        country_names={"AT": "Austria", "BE": "Belgium"},
    )

    assert data["country_iso2"].astype(str).drop_duplicates().tolist() == ["BE", "AT"]
    assert data["country_name"].drop_duplicates().tolist() == ["Belgium", "Austria"]
    assert data["ea_meu"].tolist() == [0.3, 0.4, 0.3, 0.4]
    assert data["country_meu"].tolist() == [0.7, 0.75, 0.8, 0.85]


def test_build_appendix_availability_comparison_data_uses_appendix_order():
    strict = _make_long_panel(
        ["2020-02", "2020-03"],
        [
            ("DE_A", "DE"),
            ("DE_B", "DE"),
            ("AT_A", "AT"),
            ("PT_A", "PT"),
            ("PT_B", "PT"),
            ("PT_C", "PT"),
            ("CY_A", "CY"),
        ],
    )

    data = build_appendix_availability_comparison_data(
        strict_panel=strict,
        country_names={
            "DE": "Germany",
            "AT": "Austria",
            "PT": "Portugal",
            "CY": "Cyprus",
        },
    )

    ordered = (
        data.loc[:, ["country_iso2", "country_rank"]]
        .drop_duplicates()
        .sort_values("country_rank")["country_iso2"]
        .tolist()
    )
    assert ordered[:4] == ["DE", "AT", "PT", "SK"]
    repo_counts = data.loc[data["source"] == "Repo 2021 strict"].set_index("country_iso2")
    assert int(repo_counts.loc["PT", "series_count"]) == 3
    assert int(repo_counts.loc["CY", "series_count"]) == 1


def test_country_availability_figure_adds_appendix_reference_line():
    data = pd.DataFrame(
        {
            "panel_end_year": [2021, 2021, 2021, 2021, 2021, 2021],
            "country_iso2": ["AT", "AT", "AT", "BE", "BE", "BE"],
            "country_name": ["Austria", "Austria", "Austria", "Belgium", "Belgium", "Belgium"],
            "stage": [
                "Before cleaning",
                "After incomplete-drop",
                "Fully cleaned",
                "Before cleaning",
                "After incomplete-drop",
                "Fully cleaned",
            ],
            "series_count": [110, 100, 90, 95, 85, 80],
        }
    )
    data["stage"] = pd.Categorical(
        data["stage"],
        categories=[
            "Before cleaning",
            "After incomplete-drop",
            "Fully cleaned",
        ],
        ordered=True,
    )

    fig = build_country_availability_figure(data, year=2021, country_order=("AT", "BE"))

    assert len(fig.data) == 3
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == 122
    assert list(fig.layout.yaxis.categoryarray)[-1] == "Austria"


def test_appendix_availability_comparison_figure_has_two_traces():
    data = pd.DataFrame(
        {
            "country_iso2": ["DE", "AT", "DE", "AT"],
            "country_name": ["Germany", "Austria", "Germany", "Austria"],
            "source": [
                "Paper appendix (Figure A1)",
                "Paper appendix (Figure A1)",
                "Repo 2021 strict",
                "Repo 2021 strict",
            ],
            "series_count": [122, 116, 116, 106],
            "country_rank": [0, 1, 0, 1],
        }
    )
    data["source"] = pd.Categorical(
        data["source"],
        categories=[
            "Paper appendix (Figure A1)",
            "Repo 2021 strict",
        ],
        ordered=True,
    )

    fig = build_appendix_availability_comparison_figure(data)

    assert len(fig.data) == 2
    assert fig.data[0].name == "Paper appendix (Figure A1)"
    assert fig.data[1].name == "Repo 2021 strict"
    assert len(fig.layout.annotations) >= 1


def test_country_vs_ea_figure_uses_one_facet_per_country():
    data = pd.DataFrame(
        {
            "country_iso2": pd.Categorical(
                ["AT", "AT", "BE", "BE"],
                categories=["AT", "BE"],
                ordered=True,
            ),
            "country_name": ["Austria", "Austria", "Belgium", "Belgium"],
            "date": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "date_ts": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]),
            "ea_meu": [0.3, 0.4, 0.3, 0.4],
            "country_meu": [0.35, 0.45, 0.25, 0.38],
        }
    )

    fig = build_country_vs_ea_figure(data, year=2021)

    assert len(fig.data) == 4
    assert len(fig.layout.annotations) == 2
