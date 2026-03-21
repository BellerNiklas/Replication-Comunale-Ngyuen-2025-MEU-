"""Plotly helpers for final availability and MEU figures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from meu_replication.config import MEU_COUNTRIES, load_countries

AVAILABILITY_STAGE_ORDER: tuple[str, ...] = (
    "Before cleaning",
    "After incomplete-drop",
    "Fully cleaned",
)
AVAILABILITY_STAGE_COLORS: dict[str, str] = {
    "Before cleaning": "#355C7D",
    "After incomplete-drop": "#F39C12",
    "Fully cleaned": "#2A9D8F",
}
EA_COLOR = "#1F5AA6"
COUNTRY_COLOR = "#D1495B"
PANEL_COLORS: dict[int, str] = {
    2021: "#4C78A8",
    2022: "#F58518",
    2025: "#54A24B",
}
PLOT_FONT_FAMILY = "Source Sans Pro, Segoe UI, Arial, sans-serif"
MAX_COUNTRY_VARIABLES = 122
MAX_FACET_COLS = 4


def build_country_name_map() -> dict[str, str]:
    """Return an ISO2 -> display-name mapping."""
    countries = load_countries()
    return dict(
        zip(
            countries["country_iso2"].astype(str),
            countries["country_name"].astype(str),
            strict=True,
        )
    )


def month_strings_to_timestamp(months: Sequence[str] | pd.Series) -> pd.Series:
    """Convert YYYY-MM strings to month-start timestamps."""
    return pd.PeriodIndex(pd.Index(months).astype(str), freq="M").to_timestamp()


def restrict_to_panel_window(
    transformed_panel: pd.DataFrame,
    panel_window: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict a transformed panel to the exact sample window of a cleaned panel."""
    if transformed_panel.empty or panel_window.empty:
        return transformed_panel.iloc[0:0].copy()

    start = str(panel_window["date"].min())
    end = str(panel_window["date"].max())
    return transformed_panel.loc[
        transformed_panel["date"].astype(str).between(start, end)
    ].copy()


def build_availability_overview_data(
    transformed_panel: pd.DataFrame,
    strict_panels: Mapping[int, pd.DataFrame],
    clean_panels: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    """Count unique retained series across cleaning stages for each panel year."""
    rows: list[dict[str, object]] = []
    for year in sorted(strict_panels):
        strict_panel = strict_panels[year]
        clean_panel = clean_panels[year]
        before_panel = restrict_to_panel_window(transformed_panel, strict_panel)
        for stage, panel in (
            ("Before cleaning", before_panel),
            ("After incomplete-drop", strict_panel),
            ("Fully cleaned", clean_panel),
        ):
            rows.append(
                {
                    "panel_end_year": int(year),
                    "stage": stage,
                    "series_count": int(panel["series_id"].astype(str).nunique()),
                }
            )

    overview = pd.DataFrame(rows)
    overview["stage"] = pd.Categorical(
        overview["stage"],
        categories=list(AVAILABILITY_STAGE_ORDER),
        ordered=True,
    )
    return overview.sort_values(["panel_end_year", "stage"]).reset_index(drop=True)


def build_country_availability_data(
    transformed_panel: pd.DataFrame,
    strict_panel: pd.DataFrame,
    clean_panel: pd.DataFrame,
    *,
    year: int,
    country_order: Sequence[str] = MEU_COUNTRIES,
    country_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Count unique country-specific series for each stage of one panel."""
    country_names = (
        build_country_name_map() if country_names is None else dict(country_names)
    )
    before_panel = restrict_to_panel_window(transformed_panel, strict_panel)
    rows: list[dict[str, object]] = []

    for stage, panel in (
        ("Before cleaning", before_panel),
        ("After incomplete-drop", strict_panel),
        ("Fully cleaned", clean_panel),
    ):
        counts = (
            panel.loc[panel["country_iso2"].astype(str) != "U2", ["country_iso2", "series_id"]]
            .drop_duplicates()
            .groupby("country_iso2")
            .size()
            .reindex(country_order, fill_value=0)
        )
        for country_iso2 in country_order:
            rows.append(
                {
                    "panel_end_year": int(year),
                    "country_iso2": str(country_iso2),
                    "country_name": country_names.get(str(country_iso2), str(country_iso2)),
                    "stage": stage,
                    "series_count": int(counts.loc[country_iso2]),
                }
            )

    country_availability = pd.DataFrame(rows)
    country_availability["stage"] = pd.Categorical(
        country_availability["stage"],
        categories=list(AVAILABILITY_STAGE_ORDER),
        ordered=True,
    )
    return country_availability.sort_values(
        ["country_iso2", "stage"],
    ).reset_index(drop=True)


def prepare_ea_meu_plot_data(
    ea_frames: Mapping[int, pd.DataFrame],
    *,
    horizon: int = 3,
) -> pd.DataFrame:
    """Filter EA MEU outputs to one horizon and stack them across panel years."""
    parts: list[pd.DataFrame] = []
    for year in sorted(ea_frames):
        frame = ea_frames[year].copy()
        filtered = frame.loc[frame["horizon"].astype(int) == horizon, ["date", "meu"]].copy()
        filtered["panel_end_year"] = int(year)
        filtered["date_ts"] = month_strings_to_timestamp(filtered["date"])
        parts.append(filtered)

    return (
        pd.concat(parts, ignore_index=True)
        .loc[:, ["panel_end_year", "date", "date_ts", "meu"]]
        .sort_values(["panel_end_year", "date"])
        .reset_index(drop=True)
    )


def prepare_country_vs_ea_plot_data(
    ea_meu: pd.DataFrame,
    country_meu: pd.DataFrame,
    *,
    horizon: int = 3,
    country_order: Sequence[str] = MEU_COUNTRIES,
    country_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Align country MEUs with the EA MEU for a single panel year."""
    country_names = (
        build_country_name_map() if country_names is None else dict(country_names)
    )
    ea_filtered = ea_meu.loc[
        ea_meu["horizon"].astype(int) == horizon,
        ["date", "meu"],
    ].rename(columns={"meu": "ea_meu"})
    country_filtered = country_meu.loc[
        (country_meu["horizon"].astype(int) == horizon)
        & (country_meu["country_iso2"].astype(str).isin(country_order)),
        ["country_iso2", "date", "meu"],
    ].rename(columns={"meu": "country_meu"})

    merged = country_filtered.merge(ea_filtered, on="date", how="inner")
    merged["country_iso2"] = pd.Categorical(
        merged["country_iso2"].astype(str),
        categories=list(country_order),
        ordered=True,
    )
    merged["country_name"] = merged["country_iso2"].astype(str).map(country_names)
    merged["date_ts"] = month_strings_to_timestamp(merged["date"])
    return merged.sort_values(["country_iso2", "date"]).reset_index(drop=True)


def build_availability_overview_figure(data: pd.DataFrame) -> go.Figure:
    """Build the grouped availability overview bar chart."""
    fig = go.Figure()
    for stage in AVAILABILITY_STAGE_ORDER:
        stage_data = data.loc[data["stage"] == stage]
        fig.add_bar(
            x=stage_data["panel_end_year"].astype(str),
            y=stage_data["series_count"],
            name=stage,
            marker_color=AVAILABILITY_STAGE_COLORS[stage],
            hovertemplate="Year %{x}<br>Series %{y}<extra>" + stage + "</extra>",
        )

    _apply_shared_layout(
        fig,
        title="Variable Availability Across Cleaning Stages",
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Panel endpoint",
        yaxis_title="Unique series count",
    )
    return fig


def build_country_availability_figure(
    data: pd.DataFrame,
    *,
    year: int,
    country_order: Sequence[str] = MEU_COUNTRIES,
) -> go.Figure:
    """Build the appendix-style country availability comparison figure."""
    fig = go.Figure()
    category_order = [data.loc[data["country_iso2"] == iso, "country_name"].iloc[0] for iso in country_order]
    for stage in AVAILABILITY_STAGE_ORDER:
        stage_data = data.loc[data["stage"] == stage]
        fig.add_bar(
            x=stage_data["series_count"],
            y=stage_data["country_name"],
            name=stage,
            orientation="h",
            marker_color=AVAILABILITY_STAGE_COLORS[stage],
            hovertemplate="%{y}<br>Series %{x}<extra>" + stage + "</extra>",
        )

    _apply_shared_layout(
        fig,
        title=f"Country-Specific Variable Availability ({year})",
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Unique country-specific series count",
        yaxis_title="Country",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(category_order)),
    )
    fig.add_vline(
        x=MAX_COUNTRY_VARIABLES,
        line_dash="dash",
        line_color="#7A7A7A",
        opacity=0.8,
        annotation_text="Appendix maximum (122)",
        annotation_position="top right",
    )
    return fig


def build_ea_meu_h3_figure(data: pd.DataFrame) -> go.Figure:
    """Build the stacked EA MEU figure for h = 3 across panel endpoints."""
    years = sorted(data["panel_end_year"].unique().tolist())
    fig = make_subplots(
        rows=len(years),
        cols=1,
        shared_yaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f"Sample through {year}" for year in years],
    )

    y_min = float(data["meu"].min())
    y_max = float(data["meu"].max())
    padding = (y_max - y_min) * 0.08 if y_max > y_min else 0.05

    for row_idx, year in enumerate(years, start=1):
        panel = data.loc[data["panel_end_year"] == year]
        fig.add_trace(
            go.Scatter(
                x=panel["date_ts"],
                y=panel["meu"],
                mode="lines",
                line={"color": PANEL_COLORS.get(int(year), EA_COLOR), "width": 2.4},
                name=str(year),
                showlegend=False,
                hovertemplate="%{x|%Y-%m}<br>MEU %{y:.3f}<extra>Through "
                + str(year)
                + "</extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(range=[y_min - padding, y_max + padding], row=row_idx, col=1)
        fig.update_xaxes(tickformat="%Y", row=row_idx, col=1)

    _apply_shared_layout(fig, title="Euro Area MEU (h = 3)")
    fig.update_layout(height=320 * len(years))
    fig.update_yaxes(title_text="MEU", row=2 if len(years) >= 2 else 1, col=1)
    fig.update_xaxes(title_text="Date", row=len(years), col=1)
    return fig


def build_country_vs_ea_figure(
    data: pd.DataFrame,
    *,
    year: int,
) -> go.Figure:
    """Build the small-multiples appendix-style country-vs-EA figure."""
    countries = data["country_iso2"].cat.categories.tolist() if hasattr(data["country_iso2"], "cat") else data["country_iso2"].drop_duplicates().astype(str).tolist()
    countries = [country for country in countries if country in set(data["country_iso2"].astype(str))]
    n_countries = len(countries)
    n_cols = min(MAX_FACET_COLS, n_countries)
    n_rows = math.ceil(n_countries / n_cols)
    title_map = (
        data.loc[:, ["country_iso2", "country_name"]]
        .drop_duplicates()
        .set_index("country_iso2")["country_name"]
        .to_dict()
    )

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        subplot_titles=[title_map[country] for country in countries],
    )

    series_min = float(data[["country_meu", "ea_meu"]].min().min())
    series_max = float(data[["country_meu", "ea_meu"]].max().max())
    padding = (series_max - series_min) * 0.06 if series_max > series_min else 0.05

    for idx, country in enumerate(countries):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        panel = data.loc[data["country_iso2"].astype(str) == country]
        showlegend = idx == 0
        fig.add_trace(
            go.Scatter(
                x=panel["date_ts"],
                y=panel["ea_meu"],
                mode="lines",
                name="EA MEU",
                showlegend=showlegend,
                line={"color": EA_COLOR, "width": 2.0},
                hovertemplate="%{x|%Y-%m}<br>EA %{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=panel["date_ts"],
                y=panel["country_meu"],
                mode="lines",
                name="Country MEU",
                showlegend=showlegend,
                line={"color": COUNTRY_COLOR, "width": 1.9},
                hovertemplate="%{x|%Y-%m}<br>Country %{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(tickformat="%Y", row=row, col=col)
        fig.update_yaxes(range=[series_min - padding, series_max + padding], row=row, col=col)

    _apply_shared_layout(fig, title=f"Country vs Euro Area MEU (h = 3, through {year})")
    fig.update_layout(height=max(320 * n_rows, 700))
    return fig


def write_plot_outputs(
    fig: go.Figure,
    *,
    html_path: Path,
    png_path: Path,
) -> None:
    """Write interactive HTML and static PNG outputs for one figure."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    fig.write_image(png_path, scale=2)


def _apply_shared_layout(
    fig: go.Figure,
    *,
    title: str,
) -> None:
    """Apply a consistent visual theme across all figures."""
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor="#FAF9F6",
        plot_bgcolor="#FFFFFF",
        font={"family": PLOT_FONT_FAMILY, "size": 13, "color": "#1F2933"},
        margin={"l": 72, "r": 28, "t": 78, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        hovermode="closest",
    )
