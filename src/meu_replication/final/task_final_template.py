"""Generate final Plotly figures for availability and MEU outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytask

from meu_replication.analysis.panel_specs import ANALYSIS_PANELS
from meu_replication.config import BLD, FINAL_PLOTS
from meu_replication.final.plotly_figures import (
    build_appendix_availability_comparison_data,
    build_appendix_availability_comparison_figure,
    build_availability_overview_data,
    build_availability_overview_figure,
    build_country_availability_data,
    build_country_availability_figure,
    build_country_name_map,
    build_country_vs_ea_figure,
    build_ea_meu_h3_figure,
    prepare_country_vs_ea_plot_data,
    prepare_ea_meu_plot_data,
    write_plot_outputs,
)

_PANEL_SPECS_BY_YEAR = {
    int(spec.panel_name.split("_")[2]): spec for spec in ANALYSIS_PANELS
}
_PLOT_YEARS = tuple(sorted(_PANEL_SPECS_BY_YEAR))


def _availability_paths(year: int) -> dict[str, Path]:
    return {
        "strict": BLD / "data" / "clean" / f"panel_2003_{year}_strict.parquet",
        "clean": BLD / "data" / "clean" / f"panel_2003_{year}_strict_corr.parquet",
    }


def _availability_outputs(name: str) -> dict[str, Path]:
    return {
        "html": FINAL_PLOTS / "availability" / f"{name}.html",
        "png": FINAL_PLOTS / "availability" / f"{name}.png",
    }


def _meu_outputs(name: str) -> dict[str, Path]:
    return {
        "html": FINAL_PLOTS / "meu" / f"{name}.html",
        "png": FINAL_PLOTS / "meu" / f"{name}.png",
    }


def _ea_meu_path(year: int) -> Path:
    return _PANEL_SPECS_BY_YEAR[year].layout.euro_area_results_dir / "meu_ea.parquet"


def _country_meu_path(year: int) -> Path:
    return (
        _PANEL_SPECS_BY_YEAR[year].layout.country_results_dir / "all_countries_meu.parquet"
    )


def run_availability_overview_plot(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
) -> None:
    """Render the aggregate availability overview figure."""
    transformed_panel = pd.read_parquet(depends_on["transformed_panel"])
    strict_panels = {
        year: pd.read_parquet(depends_on[f"strict_{year}"]) for year in _PLOT_YEARS
    }
    clean_panels = {
        year: pd.read_parquet(depends_on[f"clean_{year}"]) for year in _PLOT_YEARS
    }

    data = build_availability_overview_data(
        transformed_panel=transformed_panel,
        strict_panels=strict_panels,
        clean_panels=clean_panels,
    )
    fig = build_availability_overview_figure(data)
    write_plot_outputs(fig, html_path=produces["html"], png_path=produces["png"])


def run_country_availability_plot(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    year: int,
) -> None:
    """Render the country availability figure for one panel year."""
    country_names = build_country_name_map()
    data = build_country_availability_data(
        transformed_panel=pd.read_parquet(depends_on["transformed_panel"]),
        strict_panel=pd.read_parquet(depends_on["strict"]),
        clean_panel=pd.read_parquet(depends_on["clean"]),
        year=year,
        country_names=country_names,
    )
    fig = build_country_availability_figure(data, year=year)
    write_plot_outputs(fig, html_path=produces["html"], png_path=produces["png"])


def run_appendix_availability_comparison_plot(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
) -> None:
    """Render the repo-versus-appendix availability comparison figure."""
    data = build_appendix_availability_comparison_data(
        strict_panel=pd.read_parquet(depends_on["strict_2021"]),
        country_names=build_country_name_map(),
    )
    fig = build_appendix_availability_comparison_figure(data)
    write_plot_outputs(fig, html_path=produces["html"], png_path=produces["png"])


def run_ea_meu_h3_plot(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
) -> None:
    """Render the stacked EA MEU figure at horizon three."""
    data = prepare_ea_meu_plot_data(
        {
            year: pd.read_parquet(depends_on[f"ea_{year}"])
            for year in _PLOT_YEARS
        },
        horizon=3,
    )
    fig = build_ea_meu_h3_figure(data)
    write_plot_outputs(fig, html_path=produces["html"], png_path=produces["png"])


def run_country_vs_ea_plot(
    *,
    depends_on: dict[str, Path],
    produces: dict[str, Path],
    year: int,
) -> None:
    """Render the appendix-style country-vs-EA figure for one panel year."""
    data = prepare_country_vs_ea_plot_data(
        ea_meu=pd.read_parquet(depends_on["ea_meu"]),
        country_meu=pd.read_parquet(depends_on["country_meu"]),
        horizon=3,
        country_names=build_country_name_map(),
    )
    fig = build_country_vs_ea_figure(data, year=year)
    write_plot_outputs(fig, html_path=produces["html"], png_path=produces["png"])


@pytask.task(id="availability_overview")
def task_plot_availability_overview(
    depends_on: dict[str, Path] = {
        "transformed_panel": BLD / "data" / "clean" / "transformed_panel.parquet",
        **{
            f"{stage}_{year}": _availability_paths(year)[stage]
            for year in _PLOT_YEARS
            for stage in ("strict", "clean")
        },
    },
    produces: dict[str, Path] = _availability_outputs("availability_overview"),
) -> None:
    """Generate the aggregate availability overview figure."""
    run_availability_overview_plot(depends_on=depends_on, produces=produces)


@pytask.task(id="availability_appendix_2021")
def task_plot_appendix_availability_comparison(
    depends_on: dict[str, Path] = {
        "strict_2021": _availability_paths(2021)["strict"],
    },
    produces: dict[str, Path] = _availability_outputs("availability_vs_appendix_2021"),
) -> None:
    """Generate the appendix-style 2021 availability comparison figure."""
    run_appendix_availability_comparison_plot(
        depends_on=depends_on,
        produces=produces,
    )


for _year in _PLOT_YEARS:

    @pytask.task(id=f"availability_country_{_year}")
    def task_plot_availability_by_country(
        year: int = _year,
        depends_on: dict[str, Path] = {
            "transformed_panel": BLD / "data" / "clean" / "transformed_panel.parquet",
            **_availability_paths(_year),
        },
        produces: dict[str, Path] = _availability_outputs(
            f"availability_by_country_{_year}"
        ),
    ) -> None:
        """Generate the country availability figure for one panel year."""
        run_country_availability_plot(
            depends_on=depends_on,
            produces=produces,
            year=year,
        )


@pytask.task(id="ea_meu_h3")
def task_plot_ea_meu_h3(
    depends_on: dict[str, Path] = {
        f"ea_{year}": _ea_meu_path(year) for year in _PLOT_YEARS
    },
    produces: dict[str, Path] = _meu_outputs("ea_meu_h3_by_panel"),
) -> None:
    """Generate the EA MEU figure at horizon three."""
    run_ea_meu_h3_plot(depends_on=depends_on, produces=produces)


for _year in _PLOT_YEARS:

    @pytask.task(id=f"country_vs_ea_{_year}")
    def task_plot_country_vs_ea_h3(
        year: int = _year,
        depends_on: dict[str, Path] = {
            "ea_meu": _ea_meu_path(_year),
            "country_meu": _country_meu_path(_year),
        },
        produces: dict[str, Path] = _meu_outputs(f"country_vs_ea_h3_{_year}"),
    ) -> None:
        """Generate the country-vs-EA MEU figure for one panel year."""
        run_country_vs_ea_plot(
            depends_on=depends_on,
            produces=produces,
            year=year,
        )
