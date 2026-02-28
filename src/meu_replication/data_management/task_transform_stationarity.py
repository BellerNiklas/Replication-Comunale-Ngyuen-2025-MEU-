"""Apply stationarity transformations to the clean macro panel."""

from pathlib import Path

import pandas as pd

from meu_replication.config import BLD, SRC
from meu_replication.cleaning.stationarity import (
    apply_stationarity_transforms,
    build_transform_map,
)


def task_transform_stationarity(
    depends_on: dict = {
        "panel": BLD / "data" / "clean" / "macro_panel.parquet",
        "registry": SRC / "registry" / "series_registry.csv",
    },
    produces: Path = BLD / "data" / "clean" / "transformed_panel.parquet",
) -> None:
    """Join transformationcode from registry and apply stationarity transforms.

    Short and boring: read panel + registry, call pure function, write.
    """
    panel = pd.read_parquet(depends_on["panel"])
    registry = pd.read_csv(depends_on["registry"])

    transform_map = build_transform_map(registry)
    transformed = apply_stationarity_transforms(panel, transform_map)

    produces.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(produces, index=False)

    n_before = panel["series_id"].nunique()
    n_after = transformed["series_id"].nunique()
    print(
        f"Stationarity transform: {n_before} -> {n_after} series, "
        f"{len(panel)} -> {len(transformed)} rows"
    )
