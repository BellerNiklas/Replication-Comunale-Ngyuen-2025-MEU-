"""Pytask tasks for fetching ECB data."""

from pathlib import Path

import pandas as pd
import pytask

from template_project.config import BLD
from template_project.data_fetch.ecb import fetch_one, generate_all_variable_configs

# Generate all configs once at module load
ALL_CONFIGS = generate_all_variable_configs()

# Group by category
CATEGORIES = {cfg["category"]: cfg["category_name"] for cfg in ALL_CONFIGS}

for category_num, category_name in CATEGORIES.items():

    @pytask.task(id=f"category_{category_num}")
    def task_fetch_ecb_category(
        category_num: int = category_num,
        category_name: str = category_name,
        produces: Path = BLD
        / "data"
        / "raw"
        / "ecb"
        / f"category_{category_num}_{category_name}.csv",
    ) -> None:
        """Fetch all ECB variables for one category."""
        # Filter configs for this category
        cat_configs = [c for c in ALL_CONFIGS if c["category"] == category_num]

        # Fetch all series
        dfs = []
        for spec in cat_configs:
            try:
                df = fetch_one(spec)
                dfs.append(df)
            except (ValueError, RuntimeError) as e:
                print(f"SKIPPED {spec['series_id']}: {e}")

        # Concatenate and save
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.sort_values(["date", "series_id"]).reset_index(
                drop=True
            )
            produces.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(produces, index=False)
        else:
            msg = f"No data fetched for category {category_num}"
            raise ValueError(msg)
