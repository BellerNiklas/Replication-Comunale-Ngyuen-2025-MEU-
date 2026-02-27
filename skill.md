# Reproducibility Rules for Final Project (Agentic Coding Guide)

**Project**: MacroEconomic Uncertainty database replication (Comunale & Nguyen 2025)
**Environment**: Pixi + pytask + Python 3.14
**Critical Rule**: Everything must be reproducible from `main` branch with zero uncommitted changes

---

## 1. CARDINAL RULES (Never Break These)

### 1.1 Always Use Pixi

```bash
# CORRECT: Run through pixi
pixi run pytask
pixi run pytest
pixi run prek

# WRONG: Direct Python calls (breaks reproducibility)
python script.py        # ❌ Uses wrong Python/environment
pytest                  # ❌ Uses system pytest, not project version
```

**Why**: Pixi ensures exact package versions via `pixi.lock`. Direct calls use system Python.

### 1.2 Source vs Output Separation

```
src/                    # Hand-written code ONLY (version controlled)
bld/                    # Generated outputs (safe to delete, NOT committed)
_build/                 # Document outputs (safe to delete, NOT committed)
```

**Rules**:
- ✅ Edit files in `src/`
- ❌ Never edit generated files in `bld/` or `_build/`
- ❌ Never commit `bld/` or `_build/` contents (they're in `.gitignore`)
- ✅ Final results must regenerate from clean checkout via `pixi run pytask`

### 1.3 No Hardcoded Paths

```python
# CORRECT: Portable paths
from pathlib import Path
from meu_replication.config import BLD, SRC

output_path = BLD / "data" / "cleaned.csv"

# WRONG: Breaks on other machines
output_path = "C:/Users/nikla/Projects/..."  # ❌
```

---

## 2. DEPENDENCY MANAGEMENT (Pixi + pixi.lock)

### 2.1 How to Add Dependencies

```bash
# Conda packages (prefer this)
pixi add pandas numpy

# PyPI-only packages
pixi add --pypi some-package

# Or manually edit pyproject.toml:
[tool.pixi.dependencies]
pandas = ">=2.0"

[tool.pixi.pypi-dependencies]
some-package = ">=1.0"
```

**Then**:
```bash
pixi install          # Updates pixi.lock
```

### 2.2 Lockfile Policy (Critical for Reproducibility)

- **Always commit `pixi.lock`** when dependencies change
- `pixi.lock` pins exact versions (e.g., `pandas==2.2.1` not `pandas>=2.0`)
- At submission time: verify all versions are pinned and consistent
- **Never** edit `pixi.lock` manually

---

## 3. PYTASK: Building the Computational DAG

### 3.1 Task Discovery Rules

pytask auto-discovers:
- Files named `task_*.py`
- Functions named `task_*` inside them

### 3.2 Task Function Signature (STRICT)

```python
from pathlib import Path
from meu_replication.config import BLD

# CORRECT: Explicit dependencies and products
def task_clean_data(
    depends_on: Path = BLD / "data" / "raw.csv",
    produces: Path = BLD / "data" / "cleaned.csv",
) -> None:
    """Clean raw data."""
    raw = pd.read_csv(depends_on)
    cleaned = clean_data(raw)  # Pure function
    cleaned.to_csv(produces, index=False)

# WRONG: No produces declared
def task_clean_data():  # ❌
    pd.read_csv("raw.csv").to_csv("cleaned.csv")  # pytask can't track this
```

**Rules**:
- Use `produces` for all outputs
- Use other default arguments for dependencies
- pytask uses these signatures to build the DAG
- **Never write files not declared in `produces`**

### 3.3 Multiple Products

```python
# Option 1: Dict of products (for related outputs)
def task_fit_models(
    depends_on: Path = BLD / "data" / "cleaned.csv",
    produces: dict[str, Path] = {
        "model_1": BLD / "models" / "model_1.pkl",
        "model_2": BLD / "models" / "model_2.pkl",
    },
) -> None:
    ...

# Option 2: Loop with @task(id=...) for distinct tasks
from meu_replication.config import COUNTRIES

for country in COUNTRIES:
    @pytask.task(id=country)
    def task_fetch_data(
        produces: Path = BLD / "data" / f"{country}.csv",
    ) -> None:
        ...
```

### 3.4 Task Hygiene

- Task functions should be **short and boring** (read → transform → write)
- Real logic goes in **pure helper functions** under `src/`
- Never mutate raw inputs
- Never write outside `produces`

---

## 4. DATA CLEANING: The Three Functional Rules

### 4.1 The Rules (Mandatory for All Cleaning Code)

1. **Start with an empty DataFrame** (construct cleaned columns from raw)
2. **Touch every variable only once** (each cleaned column assigned exactly once)
3. **Touch with a pure function** (no side effects, depends only on inputs)

### 4.2 Example: Good vs Bad

```python
# GOOD: Functional cleaning
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data following the three rules."""
    return pd.DataFrame({
        "date": clean_dates(raw["date_raw"]),
        "value": clean_values(raw["value_raw"]),
        "country": clean_country_codes(raw["geo"]),
    })

def clean_dates(dates: pd.Series) -> pd.Series:
    """Convert YYYY-MM format to datetime."""
    return pd.to_datetime(dates, format="%Y-%m")

# BAD: Mutating transformations
def clean_data(df):  # ❌
    df["date"] = pd.to_datetime(df["date"])     # Mutates input
    df["value"] = df["value"].fillna(0)         # Mutates again
    df.drop(columns=["old_col"], inplace=True)  # More mutation
    return df  # Unclear what happened
```

### 4.3 Why These Rules Matter

- **Debuggability**: Search for `"column_name"` finds exactly where it's defined
- **No hidden state**: Can't have accidental dependencies between transformations
- **Testability**: Pure functions are trivial to test
- **Reproducibility**: Same inputs → same outputs, always

---

## 5. TESTING (Minimum Standard)

### 5.1 What to Test

Test **behavior**, not implementation:
- Typical inputs
- Corner cases (empty data, missing values, edge values)
- Error conditions (invalid inputs should raise informative errors)
- **Every bug you've encountered** (regression tests)

### 5.2 Test Structure

```python
import pytest
import pandas as pd
from meu_replication.data_management.clean import clean_dates

def test_clean_dates_typical():
    """Test typical YYYY-MM input."""
    raw = pd.Series(["2024-01", "2024-02"])
    result = clean_dates(raw)
    expected = pd.to_datetime(["2024-01-01", "2024-02-01"])
    pd.testing.assert_series_equal(result, expected)

def test_clean_dates_invalid():
    """Test that invalid dates raise ValueError."""
    raw = pd.Series(["not-a-date"])
    with pytest.raises(ValueError, match="does not match format"):
        clean_dates(raw)

@pytest.mark.parametrize("input_val,expected", [
    ("2024-01", "2024-01-01"),
    ("2024-12", "2024-12-01"),
])
def test_clean_dates_parametrized(input_val, expected):
    """Parametrized tests for multiple cases."""
    result = clean_dates(pd.Series([input_val]))
    assert result.iloc[0] == pd.Timestamp(expected)
```

### 5.3 Testing Checklist

- ✅ One assertion per test (when possible)
- ✅ Test fails when it should (verify counterexample)
- ✅ Use `pytest.raises(...)` for expected errors
- ✅ Use `@pytest.mark.parametrize` to avoid duplication
- ❌ No "or-style" assertions that can pass for wrong reasons

---

## 6. PLOTLY EXPORT (Static Figures for Documents)

### 6.1 Setup Kaleido (Browser for Static Export)

```bash
# One-time setup in pixi environment
pixi run plotly_get_chrome
```

### 6.2 Export in pytask Tasks

```python
import plotly.express as px
from pathlib import Path

def task_create_figure(
    depends_on: Path = BLD / "data" / "cleaned.csv",
    produces: Path = BLD / "figures" / "plot.png",
) -> None:
    """Create publication-ready static figure."""
    df = pd.read_csv(depends_on)
    fig = px.line(df, x="date", y="value", title="My Plot")

    # Update layout for publication
    fig.update_layout(
        font=dict(size=14),
        title_font_size=16,
        showlegend=False,  # Avoid unnecessary legends
    )

    # Write static export
    fig.write_image(produces, width=800, height=600)
```

---

## 7. DOCUMENTATION REQUIREMENTS

### 7.1 README Must Cover

1. **What**: Project purpose and pipeline entry point
2. **How**: Exact commands to install and run
   ```bash
   pixi install
   pixi run pytask
   pixi run pytest
   ```
3. **Where**: Directory layout (src, bld, documents, etc.)
4. **Special requirements**: Runtime, memory, data restrictions, credentials

### 7.2 Replication Package Checklist

For final submission:
- ✅ Data availability statement
- ✅ Variable definitions and metadata
- ✅ Code for all transformations
- ✅ Software dependencies (via `pixi.lock`)
- ✅ Expected runtime estimate
- ✅ License information
- ✅ Document any omissions/deviations

---

## 8. DEBUGGING PLAYBOOK

When something breaks:

1. **State expected behavior**: "What should this do?"
2. **Check environment**: Are you in the right pixi env?
3. **Minimal failing case**: Reduce to smallest example
4. **Isolate**: Test individual functions separately
5. **One change at a time**: Form hypothesis, test, repeat
6. **Write it down**: Turn failure into a regression test
7. **Prefer debugger over print**: Use `pdbp` breakpoints

```python
# Add breakpoint for debugging
import pdbp
pdbp.set_trace()  # Execution pauses here
```

---

## 9. DEFINITION OF DONE (Checklist Before Commit)

Before considering any change "done":

- [ ] `pixi run pytest` passes
- [ ] `pixi run pytask` completes successfully
- [ ] No raw/source inputs edited in place
- [ ] New outputs declared in `produces` and written only there
- [ ] Logic in reusable helper functions (pure where feasible)
- [ ] Bug fixes include regression tests
- [ ] README/docs updated if commands/deps/outputs changed
- [ ] `pixi.lock` committed if dependencies changed
- [ ] No uncommitted changes remain
- [ ] Code follows style (`pixi run prek` passes)

---

## 10. QUICK REFERENCE: Common Commands

```bash
# Full pipeline
pixi run pytask

# Tests
pixi run pytest                              # All tests
pixi run pytest tests/test_specific.py       # One file
pixi run pytest -k test_function_name        # One test

# Pre-commit checks
pixi run prek

# Clean build (when DAG is confused)
rm -rf bld/ _build/
pixi run pytask

# View outputs
pixi run view-paper      # Paper with live reload
pixi run view-pres       # Presentation with live reload

# Documentation
pixi run -e docs docs    # Build docs
pixi run -e docs view-docs
```

---

## 11. ANTI-PATTERNS TO AVOID

❌ **Mutating DataFrames in place**
```python
df["new_col"] = ...  # Creates side effects
```

✅ **Constructing new DataFrames**
```python
cleaned = pd.DataFrame({"new_col": ...})
```

---

❌ **Hardcoded absolute paths**
```python
data = pd.read_csv("C:/Users/...")
```

✅ **Relative paths from config**
```python
from meu_replication.config import BLD
data = pd.read_csv(BLD / "data" / "file.csv")
```

---

❌ **Direct Python/pip calls**
```bash
python script.py
pip install pandas
```

✅ **Always through Pixi**
```bash
pixi run python script.py
pixi add pandas
```

---

❌ **Writing undeclared outputs**
```python
def task_clean():
    df.to_csv("output.csv")  # pytask doesn't know about this
```

✅ **Declaring all products**
```python
def task_clean(produces: Path = ...):
    df.to_csv(produces)
```

---

## 12. PROJECT-SPECIFIC NOTES

### Data Fetchers (Current State)

- `eurostat.py`: 87 variables (Categories 1-6) — **Hardcoded to DE**
- `ecb.py`: 52 variables (Cat 4, 7, 8) — **Cat 8 is EA-level (shared)**
- `bis.py`: 1 variable (Cat 7 NEER) — **Already parameterized**
- `oecd.py`: 8 variables (Cat 6, 7) — **Already parameterized**

### Next Phase: Multi-Country Expansion

**To parameterize**:
1. Eurostat: Replace `geo="DE"` and `"DE_"` prefixes with country parameter
2. ECB Cat 4/7: Replace `.DE.` in SDMX keys with `{country}` placeholder
3. ECB Cat 8: Fetch once (EA-level), share across all countries

**19 EA members to support**:
DE, FR, IT, ES, NL, BE, AT, FI, GR, PT, IE, SK, SI, LT, LV, EE, LU, CY, MT

---

## FINAL REMINDER

**Reproducibility = Someone else can get your exact results**

This requires:
1. Exact environment (`pixi.lock`)
2. Complete DAG (`pytask` with all `produces` declared)
3. No hidden mutations (functional data cleaning)
4. No manual steps (everything scripted)
5. No uncommitted changes (main branch is source of truth)

**When in doubt, ask: "Could a collaborator reproduce this from a clean checkout?"**
