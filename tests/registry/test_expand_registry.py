import pandas as pd
import pytest

from meu_replication.registry.expand_registry import (
    _expand_placeholders,
    _expand_template_for_country,
    _expand_template_for_ea,
    _nan_to_empty,
    expand_registry,
)


# --- Fixtures ---


def _make_country(
    iso2="DE",
    iso3="DEU",
    eurostat="DE",
    oecd="DEU",
    bis="DE",
):
    return pd.Series(
        {
            "country_iso2": iso2,
            "country_iso3": iso3,
            "country_name": "Test",
            "source_eurostat": eurostat,
            "source_ecb": iso2,
            "source_oecd": oecd,
            "source_bis": bis,
            "ea_member": True,
        }
    )


def _make_template(
    template_id="IP_001",
    source="eurostat",
    scope="country",
    key_template="",
    filters_json_template='{"geo": "{COUNTRY_EUROSTAT}"}',
    category=1,
    category_name="Industrial_production",
    variable_name="Test variable",
    dataset="STS_INPR_M",
    transformationcode=5,
):
    return pd.Series(
        {
            "template_id": template_id,
            "source": source,
            "category": category,
            "category_name": category_name,
            "variable_name": variable_name,
            "dataset": dataset,
            "key_template": key_template,
            "filters_json_template": filters_json_template,
            "scope": scope,
            "unit_measure_filter": "",
            "frequency": "M",
            "start_period": "2003-01",
            "transformationcode": transformationcode,
        }
    )


# --- _expand_placeholders ---


@pytest.mark.parametrize(
    ("template", "country_kwargs", "expected"),
    [
        ("M.{COUNTRY_ISO2}.N.A", {"iso2": "DE"}, "M.DE.N.A"),
        ('{"geo": "{COUNTRY_EUROSTAT}"}', {"iso2": "GR", "eurostat": "EL"}, '{"geo": "EL"}'),
        ("{COUNTRY_OECD}.M.BCICP", {"iso2": "FR", "iso3": "FRA", "oecd": "FRA"}, "FRA.M.BCICP"),
        ("M.N.B.{COUNTRY_BIS}", {"iso2": "DE", "bis": "DE"}, "M.N.B.DE"),
        ("", {}, ""),
        ("M.USD.EUR.SP00.A", {}, "M.USD.EUR.SP00.A"),
    ],
)
def test_expand_placeholders(template, country_kwargs, expected):
    country = _make_country(**country_kwargs)
    assert _expand_placeholders(template, country) == expected


@pytest.mark.parametrize(
    ("iso2", "eurostat"),
    [
        ("DE", "DE"),
        ("FR", "FR"),
        ("GR", "EL"),
        ("AT", "AT"),
    ],
)
def test_expand_placeholders_eurostat_per_country(iso2, eurostat):
    country = _make_country(iso2=iso2, eurostat=eurostat)
    result = _expand_placeholders('{"geo": "{COUNTRY_EUROSTAT}"}', country)
    assert result == f'{{"geo": "{eurostat}"}}'


# --- _expand_template_for_ea ---


def test_expand_template_for_ea_series_id():
    template = _make_template(template_id="FX_001", scope="ea")
    result = _expand_template_for_ea(template)
    assert result["series_id"] == "U2_FX_001"


def test_expand_template_for_ea_country_iso2():
    template = _make_template(scope="ea")
    result = _expand_template_for_ea(template)
    assert result["country_iso2"] == "U2"


def test_expand_template_for_ea_preserves_key():
    template = _make_template(
        scope="ea",
        key_template="M.U2.EUR.4F.BB.U2_10Y.YLD",
        filters_json_template="",
    )
    result = _expand_template_for_ea(template)
    assert result["key"] == "M.U2.EUR.4F.BB.U2_10Y.YLD"


# --- _expand_template_for_country ---


def test_expand_template_for_country_series_id():
    template = _make_template(template_id="IP_001")
    country = _make_country(iso2="FR", eurostat="FR")
    result = _expand_template_for_country(template, country)
    assert result["series_id"] == "FR_IP_001"


def test_expand_template_for_country_filters_resolved():
    template = _make_template(
        filters_json_template='{"geo": "{COUNTRY_EUROSTAT}", "unit": "I21"}',
    )
    country = _make_country(iso2="GR", eurostat="EL")
    result = _expand_template_for_country(template, country)
    assert result["filters_json"] == '{"geo": "EL", "unit": "I21"}'


def test_expand_template_for_country_key_resolved():
    template = _make_template(
        source="ecb",
        key_template="M.{COUNTRY_ISO2}.N.A.A20.A.1.U6.1000.Z01.E",
        filters_json_template="",
    )
    country = _make_country(iso2="IT")
    result = _expand_template_for_country(template, country)
    assert result["key"] == "M.IT.N.A.A20.A.1.U6.1000.Z01.E"


# --- expand_registry (full integration) ---


def _make_u2_country():
    return pd.Series(
        {
            "country_iso2": "U2",
            "country_iso3": "U2",
            "country_name": "Euro Area",
            "source_eurostat": "U2",
            "source_ecb": "U2",
            "source_oecd": "EA19",
            "source_bis": "EA",
            "ea_member": False,
        }
    )


def test_expand_registry_counts():
    templates = pd.DataFrame(
        [
            _make_template(template_id="IP_001", scope="country"),
            _make_template(
                template_id="FX_001",
                scope="ea",
                source="ecb",
                key_template="M.USD.EUR.SP00.A",
                filters_json_template="",
            ),
        ]
    )
    countries = pd.DataFrame(
        [
            _make_country(iso2="DE"),
            _make_country(iso2="FR", iso3="FRA", eurostat="FR", oecd="FRA"),
            _make_u2_country(),
        ]
    )
    result = expand_registry(templates, countries)
    assert len(result) == 3


def test_expand_registry_unique_series_ids():
    templates = pd.DataFrame(
        [
            _make_template(template_id="IP_001", scope="country"),
            _make_template(template_id="IP_002", scope="country"),
        ]
    )
    countries = pd.DataFrame(
        [
            _make_country(iso2="DE"),
            _make_country(iso2="FR", iso3="FRA", eurostat="FR", oecd="FRA"),
        ]
    )
    result = expand_registry(templates, countries)
    assert result["series_id"].is_unique


def test_expand_registry_unknown_scope_raises():
    templates = pd.DataFrame(
        [_make_template(template_id="BAD_001", scope="global")]
    )
    countries = pd.DataFrame([_make_country()])
    with pytest.raises(ValueError, match="Unknown scope 'global'"):
        expand_registry(templates, countries)


def test_expand_registry_u2_excluded_from_country():
    templates = pd.DataFrame(
        [_make_template(template_id="IP_001", scope="country")]
    )
    countries = pd.DataFrame(
        [
            _make_country(iso2="DE"),
            _make_u2_country(),
        ]
    )
    result = expand_registry(templates, countries)
    assert "U2" not in result["country_iso2"].values


# --- _nan_to_empty ---


def test_nan_to_empty_with_nan():
    assert _nan_to_empty(float("nan")) == ""


def test_nan_to_empty_with_none():
    assert _nan_to_empty(None) == ""


def test_nan_to_empty_with_string():
    assert _nan_to_empty("hello") == "hello"


# --- Full pipeline roundtrip ---


def test_expand_real_templates():
    from meu_replication.config import load_countries
    from meu_replication.registry.expand_registry import (
        load_templates,
    )
    from meu_replication.registry.registry_io import (
        validate_registry,
    )

    templates = load_templates()
    countries = load_countries()
    registry = expand_registry(templates, countries)

    n_country = len(templates[templates.scope == "country"])
    n_ea = len(templates[templates.scope == "ea"])
    expected = n_country * 19 + n_ea
    assert len(registry) == expected

    assert registry["series_id"].is_unique

    validate_registry(registry)


def test_fin_dsh_templates_expand_with_first_difference():
    from meu_replication.config import load_countries
    from meu_replication.registry.expand_registry import load_templates

    templates = load_templates()
    countries = load_countries()
    registry = expand_registry(templates, countries)

    fin_dsh = registry[
        registry["series_id"].isin(
            ["DE_FIN_DSH_001", "DE_FIN_DSH_002", "DE_FIN_DSH_003"]
        )
    ]
    assert set(fin_dsh["transformationcode"]) == {2}


def test_bond_006_expands_for_u2():
    from meu_replication.config import load_countries
    from meu_replication.registry.expand_registry import load_templates

    templates = load_templates()
    countries = load_countries()
    registry = expand_registry(templates, countries)

    bond_006 = registry[registry["series_id"] == "U2_BOND_006"]
    assert len(bond_006) == 1
    assert bond_006.iloc[0]["country_iso2"] == "U2"
    assert int(bond_006.iloc[0]["transformationcode"]) == 2
