"""
build_base_m_prices_uah.py
==========================
Monthly BASE_M contract prices in UAH/MWh, excluding VAT.

Under the PSO rules the BASE_M product of supply month M is priced at the published
DAM BASE index of month M-3 (OES Ukraine, Market Operator). The two source workbooks in inputs/ are
formula-based reference prices, not observed auction contracts (their own notes say so):
    inputs/BASE_M_2024_Ukraine.xlsx sheet "BASE_M 2024"  supply months 2024-01..12; sheet "DAM BASE archive"
    inputs/BASE_M_2025_Ukraine.xlsx sheet "BASE_M_2025"  supply months 2025-01..12; sheet "DAM_BASE_Indices"
The build checks that every BASE_M price equals the DAM BASE index of its index month in the index
sheets, and that the two files agree where their index sheets overlap (2024-10..12).

Output: inputs/base_m_prices_uah.csv, one row per supply month (24 rows), columns
    supply_month, index_month, base_m_price_uah_mwh, source_file, source_url
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # the folder holding code/ and inputs/
DATA = ROOT / "inputs"
SOURCES = {
    "BASE_M_2024_Ukraine.xlsx": {"sheet": "BASE_M 2024", "header": 4, "supply": "Delivery month",
                                 "index": "Reference month (M−3)", "price": "BASE_M price (UAH/MWh)",
                                 "url": "Ref. index official source", "index_sheet": "DAM BASE archive",
                                 "index_header": 3, "index_month": "index_month",
                                 "index_price": "DAM_BASE_UAH_MWh_ex_VAT"},
    "BASE_M_2025_Ukraine.xlsx": {"sheet": "BASE_M_2025", "header": 3, "supply": "Supply month",
                                 "index": "Index month (M−3)", "price": "DAM BASE, UAH/MWh",
                                 "url": "Source (Market Operator)", "index_sheet": "DAM_BASE_Indices",
                                 "index_header": 3, "index_month": "Index month",
                                 "index_price": "DAM BASE, UAH/MWh"},
}
OUT_CSV = ROOT / "inputs" / "base_m_prices_uah.csv"
COLUMNS = ["supply_month", "index_month", "base_m_price_uah_mwh", "source_file", "source_url"]
MONTH = r"^\d{4}-\d{2}$"
TOL = 1e-9                                           # UAH/MWh: prices are copied, not computed


def months_back(month: str, k: int) -> str:
    """The month k months before 'YYYY-MM'."""
    return (pd.Period(month, freq="M") - k).strftime("%Y-%m")


def _rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Keep the month rows of a sheet (the sheets end with summary and note rows)."""
    return df[df[col].astype(str).str.match(MONTH)].copy()


def read_source(path: Path, spec: dict) -> tuple[pd.DataFrame, pd.Series]:
    """BASE_M rows of one workbook and its DAM BASE index series (index month -> UAH/MWh)."""
    d = _rows(pd.read_excel(path, sheet_name=spec["sheet"], header=spec["header"]), spec["supply"])
    out = pd.DataFrame({"supply_month": d[spec["supply"]].astype(str),
                        "index_month": d[spec["index"]].astype(str),
                        "base_m_price_uah_mwh": d[spec["price"]].astype(float),
                        "source_file": path.name,
                        "source_url": d[spec["url"]].astype(str)})
    ix = _rows(pd.read_excel(path, sheet_name=spec["index_sheet"], header=spec["index_header"]),
               spec["index_month"])
    index = pd.Series(ix[spec["index_price"]].astype(float).to_numpy(),
                      index=ix[spec["index_month"]].astype(str).to_numpy())
    return out, index


def check(out: pd.DataFrame, indices: list[pd.Series]) -> None:
    """The M-3 rule, 24 consecutive supply months, and price = index of the index month."""
    if list(out.columns) != COLUMNS or len(out) != 24 or not out["supply_month"].is_unique:
        raise ValueError("expected 24 unique supply months")
    expected = pd.period_range("2024-01", "2025-12", freq="M").strftime("%Y-%m").tolist()
    if out["supply_month"].tolist() != expected:
        raise ValueError("supply months are not 2024-01..2025-12 in order")
    lag = out["supply_month"].map(lambda m: months_back(m, 3))
    if not (lag == out["index_month"]).all():
        raise ValueError("index month is not supply month minus 3")
    if not (out["base_m_price_uah_mwh"] > 0).all():
        raise ValueError("non-positive BASE_M price")
    a, b = indices
    common = a.index.intersection(b.index)
    if len(common) == 0 or (a[common] - b[common]).abs().max() > TOL:
        raise ValueError("the two DAM BASE index sheets disagree where they overlap")
    index = pd.concat([a, b[~b.index.isin(a.index)]])
    gap = (out["base_m_price_uah_mwh"] - out["index_month"].map(index)).abs()
    if gap.isna().any() or gap.max() > TOL:
        raise ValueError("a BASE_M price differs from the DAM BASE index of its index month")


def build(data: Path = DATA) -> pd.DataFrame:
    parts, indices = [], []
    for name, spec in SOURCES.items():
        rows, index = read_source(data / name, spec)
        parts.append(rows)
        indices.append(index)
    out = pd.concat(parts, ignore_index=True)[COLUMNS]
    check(out, indices)
    return out


def main() -> None:
    df = build()
    df.to_csv(OUT_CSV, index=False, lineterminator="\n")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(df)} rows, "
          f"{df['supply_month'].iloc[0]}..{df['supply_month'].iloc[-1]}, "
          f"{df['base_m_price_uah_mwh'].min():.2f}..{df['base_m_price_uah_mwh'].max():.2f} UAH/MWh")


if __name__ == "__main__":
    main()
