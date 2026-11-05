"""
build_dam_prices_uah.py
=======================
Hourly DAM prices in UAH on the model calendar (8,760 h).

Joins the DAM source workbook (inputs/DAM_2019-2025_v05_Full.xlsx, sheet DAM) on the existing
DAM_timestamp mapping of inputs/hourly_tou_dam_prices.xlsx. The hour mapping is taken as it is:
never re-derive it, and never re-apply the hour-ending relabelling that produced it (it would shift the
series by one more hour).

All prices exclude VAT. Columns in UAH/MWh, with the workbook letters:
    E  DAM_net_price_uah_mwh     OREE clearing price
    O  distribution_uah_mwh      distribution tariff in force on the delivery day
    P  transmission_uah_mwh      transmission tariff in force on the delivery day
    Q  DAM_gross_price_uah_mwh   E + O + P, the retail-equivalent price for DAM savings
fx_uah_per_usd (= E / USD price, workbook column R) is kept for the USD reconciliation only.
The gross price is E + O + P; the build stops if the workbook's gross price is not E + O + P.

Hours absent from the workbook (25 in the model window) are filled by this rule:
E = the USD price of the model file x the FX of the delivery day, and O, P of that day,
so the gross price is E + O + P. For a delivery day absent from the workbook (27 Jul 2025) the nearest day
present is used, the earlier one on a tie. These rows have filled=True and fill_day = the day used.

Output: inputs/hourly_dam_prices_uah.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # the folder holding code/ and inputs/
SOURCE_XLSX = ROOT / "inputs" / "DAM_2019-2025_v05_Full.xlsx"
MODEL_XLSX = ROOT / "inputs" / "hourly_tou_dam_prices.xlsx"
OUT_CSV = ROOT / "inputs" / "hourly_dam_prices_uah.csv"
GROSS_TOL = 1e-6                                     # UAH/MWh: the workbook's gross against E + O + P

COLUMNS = ["timestamp", "DAM_timestamp", "DAM_net_price_uah_mwh", "DAM_gross_price_uah_mwh",
           "distribution_uah_mwh", "transmission_uah_mwh", "fx_uah_per_usd", "filled", "fill_day"]
_SRC = {"price_uah_per_mwh": "DAM_net_price_uah_mwh", "gross_price_uah_per_mwh": "DAM_gross_price_uah_mwh",
        "distribution_tariff_uah_per_mwh": "distribution_uah_mwh",
        "transmission_tariff_uah_per_mwh": "transmission_uah_mwh", "FX UAH/USD": "fx_uah_per_usd"}


def delivery_day(ts: pd.Series) -> pd.Series:
    """Delivery day of hour-ending timestamps: 00:00 belongs to the previous day."""
    return (ts - pd.Timedelta(hours=1)).dt.normalize()


def nearest_day(missing: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp:
    """Nearest delivery day present in the workbook; the earlier one on a tie."""
    dist = np.abs((days - missing).days)
    return days[dist == dist.min()].min()


def check_gross(src: pd.DataFrame) -> float:
    """The workbook's gross price must be E + O + P in every hour it has; returns the largest gap."""
    gap = (src["gross_price_uah_per_mwh"] - (src["price_uah_per_mwh"] + src["distribution_tariff_uah_per_mwh"]
                                             + src["transmission_tariff_uah_per_mwh"])).abs().max()
    if not gap <= GROSS_TOL:
        raise ValueError(f"workbook gross price differs from E + O + P by up to {gap} UAH/MWh")
    return gap


def build(source_xlsx: Path = SOURCE_XLSX, model_xlsx: Path = MODEL_XLSX) -> pd.DataFrame:
    src = pd.read_excel(source_xlsx, sheet_name="DAM", usecols=["Timestamp", "Date", *_SRC])
    model = pd.read_excel(model_xlsx, usecols=["timestamp", "DAM_timestamp", "DAM_net_price_usd_kwh"])
    check_gross(src)
    by_day = src.groupby(pd.to_datetime(src["Date"]).dt.normalize()).agg(
        fx=("FX UAH/USD", "first"), O=("distribution_tariff_uah_per_mwh", "first"),
        P=("transmission_tariff_uah_per_mwh", "first"))

    m = model.merge(src, left_on="DAM_timestamp", right_on="Timestamp", how="left", validate="one_to_one")
    out = m[["timestamp", "DAM_timestamp"]].assign(**{new: m[old] for old, new in _SRC.items()})
    filled = m["price_uah_per_mwh"].isna()
    day = delivery_day(m["DAM_timestamp"])
    fill_day = pd.Series(pd.NaT, index=m.index, dtype="datetime64[ns]")
    for d in sorted(set(day[filled])):
        fill_day[filled & (day == d)] = d if d in by_day.index else nearest_day(d, by_day.index)

    ref = by_day.loc[fill_day[filled]].set_axis(m.index[filled])
    E = m.loc[filled, "DAM_net_price_usd_kwh"] * 1000 * ref["fx"]
    out.loc[filled, "DAM_net_price_uah_mwh"] = E
    out.loc[filled, "distribution_uah_mwh"] = ref["O"]
    out.loc[filled, "transmission_uah_mwh"] = ref["P"]
    out.loc[filled, "DAM_gross_price_uah_mwh"] = E + ref["O"] + ref["P"]
    out.loc[filled, "fx_uah_per_usd"] = ref["fx"]
    out["filled"] = filled
    out["fill_day"] = fill_day.dt.strftime("%Y-%m-%d")   # empty for hours taken from the workbook
    return out[COLUMNS]


def main() -> None:
    df = build()
    df.to_csv(OUT_CSV, index=False, lineterminator="\n")
    f = df[df["filled"]]
    print(f"wrote {OUT_CSV.relative_to(ROOT)}: {len(df)} rows, {len(f)} filled "
          f"(fill days: {sorted({str(d) for d in f['fill_day']})})")


if __name__ == "__main__":
    main()
