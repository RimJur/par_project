"""Render a distribution (histogram) plot per CSV data file into part_3.html."""

import html as html_lib
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.io import to_html


ROOT = Path(__file__).parent
OUT = ROOT / "part_3.html"


# (csv path, value column, human-readable title, units)
DATASETS = [
    ("ECBMRRFR.csv",          "ECBMRRFR", "ECB Main Refinancing Rate",          "% p.a."),
    ("EXSFUS.csv",            "EXSFUS",   "ZAR per 1 USD (EXSFUS)",             "ZAR / USD"),
    ("EXUSEU.csv",            "EXUSEU",   "USD per 1 EUR (EXUSEU)",             "USD / EUR"),
    ("FEDFUNDS.csv",          "FEDFUNDS", "US Federal Funds Rate (FEDFUNDS)",   "% p.a."),
    ("SAR_TOP40_INDEX.csv",   "Close",    "JSE Top 40 — monthly close",         "index level"),
    ("SAR_USD_balance.csv",   "balance",  "ZA–US trade balance",                "USD millions"),
]


def load_simple(path: Path, col: str) -> pl.Series:
    return pl.read_csv(path)[col].cast(pl.Float64).drop_nulls()


def load_historical_rate(path: Path) -> pl.Series:
    # File has a two-line preamble (Indicator/Description) then a blank line,
    # then a real Date,Value section. Skip past the preamble and re-parse.
    text = path.read_text()
    body = text.split("\n\n", 1)[1]
    df = pl.read_csv(body.encode(), schema_overrides={"Date": pl.Utf8, "Value": pl.Float64})
    return df["Value"].drop_nulls()


def panel(series: pl.Series, title: str, units: str) -> go.Figure:
    values = series.to_numpy()
    mean = float(series.mean())
    median = float(series.median())
    std = float(series.std())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=30,
        marker=dict(color="#3b6fb6", line=dict(color="#1f3a68", width=0.5)),
        opacity=0.85,
        name="count",
        hovertemplate="bin: %{x}<br>count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=mean,   line=dict(color="#d6452c", width=2),
                  annotation_text=f"mean {mean:.3g}",   annotation_position="top right")
    fig.add_vline(x=median, line=dict(color="#2c8a3a", width=2, dash="dash"),
                  annotation_text=f"median {median:.3g}", annotation_position="top left")
    fig.update_layout(
        title=dict(text=f"{title}  (n={len(values)}, σ={std:.3g})", x=0.02, xanchor="left"),
        xaxis_title=units,
        yaxis_title="frequency",
        bargap=0.02,
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def load_series(path: Path, date_col: str, value_col: str, name: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    return df.select(
        pl.col(date_col).cast(pl.Date).alias("date"),
        pl.col(value_col).cast(pl.Float64).alias(name),
    ).sort("date")


def build_returns() -> pl.DataFrame:
    """Replicate the model-input series from part_3.py: aligned monthly % changes."""
    zar_usd   = load_series(ROOT / "EXSFUS.csv",          "observation_date", "EXSFUS", "ZAR_USD")
    usd_eur   = load_series(ROOT / "EXUSEU.csv",          "observation_date", "EXUSEU", "USD_EUR")
    top40     = load_series(ROOT / "SAR_TOP40_INDEX.csv", "Date",             "Close",  "TOP40")
    trade_bal = load_series(ROOT / "SAR_USD_balance.csv", "date",             "balance","TRADE_BAL_USD")

    levels = (
        zar_usd
        .join(usd_eur,   on="date", how="inner")
        .join(top40,     on="date", how="inner")
        .join(trade_bal, on="date", how="inner")
        .with_columns((pl.col("ZAR_USD") * pl.col("USD_EUR")).alias("ZAR_EUR"))
        .sort("date")
    )
    return levels.select(
        pl.col("date"),
        (pl.col("ZAR_USD").pct_change() * 100).alias("r_ZAR_USD"),
        (pl.col("ZAR_EUR").pct_change() * 100).alias("r_ZAR_EUR"),
        (pl.col("TOP40").pct_change()   * 100).alias("r_TOP40"),
        (pl.col("TRADE_BAL_USD").abs().pct_change() * 100).alias("r_TBAL"),
    ).drop_nulls()


RETURN_PANELS = [
    ("r_ZAR_USD", "Monthly %Δ ZAR/USD", "% change"),
    ("r_ZAR_EUR", "Monthly %Δ ZAR/EUR", "% change"),
    ("r_TOP40",   "Monthly %Δ JSE Top 40", "% change"),
    ("r_TBAL",    "Monthly %Δ |Trade balance|", "% change"),
]


def run_part_3() -> str:
    """Execute part_3.py and capture its textual output."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "part_3.py")],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    return result.stdout


def render_block(fig: go.Figure, source_label: str, series: pl.Series, include_js: bool) -> str:
    chart_html = to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    pairs = [
        ("source", f"<code>{source_label}</code>"),
        ("n",      str(len(series))),
        ("mean",   f"{float(series.mean()):.4g}"),
        ("median", f"{float(series.median()):.4g}"),
        ("std",    f"{float(series.std()):.4g}"),
        ("min",    f"{float(series.min()):.4g}"),
        ("max",    f"{float(series.max()):.4g}"),
    ]
    stats = "<dl class='stats'>" + "".join(
        f"<div class='stat'><dt>{k}</dt><dd>{v}</dd></div>" for k, v in pairs
    ) + "</dl>"
    return f"<section>{chart_html}{stats}</section>"


def main() -> None:
    analysis_text = run_part_3()

    raw_panels: list[tuple[str, go.Figure, pl.Series]] = []
    for fname, col, title, units in DATASETS:
        series = load_simple(ROOT / fname, col)
        raw_panels.append((fname, panel(series, title, units), series))

    hr_series = load_historical_rate(ROOT / "HistoricalRateDetail.csv")
    raw_panels.append((
        "HistoricalRateDetail.csv",
        panel(hr_series, "SARB Policy Rate (HistoricalRateDetail)", "% p.a."),
        hr_series,
    ))

    returns = build_returns()
    ret_panels: list[tuple[str, go.Figure, pl.Series]] = []
    for col, title, units in RETURN_PANELS:
        s = returns[col]
        ret_panels.append((f"returns[{col}]", panel(s, title, units), s))

    raw_blocks: list[str] = []
    for i, (lbl, fig, s) in enumerate(raw_panels):
        raw_blocks.append(render_block(fig, lbl, s, include_js=(i == 0)))

    ret_blocks: list[str] = [
        render_block(fig, lbl, s, include_js=False) for lbl, fig, s in ret_panels
    ]

    blocks_raw_html = "".join(raw_blocks)
    blocks_ret_html = "".join(ret_blocks)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Part 3 — Distribution plots</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  p.lede {{ color: #555; margin-top: 0; }}
  section {{ margin: 2rem 0 3rem; padding: 1rem 0;
             border-top: 1px solid #e6e6e6; }}
  dl.stats {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.75rem;
              margin: 0.5rem 0 0 60px; font-size: 0.9rem; color: #444; }}
  dl.stats .stat {{ display: flex; align-items: baseline; gap: 0.4rem;
                    white-space: nowrap; }}
  dl.stats dt {{ font-weight: 600; color: #666; }}
  dl.stats dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
  code {{ background: #f4f4f4; padding: 0.1rem 0.35rem; border-radius: 3px; }}
  pre.analysis {{ background: #0f1115; color: #e8e8e8; padding: 1rem 1.25rem;
                  border-radius: 6px; overflow-x: auto; font-size: 0.82rem;
                  line-height: 1.35; white-space: pre;
                  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  h2 {{ margin-top: 2.5rem; }}
</style>
</head>
<body>
<h1>Part 3 — South Africa: exchange rates, equity index, trade balance</h1>
<p class="lede">Regression analysis (text output from <code>part_3.py</code>) followed by distribution histograms of (a) the raw input series and (b) the monthly %-change series used by the model. Mean shown in red, median in green.</p>
<h2>Analysis output</h2>
<pre class="analysis">{html_lib.escape(analysis_text)}</pre>
<h2>Distribution plots — raw data files</h2>
{blocks_raw_html}
<h2>Distribution plots — monthly % changes (model inputs)</h2>
{blocks_ret_html}
</body>
</html>
"""
    OUT.write_text(html)
    total = len(raw_panels) + len(ret_panels)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes, "
          f"{len(raw_panels)} raw + {len(ret_panels)} returns = {total} panels)")


if __name__ == "__main__":
    main()
