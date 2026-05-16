"""Distribution-histogram panels (raw level + monthly %-change) for part_3.

Exposes `build_distribution_section(returns_df=None) -> str`, which returns
a self-contained block of HTML (Plotly figures + summary stats) that
part_3.py embeds inside its `<body>` just before `</body>`. Each model
input is shown as a paired row: raw level on the left, monthly %-change
on the right. Policy-rate reference series follow in their own section.
Running this file as a script delegates to part_3.py so the user can run
either entry point and get the same complete report.
"""

import math
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.io import to_html
from statsmodels.tsa.stattools import acf as _acf, pacf as _pacf


ROOT = Path(__file__).parent


# Paired model inputs: each row shows the raw level (left) next to its
# monthly %-change (right). The raw level uses the full source CSV so the
# observation count matches the file; the %-change side is on the
# inner-joined sample used by the regression, so its n is smaller.
# EUR/ZAR has no single CSV — derived as ZAR_USD × USD_EUR (ZAR per USD
# times USD per EUR → ZAR per EUR), so its raw level falls back to the
# joined level series.
#
# (raw_title, raw_units, raw_source_csv_or_None, raw_col, derived_level_col,
#  ret_col, ret_title)
PAIRED_SERIES = [
    ("USD/ZAR",             "USD/ZAR rate", "USD_ZAR.csv",             "EXSFUS",
     None,            "r_USD/ZAR", "Monthly %Δ USD/ZAR"),
    ("EUR/ZAR",             "EUR/ZAR rate", None,                       None,
     "ZAR_EUR",       "r_EUR/ZAR", "Monthly %Δ EUR/ZAR"),
    ("JSE Top 40 close",    "index level",  "ZA_TOP40_index.csv",      "Close",
     None,            "r_TOP40",   "Monthly %Δ JSE Top 40"),
    ("ZA–US trade balance", "USD millions", "ZA_US_trade_balance.csv", "balance",
     None,            "r_TBAL",    "Monthly %Δ |Trade balance|"),
]

# Reference series (policy rates) — not regression inputs, shown for context.
POLICY_RATES = [
    ("ECB_policy_rate.csv", "ECBMRRFR", "ECB Main Refinancing Rate", "% p.a."),
    ("US_policy_rate.csv",  "FEDFUNDS", "US Federal Funds Rate",     "% p.a."),
]


INJECTED_STYLE = """
<style>
  .dist-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;
              align-items: start; margin: 1rem 0 2rem;
              padding-top: 1rem; border-top: 1px solid #e6e6e6; }
  .dist-row .dist-panel { margin: 0; padding: 0; border-top: none; }
  .dist-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr);
                 gap: 1.5rem; align-items: start; margin: 1rem 0 2rem;
                 padding-top: 1rem; border-top: 1px solid #e6e6e6; }
  .dist-grid-3 .dist-panel { margin: 0; padding: 0; border-top: none; }
  @media (max-width: 900px) {
    .dist-row, .dist-grid-3 { grid-template-columns: 1fr; }
  }
  .dist-panel { margin: 1rem 0 2rem; padding: 1rem 0;
                border-top: 1px solid #e6e6e6; }
  .dist-panel dl.stats { display: flex; flex-wrap: wrap; gap: 0.4rem 1.25rem;
                         margin: 0.5rem 0 0 50px; font-size: 0.85rem; color: #444; }
  .dist-panel dl.stats .stat { display: flex; align-items: baseline;
                               gap: 0.35rem; white-space: nowrap; }
  .dist-panel dl.stats dt { font-weight: 600; color: #666; }
  .dist-panel dl.stats dd { margin: 0; font-variant-numeric: tabular-nums; }
</style>
"""


def load_simple(path: Path, col: str) -> pl.Series:
    return pl.read_csv(path)[col].cast(pl.Float64).drop_nulls()


def load_historical_rate(path: Path) -> pl.Series:
    # SARB_policy_rate.csv has a two-line preamble then a blank line, then
    # the real Date,Value section. Skip past the preamble and re-parse.
    text = path.read_text()
    body = text.split("\n\n", 1)[1]
    df = pl.read_csv(body.encode(), schema_overrides={"Date": pl.Utf8, "Value": pl.Float64})
    return df["Value"].drop_nulls()


def load_series(path: Path, date_col: str, value_col: str, name: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    return df.select(
        pl.col(date_col).cast(pl.Date).alias("date"),
        pl.col(value_col).cast(pl.Float64).alias(name),
    ).sort("date")


def build_levels(root: Path = ROOT) -> pl.DataFrame:
    """Aligned monthly level series (the basis for both raw and %-change plots)."""
    zar_usd   = load_series(root / "USD_ZAR.csv",             "observation_date", "EXSFUS",  "ZAR_USD")
    usd_eur   = load_series(root / "EUR_USD.csv",             "observation_date", "EXUSEU",  "USD_EUR")
    top40     = load_series(root / "ZA_TOP40_index.csv",      "Date",             "Close",   "TOP40")
    trade_bal = load_series(root / "ZA_US_trade_balance.csv", "date",             "balance", "TRADE_BAL_USD")
    return (
        zar_usd
        .join(usd_eur,   on="date", how="inner")
        .join(top40,     on="date", how="inner")
        .join(trade_bal, on="date", how="inner")
        .with_columns((pl.col("ZAR_USD") * pl.col("USD_EUR")).alias("ZAR_EUR"))
        .sort("date")
    )


def build_returns(root: Path = ROOT) -> pl.DataFrame:
    """Aligned monthly %-changes that match the regression inputs in part_3.py."""
    levels = build_levels(root)
    return levels.select(
        pl.col("date"),
        (pl.col("ZAR_USD").pct_change() * 100).alias("r_USD/ZAR"),
        (pl.col("ZAR_EUR").pct_change() * 100).alias("r_EUR/ZAR"),
        (pl.col("TOP40").pct_change()   * 100).alias("r_TOP40"),
        (pl.col("TRADE_BAL_USD").abs().pct_change() * 100).alias("r_TBAL"),
    ).drop_nulls()


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
    # Vlines mark position; values go into a corner box so the labels can't
    # collide when mean ≈ median (common for flat policy-rate windows).
    fig.add_vline(x=mean,   line=dict(color="#d6452c", width=2))
    fig.add_vline(x=median, line=dict(color="#2c8a3a", width=2, dash="dash"))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        text=(f"<span style='color:#d6452c'>mean&nbsp;{mean:.3g}</span><br>"
              f"<span style='color:#2c8a3a'>median&nbsp;{median:.3g}</span>"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d0d0d0", borderwidth=1, borderpad=4,
        font=dict(size=11),
    )
    fig.update_layout(
        title=dict(text=f"{title}  (n={len(values)}, σ={std:.3g})", x=0.02, xanchor="left"),
        xaxis_title=units,
        yaxis_title="frequency",
        bargap=0.02,
        template="plotly_white",
        height=360,
        margin=dict(l=55, r=20, t=55, b=45),
    )
    return fig


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
    return f"<div class='dist-panel'>{chart_html}{stats}</div>"


def _stem_fig(lags: list[int], vals, ci: float, title: str, ylabel: str) -> go.Figure:
    """Stem-style bar plot used for ACF/PACF panels."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags, y=vals, width=0.18,
        marker=dict(color="#3b6fb6"),
        hovertemplate=f"lag %{{x}}<br>{ylabel} = %{{y:.3f}}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=lags, y=vals, mode="markers",
        marker=dict(size=7, color="#1f3a68"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.add_hline(y=ci,  line=dict(color="#d6452c", width=1, dash="dash"))
    fig.add_hline(y=-ci, line=dict(color="#d6452c", width=1, dash="dash"))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(title="lag", dtick=1, tick0=1),
        yaxis_title=ylabel,
        template="plotly_white",
        height=320,
        margin=dict(l=55, r=20, t=50, b=45),
    )
    return fig


def acf_pacf_pair(resid, label: str, n_lags: int = 15) -> tuple[go.Figure, go.Figure]:
    """Return (ACF, PACF) figures for an array-like residual series.

    Confidence band is the white-noise null ±1.96/√n, so bars piercing the
    band are evidence of significant autocorrelation at that lag.
    """
    n = len(resid)
    ci = 1.96 / math.sqrt(n)
    acf_vals = _acf(resid, nlags=n_lags, fft=False)
    pacf_vals = _pacf(resid, nlags=n_lags)
    lags = list(range(1, n_lags + 1))
    return (
        _stem_fig(lags, list(acf_vals[1:]),  ci,
                  f"ACF — {label}  (n={n}, band ±{ci:.3f})",  "ACF"),
        _stem_fig(lags, list(pacf_vals[1:]), ci,
                  f"PACF — {label}  (n={n}, band ±{ci:.3f})", "PACF"),
    )


def build_residual_acf_section(residuals_by_label: list[tuple[str, object]],
                               include_plotly_js: bool = True) -> str:
    """Return HTML with one row per residual series: ACF (left) + PACF (right)."""
    rows: list[str] = []
    include_js = include_plotly_js
    for label, resid in residuals_by_label:
        acf_fig, pacf_fig = acf_pacf_pair(resid, label)
        acf_html = to_html(
            acf_fig,
            include_plotlyjs="cdn" if include_js else False,
            full_html=False,
            config={"displaylogo": False, "responsive": True},
        )
        include_js = False
        pacf_html = to_html(
            pacf_fig, include_plotlyjs=False, full_html=False,
            config={"displaylogo": False, "responsive": True},
        )
        rows.append(
            "<div class='dist-row'>"
            f"<div class='dist-panel'>{acf_html}</div>"
            f"<div class='dist-panel'>{pacf_html}</div>"
            "</div>"
        )
    return (
        INJECTED_STYLE
        + "<h2>Residual autocorrelation — ACF / PACF</h2>"
        + "<p>One row per OLS specification. Bars show the sample "
          "autocorrelation (left) and partial autocorrelation (right) at each "
          "lag; red dashed lines mark the ±1.96/√n white-noise band. Bars "
          "piercing the band are evidence of significant autocorrelation at "
          "that lag.</p>"
        + "".join(rows)
    )


def build_distribution_section(returns_df: pl.DataFrame | None = None,
                               root: Path = ROOT,
                               include_plotly_js: bool = True) -> str:
    """Return HTML for the distribution sections.

    Layout: each model-input series occupies one row with two columns —
    raw level on the left, monthly %-change on the right. Policy-rate
    reference series follow as a separate 3-up section. Includes its own
    <style> block and loads Plotly.js from CDN on the first figure.
    """
    if returns_df is None:
        returns_df = build_returns(root)
    levels = build_levels(root)

    rows: list[str] = []
    include_js = include_plotly_js
    for raw_title, raw_units, raw_csv, raw_col, lvl_col, ret_col, ret_title in PAIRED_SERIES:
        if raw_csv is not None:
            raw_s = load_simple(root / raw_csv, raw_col)
            raw_src = raw_csv
        else:
            raw_s = levels[lvl_col].drop_nulls()
            raw_src = "derived: USD/ZAR × EUR/USD (joined sample)"
        ret_s = returns_df[ret_col]
        raw_block = render_block(panel(raw_s, raw_title, raw_units),
                                 raw_src, raw_s, include_js=include_js)
        include_js = False
        ret_block = render_block(panel(ret_s, ret_title, "% change"),
                                 f"returns[{ret_col}]", ret_s, include_js=False)
        rows.append(f"<div class='dist-row'>{raw_block}{ret_block}</div>")

    policy_blocks: list[str] = []
    for fname, col, title, units in POLICY_RATES:
        s = load_simple(root / fname, col)
        policy_blocks.append(render_block(panel(s, title, units),
                                          fname, s, include_js=False))
    sarb = load_historical_rate(root / "SARB_policy_rate.csv")
    policy_blocks.append(render_block(panel(sarb, "SARB Policy Rate", "% p.a."),
                                      "SARB_policy_rate.csv", sarb, include_js=False))

    return (
        INJECTED_STYLE
        + "<h2>Distribution plots — level vs. monthly %-change</h2>"
        + "<p>Each row pairs a regression-input series with its monthly"
          " percentage-change form. Mean shown in red, median in green.</p>"
        + "".join(rows)
        + "<h2>Reference series — policy rates</h2>"
        + "<p>Not used in the regression; included for context on the rate"
          " environment over the sample.</p>"
        + f"<div class='dist-grid-3'>{''.join(policy_blocks)}</div>"
    )


if __name__ == "__main__":
    # part_3.py handles the full pipeline (analysis + distribution plots),
    # so delegate to it. Running either script produces the same part_3.html.
    subprocess.run(
        [sys.executable, str(ROOT / "part_3.py")],
        cwd=ROOT, check=True,
    )
