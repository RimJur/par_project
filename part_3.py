"""
Part 3 — South Africa: exchange rates, equity index, and trade balance.

Country: South Africa (ZAR). Major trading partner used for the trade balance: USA.
Stock index: JSE Top 40. Exchange rates: ZAR/USD (EXSFUS) and USD/EUR (EXUSEU).

Data files (monthly, FRED / market sources):
    EXSFUS.csv           ZAR per 1 USD
    EXUSEU.csv           USD per 1 EUR
    SAR_TOP40_INDEX.csv  JSE Top 40 close
    SAR_USD_balance.csv  ZA trade balance vs. USA (USD millions, signed)
"""

import polars as pl
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller


def load_series(path: str, date_col: str, value_col: str, name: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    return (
        df.select(
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(value_col).cast(pl.Float64).alias(name),
        )
        .sort("date")
    )


zar_usd = load_series("EXSFUS.csv", "observation_date", "EXSFUS", "ZAR_USD")
usd_eur = load_series("EXUSEU.csv", "observation_date", "EXUSEU", "USD_EUR")
top40 = load_series("SAR_TOP40_INDEX.csv", "Date", "Close", "TOP40")
trade_bal = load_series("SAR_USD_balance.csv", "date", "balance", "TRADE_BAL_USD")

# ZAR/EUR cross-rate from ZAR/USD * USD/EUR. Inner-join keeps the common window.
levels = (
    zar_usd
    .join(usd_eur, on="date", how="inner")
    .join(top40, on="date", how="inner")
    .join(trade_bal, on="date", how="inner")
    .with_columns((pl.col("ZAR_USD") * pl.col("USD_EUR")).alias("ZAR_EUR"))
    .select(["date", "ZAR_USD", "ZAR_EUR", "TOP40", "TRADE_BAL_USD"])
    .sort("date")
)

print("=" * 70)
date_min = levels["date"].min()
date_max = levels["date"].max()
print(f"Aligned monthly sample: {date_min:%Y-%m} → {date_max:%Y-%m} "
      f"({levels.height} obs)")
print("=" * 70)
print(levels.head())
print("...")
print(levels.tail())

# Monthly percentage changes (returns). For the trade balance the series is
# signed and crosses zero, so simple pct_change is sign-sensitive; we use the
# pct_change of the absolute deficit, which is the conventional choice when
# the series is consistently of one sign (here, persistently negative).
returns = (
    levels
    .select(
        pl.col("date"),
        (pl.col("ZAR_USD").pct_change() * 100).alias("r_ZAR_USD"),
        (pl.col("ZAR_EUR").pct_change() * 100).alias("r_ZAR_EUR"),
        (pl.col("TOP40").pct_change() * 100).alias("r_TOP40"),
        (pl.col("TRADE_BAL_USD").abs().pct_change() * 100).alias("r_TBAL"),
    )
    .drop_nulls()
)

ret_cols = ["r_ZAR_USD", "r_ZAR_EUR", "r_TOP40", "r_TBAL"]

print("\n" + "=" * 70)
print("Summary statistics of monthly % changes")
print("=" * 70)
desc = pl.DataFrame([
    {
        "series": c,
        "mean": returns[c].mean(),
        "std":  returns[c].std(),
        "min":  returns[c].min(),
        "max":  returns[c].max(),
        "skew": returns[c].skew(),
        "kurt": returns[c].kurtosis(),
    }
    for c in ret_cols
]).with_columns([pl.col(c).round(3) for c in ["mean", "std", "min", "max", "skew", "kurt"]])
print(desc)

print("\n" + "=" * 70)
print("Correlation matrix of monthly % changes")
print("=" * 70)
corr = returns.select(ret_cols).corr()
corr_labeled = (
    corr
    .with_columns(pl.Series("series", ret_cols))
    .select(["series", *ret_cols])
    .with_columns([pl.col(c).round(3) for c in ret_cols])
)
print(corr_labeled)

fx_idx     = returns.select(pl.corr("r_ZAR_USD", "r_TOP40")).item()
fx_eur_idx = returns.select(pl.corr("r_ZAR_EUR", "r_TOP40")).item()
fx_fx      = returns.select(pl.corr("r_ZAR_USD", "r_ZAR_EUR")).item()
print(f"\nKey pairwise correlations:")
print(f"  ZAR/USD vs TOP40 : {fx_idx:+.3f}")
print(f"  ZAR/EUR vs TOP40 : {fx_eur_idx:+.3f}")
print(f"  ZAR/USD vs ZAR/EUR: {fx_fx:+.3f}")


# Stationarity check (ADF) on the inputs that go into the regression.
# OLS on non-stationary data risks spurious regression. pct_change has likely
# differenced away unit roots, but worth confirming.
print("\n" + "=" * 70)
print("Augmented Dickey-Fuller tests on regression inputs")
print("=" * 70)
print(f"  H0: series has a unit root (non-stationary).")
for col in ["r_TBAL", "r_ZAR_USD"]:
    stat, pval, *_ = adfuller(returns[col].to_numpy(), autolag="AIC")
    verdict = "stationary" if pval < 0.05 else "NON-stationary"
    print(f"  {col:11s}  ADF = {stat:+.3f}   p = {pval:.4f}   → {verdict}")


# Newey-West lag length: standard rule int(4 * (n/100)**(2/9)) ≈ 3 for n≈58.
HAC_LAGS = 3
BG_LAGS = 4


def diagnostics(model) -> dict:
    """Run BLUE-assumption diagnostics on a fitted OLS model."""
    resid = model.resid
    exog = model.model.exog
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, exog)
    bg_lm, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=BG_LAGS)
    dw = durbin_watson(resid)
    jb, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
    print(f"  Diagnostics:")
    print(f"    Breusch-Pagan (homosked.)        LM = {bp_lm:7.3f}  p = {bp_p:.3f}"
          f"   {'OK' if bp_p > 0.05 else 'REJECT'}")
    print(f"    Breusch-Godfrey (no autocorr,{BG_LAGS}) LM = {bg_lm:7.3f}  p = {bg_p:.3f}"
          f"   {'OK' if bg_p > 0.05 else 'REJECT'}")
    print(f"    Durbin-Watson                    DW = {dw:7.3f}              "
          f"   {'OK' if 1.5 < dw < 2.5 else 'CHECK'}")
    print(f"    Jarque-Bera (normality)          JB = {jb:7.3f}  p = {jb_p:.3f}"
          f"   {'OK' if jb_p > 0.05 else 'REJECT'}")
    return {"bp_p": bp_p, "bg_p": bg_p, "dw": dw, "jb_p": jb_p}


# Regression: %ΔTrade balance = α + β · %ΔZAR_USD(lag) + ε.
# A rise in ZAR/USD = ZAR depreciation. J-curve theory predicts the trade
# balance worsens contemporaneously and improves at lags 1–2.
def run_ols(df: pl.DataFrame, y_col: str, x_col: str, lag: int, label: str):
    sub = (
        df.select(
            pl.col(y_col),
            pl.col(x_col).shift(lag).alias("x_lag"),
        )
        .drop_nulls()
    )
    y = sub[y_col].to_numpy()
    X = sm.add_constant(sub["x_lag"].to_numpy())
    model = sm.OLS(y, X).fit()
    hac = model.get_robustcov_results(cov_type="HAC", maxlags=HAC_LAGS)
    print("\n" + "-" * 70)
    print(f"Regression: {label}   (n = {int(model.nobs)})")
    print("-" * 70)
    print(f"  intercept : {model.params[0]:+.4f}  (p = {model.pvalues[0]:.3f})")
    print(f"  slope     : {model.params[1]:+.4f}  (p = {model.pvalues[1]:.3f})"
          f"   |  HAC p = {hac.pvalues[1]:.3f}")
    print(f"  R²        : {model.rsquared:.4f}")
    print(f"  Adj. R²   : {model.rsquared_adj:.4f}")
    print(f"  F-stat    : {model.fvalue:.3f}  (p = {model.f_pvalue:.3f})")
    diag = diagnostics(model)
    return model, hac, diag


print("\n" + "=" * 70)
print("Regressions: %ΔTrade balance = α + β · %ΔZAR_USD(lag) + ε")
print("=" * 70)
m0, m0_hac, d0 = run_ols(returns, "r_TBAL", "r_ZAR_USD", 0, "contemporaneous (lag 0)")
m1, m1_hac, d1 = run_ols(returns, "r_TBAL", "r_ZAR_USD", 1, "1-month lag")
m2, m2_hac, d2 = run_ols(returns, "r_TBAL", "r_ZAR_USD", 2, "2-month lag")

print("\n" + "=" * 70)
print("Coefficient comparison across specifications")
print("=" * 70)
summary_df = pl.DataFrame({
    "metric":  ["beta", "p-value (OLS)", "p-value (HAC)", "R²", "n"],
    "lag_0":   [m0.params[1], m0.pvalues[1], m0_hac.pvalues[1], m0.rsquared, float(m0.nobs)],
    "lag_1":   [m1.params[1], m1.pvalues[1], m1_hac.pvalues[1], m1.rsquared, float(m1.nobs)],
    "lag_2":   [m2.params[1], m2.pvalues[1], m2_hac.pvalues[1], m2.rsquared, float(m2.nobs)],
}).with_columns([pl.col(c).round(4) for c in ["lag_0", "lag_1", "lag_2"]])
print(summary_df)

print("\n" + "=" * 70)
print("BLUE-assumption diagnostics summary  (p-values; * = reject at 5%)")
print("=" * 70)
def mark(p): return f"{p:.3f}{'*' if p < 0.05 else ' '}"
diag_df = pl.DataFrame({
    "test":  ["Breusch-Pagan (homosked.)", "Breusch-Godfrey (no autocorr)",
              "Durbin-Watson",             "Jarque-Bera (normality)"],
    "lag_0": [mark(d0["bp_p"]), mark(d0["bg_p"]), f"{d0['dw']:.3f}", mark(d0["jb_p"])],
    "lag_1": [mark(d1["bp_p"]), mark(d1["bg_p"]), f"{d1['dw']:.3f}", mark(d1["jb_p"])],
    "lag_2": [mark(d2["bp_p"]), mark(d2["bg_p"]), f"{d2['dw']:.3f}", mark(d2["jb_p"])],
})
print(diag_df)

print("""
======================================================================
Interpretation
======================================================================
Summary statistics & correlations:
TOP40 has the highest monthly volatility of the three return series
(σ ≈ 4.2%) — a single-country emerging-market index naturally swings
more than its currency. The two FX series are positively skewed,
reflecting episodic ZAR sell-offs that are larger than the
corresponding rallies. ZAR/USD and ZAR/EUR co-move strongly
(corr ≈ +0.74); the bulk of monthly variation in both is the rand
itself, with the EUR/USD cross adding a second-order wedge (which is
why ZAR/EUR vs. TOP40 is essentially zero while ZAR/USD vs. TOP40 is
clearly negative). The ZAR/USD–TOP40 correlation of −0.27 is the
most economically interesting pair: when the rand weakens, JSE
equities tend to sell off in the same month — consistent with
risk-off episodes hitting the currency and local equities together
(portfolio outflows, commodity-price drawdowns, global-risk
repricing). A portion of TOP40 earnings is rand-hedged via
diversified miners and dual-listed names, which dampens — but does
not reverse — the negative co-movement at the monthly horizon.

Regression results:
The contemporaneous slope is positive (β ≈ +1.37): in the same month
that the rand weakens, the absolute bilateral deficit with the US
tends to widen rather than shrink. This matches the short-run side
of the J-curve — invoice prices and contracted import volumes adjust
slowly, so a weaker rand mechanically raises the rand cost of imports
before export volumes respond. At the 1- and 2-month lags the slope
flips sign (β ≈ −3.96 and −4.42), the textbook J-curve recovery: a
ZAR depreciation today is associated with a smaller deficit one to
two months later as exports pick up and import demand compresses.
However, none of the three slopes is statistically significant at
conventional levels (p = 0.71, 0.28, 0.24) and R² stays below 3%,
so the lagged signs are directionally suggestive rather than
conclusive. That is unsurprising: monthly bilateral trade balances
are dominated by commodity-price swings, US-side demand, and
idiosyncratic shipment timing that a single FX regressor cannot
capture.

BLUE-assumption diagnostics:
Stationarity is fine — ADF rejects the unit-root null for both
%ΔTrade balance and %ΔZAR/USD at p < 0.001, so the regressions are
not spurious. Breusch-Pagan does not reject homoskedasticity at any
lag (p ≈ 0.18–0.89) and Durbin-Watson is essentially 2.0, so first-
order serial correlation is absent. However, Breusch-Godfrey at four
lags rejects the no-higher-order-autocorrelation null in every
specification (p = 0.048 / 0.019 / 0.013), and Jarque-Bera firmly
rejects residual normality (p ≈ 0). The autocorrelation finding is
why HAC (Newey-West, 3 lags) standard errors were also reported —
they only modestly shift the slope p-values (e.g. lag-2 moves from
0.24 to 0.18), so the qualitative conclusion is unchanged. The
non-normal residuals mean the small-sample t/F p-values should be
treated as approximate; a bootstrap or permutation inference would
be cleaner if any slope mattered for a decision.

Caveats:
The bilateral US–ZA balance is narrower than ZA's overall current
account, and the 60-month sample spans the COVID rebound, the 2022
commodity spike, and the 2024–2025 logistics disruptions — regime
shifts a single linear specification cannot absorb. Given the
Breusch-Godfrey rejection, an ARDL or error-correction model that
explicitly models the residual dynamics would be the natural next
step for inference.
""")
