#Module preparation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load raw data from CSV files
zar_usd = pd.read_csv("USD_ZAR.csv", parse_dates=['observation_date'])
zar_usd.columns = ["Date", 'ZAR_USD']
zar_usd = zar_usd.sort_values("Date")

usd_eur = pd.read_csv("EUR_USD.csv", parse_dates=['observation_date'])
usd_eur.columns = ["Date", 'USD_EUR']
usd_eur = usd_eur.sort_values("Date")
usd_eur['ZAR_EUR'] = zar_usd["ZAR_USD"].values * usd_eur['USD_EUR'].values

fed = pd.read_csv("US_policy_rate.csv", parse_dates=['observation_date'])
fed.columns = ["Date", 'FED']
fed = fed.sort_values("Date")

sarb = pd.read_csv("SARB_policy_rate.csv", skiprows=3, parse_dates=['Date'])
sarb.columns = ["Date", 'SARB']
sarb = sarb.sort_values("Date")
sarb = sarb.groupby(sarb['Date'].dt.to_period('M'))['SARB'].first().reset_index()
sarb['Date'] = sarb['Date'].dt.to_timestamp('M')

ecb = pd.read_csv("ECB_policy_rate.csv", parse_dates=['observation_date'])
ecb.columns = ['Date', 'ECB_Rate']
ecb = ecb.sort_values('Date')
ecb = ecb.groupby(ecb['Date'].dt.to_period('M'))['ECB_Rate'].first().reset_index()
ecb['Date'] = ecb['Date'].dt.to_timestamp('M')

print("ZAR_USD:", zar_usd.shape, "\n", zar_usd.head(3))
print("USD_EUR:", usd_eur.shape, "\n", usd_eur.head(3))
print("Fed:", fed.shape, "\n", fed.head(3))
print("ECB:", ecb.shape, "\n", ecb.head(3))
print("SARB:", sarb.shape, "\n", sarb.head(3))

# Standardise dates to month-start and filter to 61-month sample (Mar 2021 – Mar 2026)
zar_usd['Date'] = zar_usd['Date'].dt.to_period('M').dt.to_timestamp()
usd_eur['Date'] = usd_eur['Date'].dt.to_period('M').dt.to_timestamp()
fed['Date'] = fed['Date'].dt.to_period('M').dt.to_timestamp()
ecb['Date'] = ecb['Date'].dt.to_period('M').dt.to_timestamp()
sarb['Date'] = sarb['Date'].dt.to_period('M').dt.to_timestamp()

start = '2021-03-01'
end = '2026-03-01'

zar_usd = zar_usd[(zar_usd['Date'] >= start) & (zar_usd['Date'] <= end)].reset_index(drop=True)
usd_eur = usd_eur[(usd_eur['Date'] >= start) & (usd_eur['Date'] <= end)].reset_index(drop=True)
fed = fed[(fed['Date'] >= start) & (fed['Date'] <= end)].reset_index(drop=True)
ecb = ecb[(ecb['Date'] >= start) & (ecb['Date'] <= end)].reset_index(drop=True)
sarb = sarb[(sarb['Date'] >= start) & (sarb['Date'] <= end)].reset_index(drop=True)

for name, df in [('ZAR_USD', zar_usd), ('USD_EUR', usd_eur), ('Fed', fed), ('ECB', ecb), ('SARB', sarb)]:
    print(f"{name}: {df.shape} | {df['Date'].min()} to {df['Date'].max()}")

# Build master dataframe with exchange rates, interest rates, monthly % changes and differentials
master = pd.DataFrame()
master['Date'] = zar_usd['Date']
master['ZAR_USD'] = zar_usd['ZAR_USD']
master['ZAR_EUR'] = usd_eur['ZAR_EUR']
master['Fed_Rate'] = fed['FED']
master['ECB_Rate'] = ecb['ECB_Rate']
master['SARB_Rate'] = sarb['SARB']

master['pct_ZAR_USD'] = master['ZAR_USD'].pct_change()
master['pct_ZAR_EUR'] = master['ZAR_EUR'].pct_change()

master['diff_USD'] = master['Fed_Rate'] - master['SARB_Rate']
master['diff_EUR'] = master['ECB_Rate'] - master['SARB_Rate']

print(master.shape)
print(master.head())

# Create aligned series for each currency pair (drops NaN from pct_change)
series_usd = master[['Date', 'pct_ZAR_USD', 'diff_USD']].dropna().reset_index(drop=True)
series_eur = master[['Date', 'pct_ZAR_EUR', 'diff_EUR']].dropna().reset_index(drop=True)

print(series_usd)
print(series_eur)

# Summary statistics including skewness and kurtosis
for name, col, series in [("ZAR/USD % change", "pct_ZAR_USD", series_usd),
                           ("Fed-SARB diff (pp)", "diff_USD", series_usd),
                           ("ZAR/EUR % change", "pct_ZAR_EUR", series_eur),
                           ("ECB-SARB diff (pp)", "diff_EUR", series_eur)]:
    data = series[col]
    print(f"{name}:")
    print(f"  Count:    {data.count()}")
    print(f"  Mean:     {data.mean():.6f}")
    print(f"  Std:      {data.std():.6f}")
    print(f"  Min:      {data.min():.6f}")
    print(f"  Max:      {data.max():.6f}")
    print(f"  Skew:     {stats.skew(data):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(data):.6f}")

# Plot % change and interest rate differential for each currency pair
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

for ax, pct_col, diff_col, title in [
    (axes[0], "pct_ZAR_USD", "diff_USD", "ZAR/USD"),
    (axes[1], "pct_ZAR_EUR", "diff_EUR", "ZAR/EUR")]:
    
    ax2 = ax.twinx()
    ax.plot(series_usd["Date"] if "USD" in pct_col else series_eur["Date"],
            series_usd[pct_col] if "USD" in pct_col else series_eur[pct_col],
            label="% change", color="steelblue")
    ax2.plot(series_usd["Date"] if "USD" in pct_col else series_eur["Date"],
             series_usd[diff_col] if "USD" in pct_col else series_eur[diff_col],
             label="Rate diff", color="tomato", linestyle="--")
    
    ax.axhline(0, color="black", lw=0.5, linestyle=":")
    ax.set_title(title)
    ax.set_ylabel("% change")
    ax2.set_ylabel("Rate diff (pp)")
    
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=9)

plt.tight_layout()
plt.show()

# Plot interest rate differentials separately
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(series_usd['Date'], series_usd['diff_USD'], color='steelblue')
axes[0].set_title('Fed - SARB Interest Rate Differential')
axes[0].set_ylabel('Rate diff (pp)')

axes[1].plot(series_eur['Date'], series_eur['diff_EUR'], color='tomato')
axes[1].set_title('ECB - SARB Interest Rate Differential')
axes[1].set_ylabel('Rate diff (pp)')

plt.tight_layout()

# USD/ZAR baseline IFE regression: train on first 45 obs, forecast remaining 15
train = series_usd.iloc[:45]
test = series_usd.iloc[45:]

X_train = train[['diff_USD']]
y_train = train['pct_ZAR_USD']

model = LinearRegression()
model.fit(X_train, y_train)

alpha = model.intercept_
beta = model.coef_[0]
r_squared = model.score(X_train, y_train)

print(f'Alpha: {alpha:.6f}')
print(f'Beta: {beta:.6f}')
print(f'R-squared: {r_squared:.6f}')

X_test = test[['diff_USD']]
y_test = test['pct_ZAR_USD']

forecast = model.predict(X_test)

errors = y_test.values - forecast

bias = np.mean(errors)
mae = np.mean(np.abs(errors))
mse = np.mean(errors**2)
rmse = np.sqrt(np.mean(errors**2))

print(f"Bias (Mean Error): {bias:.6f}")
print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")

X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

bp = het_breuschpagan(model_sm.resid, model_sm.model.exog)
bg = acorr_breusch_godfrey(model_sm, nlags=3)
reset = linear_reset(model_sm, power=3, use_f=True)
print(f"Breusch-Pagan p-value:  {bp[1]:.4f}")
print(f"Breusch-Godfrey p-value: {bg[1]:.4f}")
print(f"Ramsey RESET p-value (ZAR/USD baseline): {reset.pvalue:.4f}")

# EUR/ZAR baseline IFE regression
train = series_eur.iloc[:45]
test = series_eur.iloc[45:]

X_train_e = train[['diff_EUR']]
y_train_e = train['pct_ZAR_EUR']

model1 = LinearRegression()
model1.fit(X_train_e, y_train_e)

alpha1 = model1.intercept_
beta1 = model1.coef_[0]
r_squared1 = model1.score(X_train_e, y_train_e)

print(f'Alpha: {alpha1:.6f}')
print(f'Beta: {beta1:.6f}')
print(f'R-squared: {r_squared1:.6f}')

X_test_e = test[['diff_EUR']]
y_test_e = test['pct_ZAR_EUR']

forecast1 = model1.predict(X_test_e)

errors1 = y_test_e.values - forecast1

bias1 = np.mean(errors1)
mae1 = np.mean(np.abs(errors1))
mse1 = np.mean(errors1**2)
rmse1 = np.sqrt(np.mean(errors1**2))

print(f"Bias (Mean Error): {bias1:.6f}")
print(f"MAE: {mae1:.6f}")
print(f"MSE: {mse1:.6f}")
print(f"RMSE: {rmse1:.6f}")

X_train_sm1 = sm.add_constant(X_train_e)
model_sm1 = sm.OLS(y_train_e, X_train_sm1).fit()
print(model_sm1.summary())

bp1 = het_breuschpagan(model_sm1.resid, model_sm1.model.exog)
bg1 = acorr_breusch_godfrey(model_sm1, nlags=3)
reset1 = linear_reset(model_sm1, power=3, use_f=True)
print(f"Breusch-Pagan p-value:  {bp1[1]:.4f}")
print(f"Breusch-Godfrey p-value: {bg1[1]:.4f}")
print(f"Ramsey RESET p-value (ZAR/EUR baseline): {reset1.pvalue:.4f}")

# USD/ZAR custom model: lagged differential + lagged 3-month volatility
series_usd['lag_diff_USD'] = series_usd['diff_USD'].shift(1)
series_usd['lag_vol_USD']  = series_usd['pct_ZAR_USD'].shift(1).rolling(3).std()

custom_usd = series_usd.dropna().reset_index(drop=True)

train_c = custom_usd.iloc[:45]
test_c  = custom_usd.iloc[45:]

X_train_c = train_c[['lag_diff_USD', 'lag_vol_USD']]
y_train_c = train_c['pct_ZAR_USD']

model_c = LinearRegression()
model_c.fit(X_train_c, y_train_c)

alpha_c    = model_c.intercept_
beta1_c    = model_c.coef_[0]
beta2_c    = model_c.coef_[1]
r_squared_c = model_c.score(X_train_c, y_train_c)

print(f'Alpha: {alpha_c:.6f}')
print(f'Beta1 (lag diff): {beta1_c:.6f}')
print(f'Beta2 (lag vol):  {beta2_c:.6f}')
print(f'R-squared: {r_squared_c:.6f}')

X_test_c = test_c[['lag_diff_USD', 'lag_vol_USD']]
y_test_c = test_c['pct_ZAR_USD']

forecast_c = model_c.predict(X_test_c)

errors_c = y_test_c.values - forecast_c

bias_c = np.mean(errors_c)
mae_c  = np.mean(np.abs(errors_c))
mse_c  = np.mean(errors_c**2)
rmse_c = np.sqrt(np.mean(errors_c**2))

print(f"Bias (Mean Error): {bias_c:.6f}")
print(f"MAE: {mae_c:.6f}")
print(f"MSE: {mse_c:.6f}")
print(f"RMSE: {rmse_c:.6f}")

X_train_sm_c = sm.add_constant(X_train_c)
model_sm_c = sm.OLS(y_train_c, X_train_sm_c).fit()
print(model_sm_c.summary())

bp_c = het_breuschpagan(model_sm_c.resid, model_sm_c.model.exog)
bg_c = acorr_breusch_godfrey(model_sm_c, nlags=3)
reset_c = linear_reset(model_sm_c, power=3, use_f=True)
print(f"Breusch-Pagan p-value:  {bp_c[1]:.4f}")
print(f"Breusch-Godfrey p-value: {bg_c[1]:.4f}")
print(f"Ramsey RESET p-value (ZAR/USD custom):   {reset_c.pvalue:.4f}")

# EUR/ZAR custom model: lagged differential + lagged 3-month volatility
series_eur['lag_diff_EUR'] = series_eur['diff_EUR'].shift(1)
series_eur['lag_vol_EUR']  = series_eur['pct_ZAR_EUR'].shift(1).rolling(3).std()

custom_eur = series_eur.dropna().reset_index(drop=True)

train_ce = custom_eur.iloc[:45]
test_ce  = custom_eur.iloc[45:]

X_train_ce = train_ce[['lag_diff_EUR', 'lag_vol_EUR']]
y_train_ce = train_ce['pct_ZAR_EUR']

model_ce = LinearRegression()
model_ce.fit(X_train_ce, y_train_ce)

alpha_ce    = model_ce.intercept_
beta1_ce    = model_ce.coef_[0]
beta2_ce    = model_ce.coef_[1]
r_squared_ce = model_ce.score(X_train_ce, y_train_ce)

print(f'Alpha: {alpha_ce:.6f}')
print(f'Beta1 (lag diff): {beta1_ce:.6f}')
print(f'Beta2 (lag vol):  {beta2_ce:.6f}')
print(f'R-squared: {r_squared_ce:.6f}')

X_test_ce = test_ce[['lag_diff_EUR', 'lag_vol_EUR']]
y_test_ce = test_ce['pct_ZAR_EUR']

forecast_ce = model_ce.predict(X_test_ce)

errors_ce = y_test_ce.values - forecast_ce

bias_ce = np.mean(errors_ce)
mae_ce  = np.mean(np.abs(errors_ce))
mse_ce  = np.mean(errors_ce**2)
rmse_ce = np.sqrt(np.mean(errors_ce**2))

print(f"Bias (Mean Error): {bias_ce:.6f}")
print(f"MAE: {mae_ce:.6f}")
print(f"MSE: {mse_ce:.6f}")
print(f"RMSE: {rmse_ce:.6f}")

X_train_sm_ce = sm.add_constant(X_train_ce)
model_sm_ce = sm.OLS(y_train_ce, X_train_sm_ce).fit()
print(model_sm_ce.summary())

bp_ce = het_breuschpagan(model_sm_ce.resid, model_sm_ce.model.exog)
bg_ce = acorr_breusch_godfrey(model_sm_ce, nlags=3)
reset_ce = linear_reset(model_sm_ce, power=3, use_f=True)
print(f"Breusch-Pagan p-value:  {bp_ce[1]:.4f}")
print(f"Breusch-Godfrey p-value: {bg_ce[1]:.4f}")
print(f"Ramsey RESET p-value (ZAR/EUR custom):   {reset_ce.pvalue:.4f}")

# Plot actual vs baseline and custom forecasts for both currency pairs
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(series_usd['Date'], series_usd['pct_ZAR_USD'], label='Actual', color='steelblue')
axes[0].plot(test['Date'], forecast, label='Baseline forecast', color='tomato', linestyle='--')
axes[0].plot(test_c['Date'], forecast_c, label='Custom forecast', color='green', linestyle='--')
axes[0].set_title('USD/ZAR: Actual vs Forecasts')
axes[0].set_ylabel('% change')
axes[0].legend(fontsize=9)

axes[1].plot(series_eur['Date'], series_eur['pct_ZAR_EUR'], label='Actual', color='steelblue')
axes[1].plot(test['Date'], forecast1, label='Baseline forecast', color='tomato', linestyle='--')
axes[1].plot(test_ce['Date'], forecast_ce, label='Custom forecast', color='green', linestyle='--')
axes[1].set_title('EUR/ZAR: Actual vs Forecasts')
axes[1].set_ylabel('% change')
axes[1].legend(fontsize=9)

plt.tight_layout()




