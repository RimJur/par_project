# par-project

## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. Python **3.14+** is required (uv will install it for you).

## Install uv

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternatives

- Homebrew: `brew install uv`
- pipx: `pipx install uv`
- pip: `pip install uv`

After installation, restart your shell (or `source` your profile) and verify:

```bash
uv --version
```

## Initialize the repository

1. Clone the repo and `cd` into it:

   ```bash
   git clone <repo-url>
   cd par_project
   ```

2. Install the matching Python version (reads `.python-version` / `pyproject.toml`):

   ```bash
   uv python install
   ```

3. Create the virtual environment and install dependencies from `pyproject.toml` / `uv.lock`:

   ```bash
   uv sync
   ```

   This creates a `.venv/` in the project root.

4. (Optional) Activate the venv directly:

   ```bash
   source .venv/bin/activate          # macOS / Linux
   .venv\Scripts\activate             # Windows
   ```

## Run the project

Use `uv run` to execute commands inside the project environment without manual activation:

```bash
uv run python main.py
uv run python part_3.py
uv run python part_4.py
```

## Data files

Monthly CSVs.

| File | Contents | Source |
|---|---|---|
| `ZAR_USD.csv` | South African Rand per 1 US Dollar | South African Reserve Bank ([resbank.co.za](https://www.resbank.co.za/)) |
| `USD_EUR.csv` | US Dollar per 1 Euro | South African Reserve Bank ([resbank.co.za](https://www.resbank.co.za/)) |
| `ZA_TOP40_index.csv` | JSE Top 40 index, monthly close | Investing.com, [FTSE/JSE Top 40 historical data](https://www.investing.com/indices/ftse-jse-top-40-historical-data) |
| `ZA_US_trade_balance.csv` | South Africa–USA bilateral trade balance, USD millions (signed) | US Census Bureau, [Trade in Goods with South Africa](https://www.census.gov/foreign-trade/balance/c7910.html) |
| `SARB_policy_rate.csv` | South African Reserve Bank policy rate | South African Reserve Bank, [Selected historical rates](https://www.resbank.co.za/en/home/what-we-do/statistics/key-statistics/selected-historical-rates) |
| `ECB_policy_rate.csv` | ECB Main Refinancing Rate | FRED [`ECBMRRFR`](https://fred.stlouisfed.org/series/ECBMRRFR) |
| `US_policy_rate.csv` | US Federal Funds Rate | FRED [`FEDFUNDS`](https://fred.stlouisfed.org/series/FEDFUNDS) |

## Managing dependencies

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv remove <package>        # remove a dependency
uv lock --upgrade          # refresh the lock file
```
