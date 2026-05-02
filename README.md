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

## Managing dependencies

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv remove <package>        # remove a dependency
uv lock --upgrade          # refresh the lock file
```
