# Retail Location DSS (Python)

Decision support system for choosing among four area types (City Centre, Recreation & Tourism, Transit Hub,
Residential) for a new convenience and specialty store. Fuzzy AHP weights the criteria, TOPSIS ranks the
locations and SAW is used as a cross-check. **All data and expert judgements are hypothetical.**

## Run
1. Install Python 3.8 or newer and NumPy:  `pip install numpy`
2. In this folder run:  `python app.py`
3. The browser opens at http://127.0.0.1:8501/ (press Ctrl+C in the terminal to stop).
   If the port is busy: `DSS_PORT=8600 python app.py` (Windows PowerShell: `$env:DSS_PORT=8600; python app.py`).

## Live website (GitHub Pages)
The same folder also works as a static website: the Python engine then runs inside the browser (Pyodide, Python compiled to
WebAssembly), so no server is needed and visitors only open a link.
1. Create a **public** GitHub repository (for example `retail-location-dss`).
2. Upload the contents of this folder (index.html, the .py files, delphi/, .nojekyll, README.md) to the repository root.
3. Repository Settings > Pages > Build and deployment > Source: *Deploy from a branch*, Branch: `main`, folder `/ (root)` > Save.
4. After 1-2 minutes the site is at `https://<your-username>.github.io/<repository-name>/`.
The first visit downloads Pyodide and NumPy (about 10-20 seconds); the visitor needs internet access.

## Test
`python test_core.py` runs hand-calculated examples, the independent re-implementation and invariance tests.
The same checks run live on the Verification tab of the dashboard.

## Files
| File | Purpose |
|---|---|
| `dss_core.py` | Calculation engine: Fuzzy AHP, TOPSIS, SAW, scenarios, sensitivity, verification (NumPy). |
| `independent_check.py` | Second implementation of the same calculations in plain Python (loops, no NumPy) used for verification. |
| `case_text.py` | Definitions, location profiles and rationale of the hypothetical judgements. |
| `app.py` | Local web server (Python standard library) that connects the engine to the dashboard. |
| `index.html` | Browser dashboard (Dashboard, Inputs, Weights, Scenarios, Sensitivity, Verification, Guide). |
| `test_core.py` | Tests. |
| `report_export.py` | Writes the numbers used in the written report (`python report_export.py <folder>`). |
| `make_figures.py` | Takes the report figures from the dashboard (needs `pip install playwright` and `playwright install chromium`). |
| `delphi/` | Long-list of criteria, simulation of the hypothetical Delphi panel and its CSV results (`python delphi_simulation.py output --mc`). |

## Using it for another case
Change the data (`CRIT`), the judgements (`PAIRS`) and the scenario weights (`SCEN`) at the top of `dss_core.py`,
or edit data and judgements in the dashboard. Run `python test_core.py` afterwards. Changing the number of
alternatives or criteria requires editing the lists in `dss_core.py` and the group sizes in `independent_check.py`.
