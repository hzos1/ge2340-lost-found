# GE2340 Group 26 · Lost-and-Found Image/Text Matcher



The web app stores records locally in `app/data/`. On its first run, it imports the held-out test gallery and synthetic test queries as clearly labelled demo records (298 found items and 149 lost reports). New uploads are saved in the same database and ranked with the trained matcher; uploads do not retrain the model.

## Run the demo

```bash
pip install -r requirements.txt
python app/server.py
```

On Windows, you can also double-click `Open Web App Windows.bat`. On macOS, open `Open Web App Mac.command` in Terminal.

Open http://127.0.0.1:8765

- **Report a lost item** — save a report and rank it against found items people have uploaded
- **Report a found item** — a photo is required; the matcher looks for a matching lost report
- **Database** — browse every live record; delete requires the admin password `12345678`



## Retrain and slides

Evaluation artifacts in `artifacts/` are included so the app can start without retraining. To rebuild them from local training photos:

```bash
python -m src.pipeline
python -m src.make_results_pptx
```

