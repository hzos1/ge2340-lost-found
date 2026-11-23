# GE2340 Group 26 · Lost-and-Found Image/Text Matcher

CityUHK campus matcher for lost and found reports. CLIP (ViT-B/32, no fine-tuning) is the preferred method from the proposal. A frozen MobileNetV3 + linear pair classifier is the comparison method.

The web app stores live records only. Training photos stay on the machine that trained the model.

## Run the demo

```bash
pip install -r requirements.txt
python app/server.py
```

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

Results slides: `slides/GE2340_Group26_Results.pptx`
