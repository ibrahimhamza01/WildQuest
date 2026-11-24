# WildQuest App (UI MVP)

A Streamlit app for uploading wildlife photos, viewing species predictions, and
browsing the user's collection, XP and observation history (issue #13).

The app currently runs in **demo mode**. Predictions, collection and XP come from
deterministic mock providers in `app/mock_data/`, and every screen is labelled as
demo data. It needs no models, no dataset, no GPU and no AWS access.

## Run it locally

From the repository root (Python 3.10 or newer):

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r app\requirements.txt
.venv\Scripts\python -m streamlit run app\streamlit_app.py
```

The app opens at http://localhost:8501. Stop it with **Ctrl + C** in the terminal.

**On the AWS instance:** run the same commands. VS Code Remote-SSH forwards port
8501 automatically, so the app still opens in your local browser.

## Screens

| Screen | What it shows |
|---|---|
| Identify | Upload and preview a photo; numbered boxes around each animal; for each one the predicted species, confidence, Top-3 alternatives and detector confidence. Clear states for loading, no animal found, uncertain predictions and service errors. |
| Collection | Progress toward all 25 species, total XP, and a card per species (discovered or locked), grouped by birds, mammals, reptiles and amphibians. |
| History | Every observation with time, species, confidence, review status, XP result and XP earned. |

An uncertain prediction is one whose confidence is below 70%. That is the
provisional `review_threshold` from `Code/configs/gamification.json` (#12), which
issue #14 will tune.

## Trying each state

Open **Demo settings** in the sidebar and pick a **Mock result**: one animal,
several animals, an uncertain prediction, no animal found, or a service error.

With **Automatic**, the result depends on the photo, so the same photo always
gives the same result. A filename containing `multi`, `uncertain`, `noanimal`,
`empty` or `error` forces that state.

## Tests

```bash
python -m unittest Code/tests/test_app_contract.py -v
```

The tests cover response parsing (valid, malformed, multiple detections,
uncertain, no animal, error), the mock providers, image loading and box drawing.
When Streamlit is installed, they also run the app headlessly and check each UI
state and page. Without Streamlit, those UI tests are skipped.

## Connecting the real backend

The UI only talks to two providers, created in `get_providers()` in
`app/streamlit_app.py`:

- **Inference (#11).** Any object with `predict(image_bytes, filename, scenario=None) -> dict`.
  The dict format is documented at the top of `app/contracts.py`:
  - `status` is `ok`, `no_animal` or `error`.
  - Each detection has a normalized `box`, a `detector_confidence`, and classifier
    `predictions` as `class_id` + `confidence`.
  - The MegaDetector `Detection` pixel box converts to this format by dividing by
    the image width and height.
- **Observations (#12).** Any object with `summary()`, `collection()` and
  `history()` returning the same shapes as `ObservationService`. The mock follows
  those shapes exactly, so `ObservationService` can replace it directly.

Once real providers are connected, have `get_providers()` return `False` as its
third value. That removes the demo banner.