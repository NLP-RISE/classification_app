# Simple Zero-Shot Classifier

## Structure

```text
simple_zero_shot_app_src/
├── app.py
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── device.py
│   └── ui.py
└── tests/
    └── test_classifier.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m pytest
```

Edit `LABELS` in `src/classifier.py` to change the fixed labels.
