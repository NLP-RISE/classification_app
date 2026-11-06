from __future__ import annotations

import streamlit as st

from src.classifier import LABELS, MODEL_NAME, ZeroShotClassifier
from src.device import available_devices


@st.cache_resource(show_spinner="Loading the classification model...")
def load_classifier(
    model_name: str,
    device: str,
) -> ZeroShotClassifier:
    return ZeroShotClassifier(
        model_name=model_name,
        device=device,
    )


def run_app() -> None:
    st.set_page_config(
        page_title="Zero-Shot Classifier",
        page_icon="🏷️",
    )

    st.title("Zero-Shot Text Classifier")

    model_name = st.text_input(
        "Hugging Face model name or local path",
        value=MODEL_NAME,
        help="Enter a Hugging Face NLI model ID or a local model directory.",
    ).strip()

    st.write("Labels: " + ", ".join(LABELS))

    device = st.selectbox(
        "Device",
        available_devices(),
    )

    text = st.text_area(
        "Paste text to classify",
        height=250,
        placeholder="Paste text here...",
    )

    if st.button("Classify", type="primary"):
        if not model_name:
            st.warning("Enter a model name or local path.")
            return

        if not text.strip():
            st.warning("Paste some text first.")
            return

        try:
            classifier = load_classifier(
                model_name,
                device,
            )
            results = classifier.classify(text)
            top_result = results.iloc[0]

            st.success(
                f"Prediction: {top_result['label']}"
            )

        except Exception as error:
            st.exception(error)