import pytest

from src.classifier import ZeroShotClassifier


@pytest.mark.integration
def test_classifies_technology_news() -> None:
    classifier = ZeroShotClassifier(
        model_name="MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
        device="cpu",
    )

    text = (
        "A semiconductor company introduced a new processor for laptop "
        "computers. The chip improves graphics performance, reduces power "
        "consumption, and includes dedicated hardware for artificial "
        "intelligence workloads."
    )

    results = classifier.classify(text)

    assert not results.empty
    assert results.iloc[0]["label"] == "technology"
    assert results.iloc[0]["score"] > 0.5