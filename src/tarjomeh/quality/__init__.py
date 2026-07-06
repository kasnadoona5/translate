"""Translation quality assurance pipeline for Tarjomeh.

Provides LLM-based critique, iterative refinement, and back-translation
verification for publication-quality Persian translations.
"""

from __future__ import annotations

from tarjomeh.quality.critique import CritiqueResult, TranslationCritique
from tarjomeh.quality.refiner import TranslationRefiner
from tarjomeh.quality.back_translator import BackTranslationResult, BackTranslator

__all__ = [
    "CritiqueResult",
    "TranslationCritique",
    "TranslationRefiner",
    "BackTranslationResult",
    "BackTranslator",
]
