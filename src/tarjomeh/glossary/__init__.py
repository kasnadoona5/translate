"""Glossary sub-package — BabelDOC-compatible glossary management.

Provides :class:`GlossaryManager` for term lookup and
:class:`GlossaryComplianceChecker` for post-translation verification.
"""

from __future__ import annotations

from tarjomeh.glossary.compliance import ComplianceReport, GlossaryComplianceChecker
from tarjomeh.glossary.manager import GlossaryEntry, GlossaryManager

__all__ = [
    "ComplianceReport",
    "GlossaryComplianceChecker",
    "GlossaryEntry",
    "GlossaryManager",
]
