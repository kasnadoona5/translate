"""Pytest configuration and global mock setup."""

import sys
from unittest.mock import MagicMock

# Inject fallback mocks into sys.modules if real modules are not present on the host.
# This prevents ModuleNotFoundError/ImportError on import of Tarjomeh modules.
try:
    import docx
except ImportError:
    mock_docx = MagicMock()
    sys.modules["docx"] = mock_docx
    sys.modules["docx.enum.text"] = MagicMock()
    sys.modules["docx.oxml"] = MagicMock()
    sys.modules["docx.oxml.ns"] = MagicMock()

try:
    import reportlab
except ImportError:
    mock_rl = MagicMock()
    sys.modules["reportlab"] = mock_rl
    sys.modules["reportlab.lib.pagesizes"] = MagicMock()
    sys.modules["reportlab.lib.styles"] = MagicMock()
    sys.modules["reportlab.platypus"] = MagicMock()
    sys.modules["reportlab.lib"] = MagicMock()
    sys.modules["reportlab.pdfbase"] = MagicMock()
    sys.modules["reportlab.pdfbase.ttfonts"] = MagicMock()

try:
    import arabic_reshaper
except ImportError:
    sys.modules["arabic_reshaper"] = MagicMock()
    sys.modules["bidi"] = MagicMock()
    sys.modules["bidi.algorithm"] = MagicMock()
