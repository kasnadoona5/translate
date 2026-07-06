"""EPUB3 document exporter for Tarjomeh."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode

logger = logging.getLogger(__name__)


class EpubExporter(BaseExporter):
    """EPUB3 RTL and bilingual document exporter.

    Packages documents into valid EPUB3 container files without external packages.
    """

    def export(
        self,
        document: TranslatedDocument,
        output_path: str | Path,
        bilingual_mode: BilingualMode = "target_only",
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # 1. Group paragraphs into chapters based on heading_level == 1
        chapters: list[dict[str, any]] = []
        current_title = "Introduction"
        current_paras = []

        for p in document.paragraphs:
            if p.heading_level == 1:
                if current_paras:
                    chapters.append({"title": current_title, "paragraphs": current_paras})
                current_title = p.translated_text if bilingual_mode == "target_only" else f"{p.source_text} / {p.translated_text}"
                current_paras = []
            current_paras.append(p)

        if current_paras:
            chapters.append({"title": current_title, "paragraphs": current_paras})
        if not chapters:
            chapters.append({"title": "Document", "paragraphs": []})

        # 2. Build CSS Stylesheet
        css_content = """body {
    direction: rtl;
    text-align: justify;
    font-family: 'Vazirmatn', sans-serif;
    line-height: 1.6;
    margin: 10px;
}

h1, h2, h3, h4 {
    direction: rtl;
    text-align: center;
}

.source {
    direction: ltr;
    text-align: left;
    color: #555555;
    margin-bottom: 0.2em;
    font-style: italic;
}

.target {
    direction: rtl;
    text-align: right;
    margin-bottom: 1em;
}

.side-by-side {
    display: flex;
    margin-bottom: 1em;
    border-bottom: 1px solid #eeeeee;
    padding-bottom: 0.5em;
}

.side-by-side .left {
    width: 50%;
    direction: ltr;
    text-align: left;
    padding-right: 10px;
    border-right: 1px solid #dddddd;
}

.side-by-side .right {
    width: 50%;
    direction: rtl;
    text-align: right;
    padding-left: 10px;
}
"""

        # 3. Create EPUB container
        with zipfile.ZipFile(out, "w") as zf:
            # mimetype must be first and uncompressed
            zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

            # META-INF/container.xml
            container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""
            zf.writestr("META-INF/container.xml", container_xml)

            # OEBPS/style.css
            zf.writestr("OEBPS/style.css", css_content)

            # Write XHTML chapters and accumulate manifest items
            manifest_items = [
                '<item id="style" href="style.css" media-type="text/css"/>'
            ]
            spine_items = []

            for idx, chap in enumerate(chapters, 1):
                xhtml_parts = []
                for p in chap["paragraphs"]:
                    if p.heading_level == 1:
                        # Chapter title is already rendered in the <h1> header
                        continue

                    # Render paragraph headings and body
                    p_tag = "p"
                    if p.heading_level is not None:
                        p_tag = f"h{min(p.heading_level, 6)}"

                    if bilingual_mode == "target_only":
                        xhtml_parts.append(f"<{p_tag} class='target'>{p.translated_text}</{p_tag}>")
                    elif bilingual_mode == "inline":
                        xhtml_parts.append(
                            f"<{p_tag} class='source'>{p.source_text}</{p_tag}>\n"
                            f"<{p_tag} class='target'>{p.translated_text}</{p_tag}>"
                        )
                    else:  # side_by_side
                        xhtml_parts.append(
                            f"<div class='side-by-side'>\n"
                            f"  <div class='left'>{p.source_text}</div>\n"
                            f"  <div class='right'>{p.translated_text}</div>\n"
                            f"</div>"
                        )

                paragraphs_xhtml = "\n".join(xhtml_parts)
                chapter_title = chap["title"]

                xhtml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fa" dir="rtl">
<head>
    <title>{chapter_title}</title>
    <link rel="stylesheet" href="style.css" type="text/css"/>
</head>
<body>
    <section epub:type="chapter">
        <h1>{chapter_title}</h1>
        {paragraphs_xhtml}
    </section>
</body>
</html>"""

                filename = f"chapter_{idx}.xhtml"
                zf.writestr(f"OEBPS/{filename}", xhtml_content)

                manifest_items.append(f'<item id="chapter_{idx}" href="{filename}" media-type="application/xhtml+xml"/>')
                spine_items.append(f'<itemref idref="chapter_{idx}"/>')

            # EPUB3 Navigation Page (nav.xhtml)
            nav_links = []
            for idx, chap in enumerate(chapters, 1):
                nav_links.append(f'<li><a href="chapter_{idx}.xhtml">{chap["title"]}</a></li>')
            nav_list = "\n".join(nav_links)

            nav_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fa" dir="rtl">
<head>
    <title>Navigation</title>
    <link rel="stylesheet" href="style.css" type="text/css"/>
</head>
<body>
    <nav epub:type="toc" id="toc">
        <h1>Table of Contents</h1>
        <ol>
            {nav_list}
        </ol>
    </nav>
</body>
</html>"""
            zf.writestr("OEBPS/nav.xhtml", nav_xhtml)
            manifest_items.append('<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>')

            # Build metadata and content.opf
            manifest_str = "\n    ".join(manifest_items)
            spine_str = "\n    ".join(spine_items)

            opf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="pub-id" version="3.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="pub-id">urn:uuid:tarjomeh-job-{document.title.replace(" ", "-")}</dc:identifier>
        <dc:title>{document.title}</dc:title>
        <dc:creator>{document.author or "Unknown"}</dc:creator>
        <dc:language>fa</dc:language>
        <meta property="dcterms:modified">2026-07-06T12:00:00Z</meta>
    </metadata>
    <manifest>
        {manifest_str}
    </manifest>
    <spine>
        {spine_str}
    </spine>
</package>"""
            zf.writestr("OEBPS/content.opf", opf_content)

        return out
