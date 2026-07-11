"""EPUB3 document exporter for Tarjomeh."""

from __future__ import annotations

import logging
import zipfile
import html
from datetime import datetime
from pathlib import Path
from tarjomeh.exporters.base import BaseExporter, TranslatedDocument, BilingualMode
from tarjomeh.exporters.term_notes import document_term_notes, paragraph_note_parts

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
        term_notes = document_term_notes(document)
        note_backlinks: dict[int, str] = {}

        def render_target(text: str, metadata: dict, chapter_index: int) -> str:
            rendered = []
            for segment, ref in paragraph_note_parts(text, metadata):
                rendered.append(html.escape(segment))
                if ref is not None:
                    number = int(ref["number"])
                    ref_id = f"term-note-ref-{number}"
                    note_backlinks[number] = f"chapter_{chapter_index}.xhtml#{ref_id}"
                    rendered.append(
                        f'<a id="{ref_id}" href="notes.xhtml#term-note-{number}" '
                        f'epub:type="noteref" role="doc-noteref">{number}</a>'
                    )
            return "".join(rendered)

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

        # 2. Build CSS Stylesheet and Font Embedding
        font_styles = ""
        reg_path = self._resolve_font_path("Vazirmatn-Regular.ttf")
        bold_path = self._resolve_font_path("Vazirmatn-Bold.ttf")

        if reg_path and reg_path.is_file():
            font_styles += """@font-face {
    font-family: 'Vazirmatn';
    src: url('fonts/Vazirmatn-Regular.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}
"""
        if bold_path and bold_path.is_file():
            font_styles += """@font-face {
    font-family: 'Vazirmatn-Bold';
    src: url('fonts/Vazirmatn-Bold.ttf') format('truetype');
    font-weight: bold;
    font-style: normal;
}
"""

        css_content = font_styles + """body {
    direction: rtl;
    text-align: justify;
    font-family: 'Vazirmatn', sans-serif;
    line-height: 1.6;
    margin: 10px;
}

h1, h2, h3, h4 {
    direction: rtl;
    text-align: center;
    font-family: 'Vazirmatn-Bold', 'Vazirmatn', sans-serif;
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

            # If fonts exist, copy them and add to manifest
            if reg_path and reg_path.is_file():
                zf.write(str(reg_path), "OEBPS/fonts/Vazirmatn-Regular.ttf")
                manifest_items.append('<item id="font-regular" href="fonts/Vazirmatn-Regular.ttf" media-type="font/ttf"/>')
            if bold_path and bold_path.is_file():
                zf.write(str(bold_path), "OEBPS/fonts/Vazirmatn-Bold.ttf")
                manifest_items.append('<item id="font-bold" href="fonts/Vazirmatn-Bold.ttf" media-type="font/ttf"/>')

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

                    escaped_source = html.escape(p.source_text)
                    escaped_translated = render_target(
                        p.translated_text,
                        p.metadata,
                        idx,
                    )

                    if bilingual_mode == "target_only":
                        xhtml_parts.append(f"<{p_tag} class='target'>{escaped_translated}</{p_tag}>")
                    elif bilingual_mode == "inline":
                        xhtml_parts.append(
                            f"<{p_tag} class='source'>{escaped_source}</{p_tag}>\n"
                            f"<{p_tag} class='target'>{escaped_translated}</{p_tag}>"
                        )
                    else:  # side_by_side
                        xhtml_parts.append(
                            f"<div class='side-by-side'>\n"
                            f"  <div class='left'>{escaped_source}</div>\n"
                            f"  <div class='right'>{escaped_translated}</div>\n"
                            f"</div>"
                        )

                paragraphs_xhtml = "\n".join(xhtml_parts)
                chapter_title = html.escape(chap["title"])

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

            if term_notes:
                note_items = []
                for note in term_notes:
                    number = int(note["number"])
                    backlink = html.escape(note_backlinks.get(number, ""))
                    back_link = (
                        f' <a href="{backlink}" role="doc-backlink">back</a>'
                        if backlink else ""
                    )
                    note_items.append(
                        f'<aside id="term-note-{number}" epub:type="footnote" '
                        f'role="doc-footnote"><p>{number}. '
                        f'{html.escape(str(note["original"]))} '
                        f'({html.escape(str(note["transliteration"]))})'
                        f'{back_link}</p></aside>'
                    )
                notes_xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="fa" dir="rtl">
<head><title>یادداشت‌ها</title><link rel="stylesheet" href="style.css" type="text/css"/></head>
<body><section epub:type="footnotes"><h1>یادداشت‌ها</h1>{''.join(note_items)}</section></body>
</html>"""
                zf.writestr("OEBPS/notes.xhtml", notes_xhtml)
                manifest_items.append(
                    '<item id="term-notes" href="notes.xhtml" '
                    'media-type="application/xhtml+xml"/>'
                )
                spine_items.append('<itemref idref="term-notes"/>')

            # EPUB3 Navigation Page (nav.xhtml)
            nav_links = []
            for idx, chap in enumerate(chapters, 1):
                nav_links.append(f'<li><a href="chapter_{idx}.xhtml">{html.escape(chap["title"])}</a></li>')
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

            modified_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            escaped_title = html.escape(document.title)
            escaped_author = html.escape(document.author or "Unknown")

            opf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="pub-id" version="3.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:identifier id="pub-id">urn:uuid:tarjomeh-job-{escaped_title.replace(" ", "-")}</dc:identifier>
        <dc:title>{escaped_title}</dc:title>
        <dc:creator>{escaped_author}</dc:creator>
        <dc:language>fa</dc:language>
        <meta property="dcterms:modified">{modified_time}</meta>
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
