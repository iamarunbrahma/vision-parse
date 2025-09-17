import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pydantic import BaseModel

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer, LTTextLine
import pdfplumber
from tqdm import tqdm


# Hardcoded configuration (no external config dependency)
BULLET_CHARS = "•◦▪▫●○*·–—-"
HEADER_MARGIN = 50.0
FOOTER_MARGIN = 50.0
_logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _indent(level: int) -> str:
    """Return 3 spaces per indent level (e.g., level 2 -> 6 spaces)."""
    try:
        lvl = max(0, int(level))
    except Exception:
        lvl = 0
    return " " * (3 * lvl)


def _is_bullet_text(text: str) -> bool:
    return text.strip().startswith(tuple(BULLET_CHARS))


def _convert_bullet(text: str) -> str:
    text = re.sub(r"^\s*", "", text)
    return re.sub(f"^[{re.escape(BULLET_CHARS)}]\\s*", "- ", text)


def _is_ordered_item(text: str) -> bool:
    return bool(re.match(r"^\d+\s{0,3}[.)]", text.strip()))


def _normalize_ordered_item(text: str) -> str:
    text = re.sub(r"^\s*", "", text)
    # Collapse double ordinals like '1. 1. Foo' -> '1. Foo'
    text = re.sub(r"^(\d+)\s*[.)]\s+(\1)\s*[.)]\s*", r"\1. ", text)
    return re.sub(r"^(\d+)\s*[.)]\s*", r"\1. ", text)


def _is_horizontal_rule(text: str) -> bool:
    s = text.strip()
    return bool(
        re.match(r"^[-_]{3,}$", s) or s == "***" or s == "___"
    )


def _estimate_indent_step(positions: List[float]) -> float:
    if not positions:
        return 18.0
    uniq = sorted({round(p, 1) for p in positions})
    diffs = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    diffs = [d for d in diffs if d > 1.0]
    if not diffs:
        return 18.0
    diffs.sort()
    mid = len(diffs) // 2
    step = diffs[mid] if len(diffs) % 2 == 1 else (diffs[mid - 1] + diffs[mid]) / 2.0
    return max(8.0, min(36.0, step))


def _iter_chars(line: LTTextLine) -> Iterable[LTChar]:
    for obj in line:
        if isinstance(obj, LTChar):
            yield obj


def _line_font_sizes(line: LTTextLine) -> List[float]:
    return [float(ch.size) for ch in _iter_chars(line)]


def _line_fontnames(line: LTTextLine) -> List[str]:
    return [str(ch.fontname or "") for ch in _iter_chars(line)]


def _is_bold_fontname(name: str) -> bool:
    name_low = name.lower()
    return any(mark in name_low for mark in ("bold", "black", "heavy"))


def _is_italic_fontname(name: str) -> bool:
    name_low = name.lower()
    return any(mark in name_low for mark in ("italic", "oblique"))


class ListState(BaseModel):
    """Lightweight state for list parsing."""

    kind: str = "none"  # 'none' | 'ordered' | 'unordered'
    indent_level: int = 0


class PageFeatures(BaseModel):
    """Container for pre-extracted page features from pdfplumber."""

    tables: List[Dict[str, Any]]
    hyperlinks: List[Dict[str, Any]]
    horizontal_lines: List[Tuple[float, float, float, float]]


 


 


class FastMarkdown:
    """Extract per-page markdown using pdfminer.six (text) and pdfplumber (tables)."""

    def __init__(self, pdf_path: Path) -> None:
        """Initialize extractor with a PDF path."""
        self.pdf_path = pdf_path
        # No analyzers required; use simple threshold-based mapping

    def extract(self) -> List[str]:
        """Extract and return a list of markdown strings per page."""
        try:
            pages_md: List[str] = []
            params = LAParams(
                all_texts=True,
                line_overlap=0.5,
                char_margin=2.0,
                line_margin=0.5,
                word_margin=0.1,
                detect_vertical=False,
            )

            layouts = list(extract_pages(self.pdf_path, laparams=params))
            _logger.info("Loaded %d pages from %s", len(layouts), self.pdf_path.name)

            # Pre-extract per-page features using pdfplumber
            page_features: List[PageFeatures] = self._preextract_page_features()

            for page_index, layout in enumerate(
                tqdm(layouts, desc="Extracting markdown from pages")
            ):
                lines = self._collect_text_lines(layout)
                page_height = float(layout.bbox[3])
                pf = page_features[page_index] if page_index < len(page_features) else PageFeatures([], [], [])
                page_tables = pf.tables
                items = self._build_items(lines, page_tables, page_height)
                page_md = self._build_page_markdown_from_items(
                    items,
                    page_height,
                    pf.hyperlinks,
                    pf.horizontal_lines,
                )
                page_md = self._post_process(page_md)
                pages_md.append(page_md)

            _logger.info("Markdown extracted from %s", self.pdf_path.name)
            return pages_md
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.exception("Failed to extract markdown: %s", exc)
            return []

    def _preextract_page_features(self) -> List[PageFeatures]:
        """Collect tables, hyperlinks, and line segments per page via pdfplumber."""
        features: List[PageFeatures] = []
        try:
            with pdfplumber.open(str(self.pdf_path)) as plumber_pdf:
                for _page_index, plumber_page in enumerate(plumber_pdf.pages):
                    # Tables
                    page_tables: List[Dict[str, Any]] = []
                    for table in plumber_page.find_tables():
                        try:
                            content = table.extract()
                            bbox = table.bbox
                            if content:
                                page_tables.append({"bbox": bbox, "content": content})
                        except Exception:
                            continue

                    # Hyperlinks
                    page_links: List[Dict[str, Any]] = []
                    for link in getattr(plumber_page, "hyperlinks", []) or []:
                        try:
                            uri = link.get("uri") or link.get("url")
                            if not uri:
                                continue
                            x0 = float(link.get("x0"))
                            x1 = float(link.get("x1"))
                            top = float(link.get("top"))
                            bottom = float(link.get("bottom"))
                            page_links.append({"bbox": (x0, top, x1, bottom), "uri": uri})
                        except Exception:
                            continue

                    # Page lines (for strikethrough detection)
                    page_line_bboxes: List[Tuple[float, float, float, float]] = []
                    for line in getattr(plumber_page, "lines", []) or []:
                        try:
                            x0 = float(line.get("x0"))
                            x1 = float(line.get("x1"))
                            top = float(line.get("top"))
                            bottom = float(line.get("bottom"))
                            page_line_bboxes.append((x0, top, x1, bottom))
                        except Exception:
                            continue

                    features.append(
                        PageFeatures(
                            tables=page_tables,
                            hyperlinks=page_links,
                            horizontal_lines=page_line_bboxes,
                        )
                    )
        except Exception:
            # In case of any failure, return what we have (possibly empty)
            return features
        return features

    # ---------- Page processing helpers ----------

    def _collect_text_lines(self, layout: Any) -> List[LTTextLine]:
        lines: List[LTTextLine] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if isinstance(line, LTTextLine):
                        lines.append(line)
        # Sort by (y1 desc, x0 asc) for visual reading order
        lines.sort(key=lambda ln: (-ln.bbox[3], ln.bbox[0]))
        return lines

    # ---------- Integrated text + tables rendering ----------

    @staticmethod
    def _is_bullet_point(text: str) -> bool:
        """Return True for typical bullet markers."""
        s = text.lstrip()
        return s.startswith(tuple(BULLET_CHARS)) or s.startswith("- ")

    @staticmethod
    def _convert_bullet_to_markdown(text: str) -> str:
        """Normalize any bullet to '- ' markdown bullet."""
        s = re.sub(r"^\s*", "", text)
        s = re.sub(f"^[{re.escape(BULLET_CHARS)}]\\s*", "- ", s)
        if not s.startswith("- "):
            s = "- " + s
        return s

    @staticmethod
    def _is_numbered_list_item(text: str) -> bool:
        """Return True if line starts with 1., 1) etc."""
        return bool(re.match(r"^\s*\d+\s*[.)]", text))

    @staticmethod
    def _rects_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    @staticmethod
    def _convert_plumber_bbox_to_miner(bbox: Tuple[float, float, float, float], page_height: float) -> Tuple[float, float, float, float]:
        x0, top, x1, bottom = bbox
        y0 = page_height - bottom
        y1 = page_height - top
        return (float(x0), float(y0), float(x1), float(y1))

    def _build_items(
        self,
        lines: List[LTTextLine],
        page_tables: List[Dict[str, Any]],
        page_height: float,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        converted_tables: List[Dict[str, Any]] = []
        for tb in page_tables:
            miner_bbox = self._convert_plumber_bbox_to_miner(tb["bbox"], page_height)
            converted_tables.append({"type": "table", "bbox": miner_bbox, "data": tb["content"]})

        table_bboxes = [it["bbox"] for it in converted_tables]
        for ln in lines:
            bbox = tuple(float(v) for v in ln.bbox)
            if any(self._rects_intersect(bbox, tb) for tb in table_bboxes):
                continue
            items.append({"type": "text", "bbox": bbox, "data": ln})

        items.extend(converted_tables)
        return self._sort_items_by_reading_order(items, page_height)

    @staticmethod
    def _sort_items_by_reading_order(items: List[Dict[str, Any]], page_height: float) -> List[Dict[str, Any]]:
        if not items:
            return []

        def key_func(it: Dict[str, Any]) -> Tuple[int, float]:
            x0, _, _, y1 = it["bbox"]
            top_from_page = page_height - y1
            band = int(round(top_from_page / 12.0))
            return (band, x0)

        return sorted(items, key=key_func)

    def _build_page_markdown_from_items(
        self,
        items: List[Dict[str, Any]],
        page_height: float,
        page_hyperlinks: List[Dict[str, Any]],
        page_lines: List[Tuple[float, float, float, float]],
    ) -> str:
        if not items:
            return ""

        # Indentation baseline from all text items
        line_lefts = [it["bbox"][0] for it in items if it["type"] == "text"]
        base_left = min(line_lefts) if line_lefts else 0.0
        indent_step = _estimate_indent_step(line_lefts)

        markdown_lines: List[str] = []
        list_state = ListState()
        last_y1: Optional[float] = None
        last_sizes: List[float] = []
        prev_plain = ""
        code_block_open = False
        consecutive_code_lines = 0
        list_intro_active = False
        list_intro_indent = 0
        last_emitted_line: Optional[str] = None
        # New suppression trackers
        emitted_headers: set[str] = set()
        # Per-indent list counters for ordered lists
        list_counters: Dict[int, int] = {}

        # Convert links and lines into miner coordinate space once
        link_rects: List[Tuple[Tuple[float, float, float, float], str]] = [
            (self._convert_plumber_bbox_to_miner(lk["bbox"], page_height), lk["uri"])
            for lk in page_hyperlinks
            if isinstance(lk, dict) and "bbox" in lk and "uri" in lk
        ]
        line_rects_miner: List[Tuple[float, float, float, float]] = [
            self._convert_plumber_bbox_to_miner(b, page_height) for b in page_lines
        ]

        for it in items:
            if it["type"] == "table":
                # Close code block if open
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                markdown_lines.append("")
                markdown_lines.append(self._table_to_markdown(it["data"]))
                markdown_lines.append("")
                self._reset_list_state(list_state)
                continue

            # Text line processing
            line: LTTextLine = it["data"]
            left, _, _, y1 = it["bbox"]
            sizes = _line_font_sizes(line)
            names = _line_fontnames(line)
            text_raw = (line.get_text() or "").replace("\n", " ")
            text = _clean_text(text_raw)

            if not text:
                prev_plain = text
                last_y1 = y1
                last_sizes = sizes
                continue

            # Paragraph break heuristic
            if last_y1 is not None:
                avg_last = sum(last_sizes) / len(last_sizes) if last_sizes else 0.0
                avg_curr = sum(sizes) / len(sizes) if sizes else 0.0
                y_gap = abs(y1 - last_y1)
                font_changed = abs(avg_curr - avg_last) > 1.0
                prev_end_sentence = bool(re.search(r"[.!?]\s*$", prev_plain or ""))
                if (y_gap > 2.0 or font_changed) and prev_end_sentence and not code_block_open:
                    markdown_lines.append("")

            max_size = max(sizes) if sizes else 0.0

            total_chars = max(len(sizes), 1)
            bold_ratio = sum(1 for n in names if _is_bold_fontname(n)) / float(total_chars)
            italic_ratio = sum(1 for n in names if _is_italic_fontname(n)) / float(total_chars)
            is_boldish = bold_ratio > 0.3
            is_italicish = italic_ratio > 0.3

            word_count = len(text.split())
            ends_punct = bool(re.search(r"[.!?:]\s*$", text))
            is_bullet_like = _is_bullet_text(text)
            is_ordered = _is_ordered_item(text)
            contains_brackets = "[" in text and ")" in text

            # Skip header/footer text using fixed margins
            _, y0, _, y1 = it["bbox"]
            if y0 <= FOOTER_MARGIN or (page_height - y1) <= HEADER_MARGIN:
                continue
            
            header_level = self._get_header_level(max_size)
            header_ok = (
                header_level > 0
                and not is_bullet_like
                and not is_ordered
                and not contains_brackets
                and not ends_punct
                and word_count <= 12
            )

            indent_level = 0
            if indent_step > 0:
                indent_level = max(0, int(round((left - base_left) / indent_step)))

            # Code block detection (generic): accumulate consecutive indented/monospace-like lines
            is_code_like = self._is_monospace_family(names) or text.startswith("    ") or text.startswith("\t")
            if is_code_like:
                consecutive_code_lines += 1
            else:
                consecutive_code_lines = 0

            # Apply hyperlink mapping (wrap the whole line if intersects any link rect)
            text = self._apply_links_to_text(text, it["bbox"], link_rects)

            # Strikethrough detection via horizontal lines crossing the text bbox midline
            if self._has_strikethrough(it["bbox"], line_rects_miner):
                text = f"~~{text}~~"

            if _is_horizontal_rule(text):
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                markdown_lines.append("---")
                self._reset_list_state(list_state)
            elif header_ok:
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                candidate = f"{'#' * min(header_level, 3)} {text}"
                def _norm_header(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().lower()
                if candidate not in emitted_headers:
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                    emitted_headers.add(candidate)
                self._reset_list_state(list_state)
            elif self._is_numbered_list_item(text):
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                # Reset deeper counters when indent decreases
                for lvl in list(list_counters.keys()):
                    if lvl > indent_level:
                        del list_counters[lvl]
                list_counters[indent_level] = list_counters.get(indent_level, 0) + 1
                content = re.sub(r"^\s*\d+\s*[.)]\s*", "", text).strip()
                candidate = _indent(indent_level) + f"{list_counters[indent_level]}. {content}"
                if candidate != (last_emitted_line or ""):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                self._set_list_state(list_state, kind="ordered", indent=indent_level)
            elif self._is_bullet_point(text):
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                content = self._convert_bullet_to_markdown(text)
                candidate = _indent(indent_level) + content
                if candidate != (last_emitted_line or ""):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                self._set_list_state(list_state, kind="unordered", indent=indent_level)
            elif text.startswith(">") or (indent_level >= 2 and is_italicish):
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                candidate = ("> " + text) if not text.startswith(">") else text
                if candidate != (last_emitted_line or ""):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                self._reset_list_state(list_state)
            elif False:
                pass
            elif is_code_like:
                if not code_block_open and consecutive_code_lines >= 2:
                    markdown_lines.append("```")
                    code_block_open = True
                if code_block_open:
                    candidate = text_raw.strip()
                    if candidate != (last_emitted_line or ""):
                        markdown_lines.append(candidate)
                        last_emitted_line = candidate
                else:
                    # Single code-like line: inline fence
                    candidate = f"`{text}`"
                    if candidate != (last_emitted_line or ""):
                        markdown_lines.append(candidate)
                        last_emitted_line = candidate
                self._reset_list_state(list_state)
            else:
                decorated = self._auto_linkify(text)
                if is_boldish and is_italicish:
                    decorated = f"***{decorated}***"
                elif is_boldish:
                    decorated = f"**{decorated}**"
                elif is_italicish:
                    decorated = f"*{decorated}*"
                if decorated != (last_emitted_line or ""):
                    markdown_lines.append(decorated)
                    last_emitted_line = decorated
                self._reset_list_state(list_state)

            # Manage list-intro activation
            if text.endswith(":") and not is_bullet_like and not is_ordered and not header_ok:
                list_intro_active = True
                list_intro_indent = indent_level
            elif text == "":
                list_intro_active = False

            prev_plain = text
            last_y1 = y1
            last_sizes = sizes

        if code_block_open:
            markdown_lines.append("```")

        return "\n".join(markdown_lines) + "\n"

    @staticmethod
    def _get_header_level(font_size: float) -> int:
        """Determine header level based on absolute font size thresholds."""
        if font_size > 24:
            return 1
        if font_size > 20:
            return 2
        if font_size > 18:
            return 3
        if font_size > 16:
            return 4
        if font_size > 14:
            return 5
        if font_size > 12:
            return 6
        return 0

    @staticmethod
    def _auto_linkify(text: str) -> str:
        url_pattern = re.compile(r"(https?://[^\s)]+)")
        def repl(match: re.Match[str]) -> str:
            url = match.group(1)
            return f"[{url}]({url})"
        return url_pattern.sub(repl, text)

    @staticmethod
    def _is_monospace_family(fontnames: List[str]) -> bool:
        lowers = [n.lower() for n in fontnames]
        return any(
            any(tag in n for tag in ("mono", "courier", "code", "consolas", "menlo"))
            for n in lowers
        )

    # Removed _looks_like_code(): generic detection is handled inline

    @staticmethod
    def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
        if not table:
            return ""
        try:
            norm = [
                [
                    ""
                    if cell is None
                    else FastMarkdown._normalize_cell_text(str(cell))
                    for cell in row
                ]
                for row in table
            ]
            if not norm or not norm[0]:
                return ""
            col_widths = [max(len(cell) for cell in col) for col in zip(*norm)]
            md_lines: List[str] = []
            for i, row in enumerate(norm):
                # Bold header row cells for readability
                render_row = [
                    (f"**{cell}**" if i == 0 and cell else cell) for cell in row
                ]
                formatted = [
                    render_row[j].ljust(col_widths[j] + (4 if i == 0 else 0))
                    for j in range(len(render_row))
                ]
                md_lines.append("| " + " | ".join(formatted) + " |")
                if i == 0:
                    md_lines.append("|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|")
            return "\n".join(md_lines)
        except Exception:
            return ""

    @staticmethod
    def _normalize_cell_text(text: str) -> str:
        """Normalize PDF extraction artifacts in table cells."""
        if not text:
            return ""
        
        # Collapse excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common PDF artifacts like zero-width characters
        cleaned = re.sub(r'[\u200b-\u200d\ufeff]', '', cleaned)
        
        # Fix common encoding issues
        cleaned = cleaned.replace('\ufeff', '')  # BOM
        cleaned = cleaned.replace('\u00a0', ' ')  # Non-breaking space
        
        return cleaned

    @staticmethod
    def _normalize_task_checkbox(text: str) -> str:
        # Normalize variations like [x], [X], [✓], [✔] into markdown task list
        text = re.sub(r"^-\s*\[(x|X|✓|✔)\]\s*", "- [x] ", text)
        text = re.sub(r"^-\s*\[\s*\]\s*", "- [ ] ", text)
        return text

    @staticmethod
    def _bbox_midline(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = b
        y = (y0 + y1) / 2.0
        return (x0, y, x1, y)

    def _has_strikethrough(
        self, text_bbox: Tuple[float, float, float, float], line_rects: List[Tuple[float, float, float, float]]
    ) -> bool:
        x0, y0, x1, y1 = text_bbox
        mid = self._bbox_midline(text_bbox)
        # Consider any line whose vertical span overlaps the midline of text and horizontally overlaps
        for lx0, ly0, lx1, ly1 in line_rects:
            # Convert line bbox to a flat horizontal segment approximation
            ly_mid = (ly0 + ly1) / 2.0
            if (min(x1, lx1) - max(x0, lx0)) > 4.0 and abs(ly_mid - ((y0 + y1) / 2.0)) <= 2.0:
                return True
        return False

    @staticmethod
    def _apply_links_to_text(
        text: str,
        bbox: Tuple[float, float, float, float],
        link_rects: List[Tuple[Tuple[float, float, float, float], str]],
    ) -> str:
        for rect, uri in link_rects:
            x0, y0, x1, y1 = bbox
            if not (x1 <= rect[0] or x0 >= rect[2] or y1 <= rect[1] or y0 >= rect[3]):
                # Wrap whole line as a link
                clean = text.strip()
                if clean:
                    return f"[{clean}]({uri})"
        return text


    @staticmethod
    def _reset_list_state(state: ListState) -> None:
        """Reset list parsing state."""
        state.kind = "none"
        state.indent_level = 0

    @staticmethod
    def _set_list_state(state: ListState, kind: str, indent: int) -> None:
        """Set list parsing state."""
        state.kind = kind
        state.indent_level = indent

    @staticmethod
    def _post_process(content: str) -> str:
        """Basic whitespace cleanup."""
        try:
            # Collapse excessive newlines
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = re.sub(r"^\n", "", content)
            
            # Collapse excessive spaces but preserve leading indentation
            lines = content.split("\n")
            processed: List[str] = []
            for ln in lines:
                m = re.match(r"^(\s*)(.*)$", ln)
                if m:
                    lead, rest = m.group(1), m.group(2)
                    rest = re.sub(r" {2,}", " ", rest)
                    processed.append(lead + rest)
                else:
                    processed.append(ln)
            
            content = "\n".join(processed)
            content = re.sub(r"\s*(---\n)+", "\n\n---\n", content)
            
            return content
        except Exception:
            return content


def main() -> List[str]:
    parser = argparse.ArgumentParser(
        description="Extract markdown-formatted content from a PDF using pdfminer.six."
    )
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file")
    args = parser.parse_args()

    # Minimal logging setup when running as a script
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    extractor = FastMarkdown(pdf_path)
    pages = extractor.extract()
    return pages


if __name__ == "__main__":
    main()


