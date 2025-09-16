import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer, LTTextLine
import pdfplumber
from tqdm import tqdm


# Hardcoded configuration (no external config dependency)
BULLET_CHARS = "•◦▪▫●○*·–—-"
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


@dataclass
class ListState:
    active: bool = False
    kind: str = "none"  # 'none' | 'ordered' | 'unordered'
    indent_level: int = 0
    


@dataclass
class PageFeatures:
    """Container for pre-extracted page features from pdfplumber."""

    tables: List[Dict[str, Any]]
    hyperlinks: List[Dict[str, Any]]
    horizontal_lines: List[Tuple[float, float, float, float]]


class FontSizeAnalyzer:
    """Analyze font sizes on a page to infer header levels and body size."""

    def build_header_level_map(self, lines: List[LTTextLine]) -> Dict[float, int]:
        sizes: List[float] = []
        for line in lines:
            sizes.extend(_line_font_sizes(line))
        unique = sorted({s for s in sizes}, reverse=True)
        level_map: Dict[float, int] = {}
        for idx, size in enumerate(unique[:6]):
            level_map[size] = idx + 1
        return level_map

    def estimate_body_size(self, lines: List[LTTextLine]) -> float:
        sizes: List[float] = []
        for line in lines:
            sizes.extend(_line_font_sizes(line))
        if not sizes:
            return 12.0
        return float(median(sorted(sizes)))


 


class FastMarkdown:
    """Extract markdown-formatted content from a PDF using pdfminer.six for text
    and pdfplumber for tables.

    Responsibilities:
      - Iterate PDF pages with layout analysis
      - Infer basic structure: headers, paragraphs, lists
      - Build per-page markdown only (no combined output)
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self._font_analyzer = FontSizeAnalyzer()

    def extract(self) -> List[str]:
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
                header_map = self._font_analyzer.build_header_level_map(lines)
                body_size = self._font_analyzer.estimate_body_size(lines)
                page_height = float(layout.bbox[3])
                pf = page_features[page_index] if page_index < len(page_features) else PageFeatures([], [], [])
                page_tables = pf.tables
                items = self._build_items(lines, page_tables, page_height)
                page_md = self._build_page_markdown_from_items(
                    items,
                    header_map,
                    body_size,
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
        """Pre-extract tables, hyperlinks, and line segments per page using pdfplumber.

        Returns:
            List[PageFeatures]: Sequence aligned with pdf pages, containing features.
        """
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
        header_map: Dict[float, int],
        body_size: float,
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
        emitted_bullet_labels: set[str] = set()
        last_emitted_ordered_label: Optional[str] = None

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
            level = self._header_level_for_size(max_size, header_map)

            total_chars = max(len(sizes), 1)
            bold_ratio = sum(1 for n in names if _is_bold_fontname(n)) / float(total_chars)
            italic_ratio = sum(1 for n in names if _is_italic_fontname(n)) / float(total_chars)
            is_boldish = bold_ratio > 0.3
            is_italicish = italic_ratio > 0.3

            word_count = len(text.split())
            ends_punct = bool(re.search(r"[.!?:]\s*$", text))
            size_delta = float(max_size) - float(body_size)
            is_bullet_like = _is_bullet_text(text)
            is_ordered = _is_ordered_item(text)
            contains_brackets = "[" in text and ")" in text

            header_ok = (
                level > 0
                and not is_bullet_like
                and not is_ordered
                and not contains_brackets
                and not ends_punct
                and word_count <= 12
                and size_delta >= 2.0
            )

            indent_level = 0
            if indent_step > 0:
                indent_level = max(0, int(round((left - base_left) / indent_step)))

            # Code block heuristics
            looks_code = self._looks_like_code(text, names)
            if looks_code:
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
                hlevel = min(level, 3)
                candidate = f"{'#' * hlevel} {text}"
                def _norm_header(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().lower()
                if candidate not in emitted_headers and (
                    not last_emitted_ordered_label or _norm_header(text) != _norm_header(last_emitted_ordered_label)
                ):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                    emitted_headers.add(candidate)
                self._reset_list_state(list_state)
            elif is_bullet_like:
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                # Normalize task list checkboxes if present
                content = self._normalize_task_checkbox(_convert_bullet(text))
                content_text = re.sub(r"^-[\s]*", "", content).strip()
                base_label = content_text.split(":", 1)[0].strip()
                has_desc = ":" in content_text
                def _norm_label(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().lower()
                # Decide nesting level: at least one deeper than ordered if inside ordered and not more indented already
                nest_level = indent_level
                if list_state.kind == "ordered" and nest_level <= list_state.indent_level:
                    nest_level = list_state.indent_level + 1
                # Skip duplicate label-only bullets and duplicates of the last ordered label
                if (
                    (not has_desc and _norm_label(base_label) in { _norm_label(lbl) for lbl in emitted_bullet_labels })
                    or (last_emitted_ordered_label and _norm_label(base_label) == _norm_label(last_emitted_ordered_label))
                ):
                    pass
                else:
                    candidate = _indent(nest_level) + content
                    if candidate != (last_emitted_line or ""):
                        markdown_lines.append(candidate)
                        last_emitted_line = candidate
                    if has_desc:
                        emitted_bullet_labels.add(base_label)
                self._set_list_state(list_state, kind="unordered", indent=nest_level)
            elif is_ordered:
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                normalized = _normalize_ordered_item(text)
                candidate = _indent(indent_level) + normalized
                if candidate != (last_emitted_line or ""):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                # Track last ordered label for de-duplication
                last_emitted_ordered_label = re.sub(r"^\s*\d+\.\s*", "", normalized).strip()
                self._set_list_state(list_state, kind="ordered", indent=indent_level)
            elif text.startswith(">") or (indent_level >= 2 and is_italicish):
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                candidate = ("> " + text) if not text.startswith(">") else text
                if candidate != (last_emitted_line or ""):
                    markdown_lines.append(candidate)
                    last_emitted_line = candidate
                self._reset_list_state(list_state)
            elif list_intro_active:
                # Treat as bullet following a list-intro ending with ':'
                if code_block_open:
                    markdown_lines.append("```")
                    code_block_open = False
                base_label = text.split(":", 1)[0].strip()
                def _norm_label2(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().lower()
                # Skip if this line is just repeating the last ordered label
                if last_emitted_ordered_label and _norm_label2(base_label) == _norm_label2(last_emitted_ordered_label):
                    pass
                else:
                    candidate = _indent(max(indent_level, list_intro_indent)) + "- " + text
                    if candidate != (last_emitted_line or ""):
                        markdown_lines.append(candidate)
                        last_emitted_line = candidate
                    emitted_bullet_labels.add(base_label)
                self._set_list_state(
                    list_state, kind="unordered", indent=max(indent_level, list_intro_indent)
                )
            elif looks_code:
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
                # Suppress stray label-only echoes that match previously emitted bullet labels
                def _norm_plain(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().lower()
                if (
                    _norm_plain(decorated) in { _norm_plain(lbl) for lbl in emitted_bullet_labels }
                ):
                    pass
                elif decorated != (last_emitted_line or ""):
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

    def _looks_like_code(self, text: str, fontnames: List[str]) -> bool:
        if self._is_monospace_family(fontnames) and len(text) >= 10:
            return True
        patterns = [
            r"^(?:from|import|def|class|if|for|while|try|except|with)\b",
            r"[{;}\[\]()]",
            r"^\s{2,}",
            r"^(?:#include|using|namespace|template|public|private|protected)\b",
            r"^<\w+.*>$",
        ]
        return any(re.search(p, text) for p in patterns)

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
        """Normalize common PDF glyph duplication artifacts and whitespace in table cells.

        Compress words that appear fully doubled per character (e.g.,
        'FFeeaattuurree' -> 'Feature'). Also collapse internal whitespace/newlines.
        """
        if not text:
            return ""

        def compress_word(word: str) -> str:
            if len(word) < 2:
                return word
            w = " ".join(word.split())  # collapse any internal whitespace/newlines
            if len(w) % 2 == 0:
                pairs = [w[i : i + 2] for i in range(0, len(w), 2)]
                if all(len(p) == 2 and p[0] == p[1] for p in pairs):
                    return "".join(p[0] for p in pairs)
            return w

        cleaned = " ".join(text.split())
        return " ".join(compress_word(tok) for tok in cleaned.split())

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
    def _header_level_for_size(size: float, level_map: Dict[float, int]) -> int:
        if not level_map:
            return 0
        if size in level_map:
            return level_map[size]
        larger = [s for s in level_map if s >= size]
        smaller = [s for s in level_map if s < size]
        if larger:
            return level_map[min(larger, key=lambda s: s - size)]
        if smaller:
            return level_map[max(smaller, key=lambda s: size - s)]
        return 0

    @staticmethod
    def _reset_list_state(state: ListState) -> None:
        state.active = False
        state.kind = "none"
        state.indent_level = 0

    @staticmethod
    def _set_list_state(state: ListState, kind: str, indent: int) -> None:
        state.active = True
        state.kind = kind
        state.indent_level = indent

    @staticmethod
    def _post_process(content: str) -> str:
        try:
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = re.sub(r"^\n", "", content)
            # Collapse excessive spaces but preserve leading indentation
            lines = content.split("\n")
            processed: List[str] = []
            for ln in lines:
                m = re.match(r"^(\s*)(.*)$", ln)
                if m:
                    lead, rest = m.group(1), m.group(2)
                else:
                    lead, rest = "", ln
                rest = re.sub(r" {2,}", " ", rest)
                processed.append(lead + rest)
            content = "\n".join(processed)
            content = re.sub(r"\s*(---\n)+", "\n\n---\n", content)
            # Suppress label-only bullet echoes following label-with-description bullets
            lines = content.split("\n")
            seen_labels: set[str] = set()
            def _norm(s: str) -> str:
                return re.sub(r"\W+", " ", s).strip().lower()
            out: List[str] = []
            for ln in lines:
                m_desc = re.match(r"^\s*-\s+([^:]+):\s*", ln)
                if m_desc:
                    seen_labels.add(_norm(m_desc.group(1)))
                    out.append(ln)
                    continue
                m_label = re.match(r"^\s*-\s+([^:]+)\s*$", ln)
                if m_label and _norm(m_label.group(1)) in seen_labels:
                    # Skip echo line
                    continue
                out.append(ln)
            return "\n".join(out)
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


