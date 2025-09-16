import fitz  # PyMuPDF
import pdfplumber
import re
import yaml
import os
import logging
import traceback
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import argparse
from typing import Any, Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# Load configuration (required)
with open(Path("config/config.yaml").resolve(), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class PDFExtractor(ABC):
    """Abstract base class for PDF extraction."""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{Path(__file__).stem}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(self):
        """Abstract method for extracting content from PDF."""
        pass


class MarkdownPDFExtractor(PDFExtractor):
    """Class for extracting markdown-formatted content from PDF."""

    BULLET_POINTS = "•◦▪▫●○*·–—-"

    def __init__(self, pdf_path):
        super().__init__(pdf_path)
        self.pdf_filename = Path(pdf_path).stem
        Path(config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
        

    def extract(self):
        try:
            markdown_content, markdown_pages = self.extract_markdown()
            self.save_markdown(markdown_content)
            self.logger.info(
                f"Markdown content has been saved to {Path(config['OUTPUT_DIR'])}/{self.pdf_filename}.md"
            )
            return markdown_content, markdown_pages

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            self.logger.exception(traceback.format_exc())
            return "", []

    def extract_markdown(self):
        """Main method to extract markdown from PDF."""
        try:
            doc = fitz.open(self.pdf_path)
            markdown_content = ""
            markdown_pages = []
            tables = self.extract_tables()
            prev_line = ""

            num_pages = len(doc)
            for page_num, page in enumerate(doc):
                self.logger.info(f"Processing page {page_num + 1}")
                page_content = ""
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                page_width = page.rect.width
                links = self.extract_links(page)

                # Prepare tables for this page
                page_tables = [t for t in tables if t["page"] == page.number]

                # Build combined items: text blocks (non-overlapping with tables) + tables
                items: List[Dict[str, Any]] = []

                def rects_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
                    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

                table_bboxes = [tuple(tb["bbox"]) for tb in page_tables]

                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                    # Skip text blocks overlapping any table bbox
                    if any(rects_intersect(bbox, tb) for tb in table_bboxes):
                        continue
                    items.append({
                        "type": "text",
                        "bbox": bbox,
                        "data": block,
                    })

                for tb in page_tables:
                    items.append({
                        "type": "table",
                        "bbox": tuple(tb["bbox"]),
                        "data": tb["content"],
                    })

                header_level_map = self._build_header_level_map(blocks)
                body_font_size = self._estimate_body_font_size(blocks)

                # Persist list state across text blocks so nested items work across block boundaries
                list_state = {
                    "active": False,            # whether we're in a list-intro state
                    "kind": "none",            # 'none' | 'ordered' | 'unordered'
                    "indent_level": 0,         # last seen list indent level
                    "prev_line_left": None,    # last line left x for proximity heuristics
                }
                for item in self._sort_items_by_reading_order(items, page_width):
                    if item["type"] == "text":
                        page_content += self.process_text_block(
                            item["data"], page_height, links, prev_line, header_level_map, body_font_size, list_state
                        )
                    else:
                        page_content += "\n\n" + self.table_to_markdown(item["data"]) + "\n\n"

                markdown_pages.append(self.post_process_markdown(page_content))
                markdown_content += page_content
                if page_num < num_pages - 1:
                    markdown_content += config["PAGE_DELIMITER"]

            markdown_content = self.post_process_markdown(markdown_content)
            return markdown_content, markdown_pages
        except Exception as e:
            self.logger.error(f"Error extracting markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return "", []

    def extract_tables(self):
        """Extract tables and their bounding boxes using pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    # Detect table structures with bounding boxes
                    found = page.find_tables()
                    for t in found:
                        content = t.extract()
                        bbox = t.bbox  # (x0, top, x1, bottom)
                        if content:
                            tables.append({
                                "page": page_number,
                                "content": content,
                                "bbox": bbox,
                            })
            self.logger.info(f"Extracted {len(tables)} tables from the PDF.")
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            self.logger.exception(traceback.format_exc())
        return tables

    def table_to_markdown(self, table):
        """Convert a table to markdown format."""
        if not table:
            return ""

        try:
            table = [
                [self._normalize_cell_text("" if cell is None else str(cell).strip()) for cell in row]
                for row in table
            ]
            if not table or not table[0]:
                return ""

            col_widths = [max(len(cell) for cell in col) for col in zip(*table)]

            markdown = ""
            for i, row in enumerate(table):
                formatted_row = [
                    cell.ljust(col_widths[j]) for j, cell in enumerate(row)
                ]
                markdown += "| " + " | ".join(formatted_row) + " |\n"

                if i == 0:
                    markdown += (
                        "|"
                        + "|".join(["-" * (width + 2) for width in col_widths])
                        + "|\n"
                    )

            return markdown
        except Exception as e:
            self.logger.error(f"Error converting table to markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def _normalize_cell_text(self, text: str) -> str:
        """Normalize common PDF glyph duplication artifacts seen in table cells.

        Compress words that appear to be fully doubled per character (e.g.,
        'FFeeaattuurree' -> 'Feature'). Leave normal words intact.
        """
        if not text:
            return text

        def compress_word(word: str) -> str:
            if len(word) < 2:
                return word
            # Remove newlines within cells before processing
            w = " ".join(word.splitlines())
            if len(w) % 2 == 0:
                pairs = [w[i : i + 2] for i in range(0, len(w), 2)]
                if all(len(p) == 2 and p[0] == p[1] for p in pairs):
                    return "".join(p[0] for p in pairs)
            return w

        return " ".join(compress_word(tok) for tok in text.split())

    def _sort_items_by_reading_order(self, items: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
        """Sort items (text blocks and tables) by reading order: top-to-bottom with
        left-to-right tie-break within the same horizontal band.
        """
        if not items:
            return []

        def key_func(it: Dict[str, Any]) -> Tuple[int, float]:
            x0, y0, _, _ = it["bbox"]
            band = int(round(y0 / 12.0))  # 12px vertical tolerance bands
            return (band, x0)

        return sorted(items, key=key_func)

    def _sort_text_blocks_reading_order(self, blocks: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
        """Return text blocks ordered by columns (left-to-right, then top-to-bottom).

        This heuristic clusters text blocks into vertical columns using their
        horizontal centers and a gap threshold relative to the page width.
        Within each column, blocks are sorted by their top (y0) position.
        """
        text_blocks = [b for b in blocks if b.get("type") == 0]
        if not text_blocks:
            return []

        prepared = []  # tuples: (center_x, y0, block)
        for b in text_blocks:
            x0, y0, x1, _ = b.get("bbox", (0, 0, 0, 0))
            center_x = (x0 + x1) / 2.0
            prepared.append((center_x, y0, b))

        prepared.sort(key=lambda t: t[0])  # sort by center_x

        # Split into columns by the largest center_x gap if significant
        centers = [t[0] for t in prepared]
        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
        gaps.sort(reverse=True)

        columns: List[Dict[str, Any]] = []
        if gaps and gaps[0][0] > max(40.0, page_width * 0.15):
            split_idx = gaps[0][1] + 1
            col_items = [prepared[:split_idx], prepared[split_idx:]]
            for items in col_items:
                if not items:
                    continue
                x_center = sum(i[0] for i in items) / len(items)
                columns.append({"x_center": x_center, "blocks": items})
            columns.sort(key=lambda c: c["x_center"])  # left-to-right
        else:
            # Single column: simple top-to-bottom
            return [t[2] for t in sorted(prepared, key=lambda t: (t[1], t[0]))]

        for col in columns:
            col["blocks"].sort(key=lambda t: t[1])

        ordered = [t[2] for col in columns for t in col["blocks"]]
        return ordered

    def clean_text(self, text):
        """Clean the given text by removing extra spaces."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def apply_formatting(self, text, flags):
        """Apply markdown formatting to the given text based on flags."""
        text = text.strip()
        if not text:
            return text

        is_bold = flags & 2**4
        is_italic = flags & 2**1
        is_monospace = flags & 2**3
        is_superscript = flags & 2**0
        is_subscript = flags & 2**5

        if is_monospace:
            text = f"`{text}`"
        elif is_superscript and not bool(re.search(r"\s+", text)):
            text = f"^{text}^"
        elif is_subscript and not bool(re.search(r"\s+", text)):
            text = f"~{text}~"

        if is_bold and is_italic:
            text = f"***{text}***"
        elif is_bold:
            text = f"**{text}**"
        elif is_italic:
            text = f"*{text}*"

        return f" {text} "

    def is_bullet_point(self, text):
        """Check if the given text is a bullet point."""
        return text.strip().startswith(tuple(self.BULLET_POINTS))

    def convert_bullet_to_markdown(self, text):
        """Convert a bullet point to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(f"^[{re.escape(self.BULLET_POINTS)}]\\s*", "- ", text)

    def is_numbered_list_item(self, text):
        """Check if the given text is a numbered list item."""
        return bool(re.match(r"^\d+\s{0,3}[.)]", text.strip()))

    def convert_numbered_list_to_markdown(self, text, list_counter):
        """Convert a numbered list item to markdown format."""
        text = re.sub(r"^\s*", "", text)
        return re.sub(r"^\d+\s{0,3}[.)]", f"{list_counter}. ", text)

    def is_horizontal_line(self, text):
        """Check if the given text represents a horizontal line."""
        return bool(re.match(r"^[_-]+$", text.strip()))

    def extract_links(self, page):
        """Extract links from the given page."""
        links = []
        try:
            for link in page.get_links():
                if link["kind"] == 2:  # URI link
                    links.append({"rect": link["from"], "uri": link["uri"]})
            self.logger.info(f"Extracted {len(links)} links from the page.")
        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            self.logger.exception(traceback.format_exc())
        return links

    def detect_code_block(self, prev_line, current_line):
        """Detect if the current line starts a code block."""
        patterns = {
            "python": [
                (
                    r"^(?:from|import)\s+\w+",
                    r"^(?:from|import|def|class|if|for|while|try|except|with)\s",
                ),
                (r"^(?:def|class)\s+\w+", r"^\s{4}"),
                (r"^\s{4}", r"^\s{4,}"),
            ],
            "javascript": [
                (
                    r"^(?:function|const|let|var)\s+\w+",
                    r"^(?:function|const|let|var|if|for|while|try|catch|class)\s",
                ),
                (r"^(?:if|for|while)\s*\(", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "html": [
                (
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                    r"^<(!DOCTYPE|html|head|body|div|p|a|script|style)",
                ),
                (r"^<\w+.*>$", r"^\s{2,}<"),
                (r"^\s{2,}<", r"^\s{2,}<"),
            ],
            "shell": [
                (r"^(?:\$|\#)\s", r"^(?:\$|\#)\s"),
                (r"^[a-z_]+\s*=", r"^[a-z_]+\s*="),
            ],
            "bash": [
                (
                    r"^(?:#!/bin/bash|alias|export|source)\s",
                    r"^(?:#!/bin/bash|alias|export|source|echo|read|if|for|while|case|function)\s",
                ),
                (r"^(?:if|for|while|case|function)\s", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "cpp": [
                (
                    r"^#include\s*<",
                    r"^(?:#include|using|namespace|class|struct|enum|template|typedef)\s",
                ),
                (r"^(?:class|struct|enum)\s+\w+", r"^\s{2,}"),
                (r"^\s{2,}", r"^\s{2,}"),
            ],
            "java": [
                (
                    r"^(?:import|package)\s+\w+",
                    r"^(?:import|package|public|private|protected|class|interface|enum)\s",
                ),
                (r"^(?:public|private|protected)\s+class\s+\w+", r"^\s{4,}"),
                (r"^\s{4,}", r"^\s{4,}"),
            ],
            "json": [
                (r"^\s*{", r'^\s*["{[]'),
                (r'^\s*"', r'^\s*["}],?$'),
                (r"^\s*\[", r"^\s*[}\]],?$"),
            ],
        }

        for lang, pattern_pairs in patterns.items():
            for prev_pattern, curr_pattern in pattern_pairs:
                if re.match(prev_pattern, prev_line.strip()) and re.match(
                    curr_pattern, current_line.strip()
                ):
                    return lang

        return None

    def process_text_block(self, block, page_height, links, prev_line, header_level_map, body_font_size, list_state):
        """Process a text block and convert it to markdown, with nested lists.

        Uses line x-positions to infer indent levels for nested unordered and
        ordered lists. Also applies cautious header detection.
        """
        try:
            block_rect = block["bbox"]
            if block_rect[1] < 15 or block_rect[3] > page_height - 15:
                pass

            # Estimate indentation baseline and step
            line_lefts = [ln.get("bbox", [0, 0, 0, 0])[0] for ln in block.get("lines", [])]
            base_left = min(line_lefts) if line_lefts else 0.0
            indent_step = self._estimate_indent_step(line_lefts)

            markdown_content = ""
            last_y1 = None
            last_font_size = None
            # Recover cross-block list state so we can continue lists across text block boundaries
            prev_list_kind: str = list_state.get("kind", "none")  # 'none' | 'ordered' | 'unordered'
            prev_indent_level: int = int(list_state.get("indent_level", 0))
            prev_line_left: Optional[float] = list_state.get("prev_line_left")
            list_intro_active = bool(list_state.get("active", False))

            for line in block["lines"]:
                line_left = line.get("bbox", [0, 0, 0, 0])[0]
                indent_level = 0
                if indent_step > 0:
                    indent_level = max(0, int(round((line_left - base_left) / indent_step)))

                # Build visible text for the line with formatting and link mapping
                line_text = ""
                curr_font_size = [span["size"] for span in line.get("spans", [])]
                max_size = max(curr_font_size) if curr_font_size else 0
                header_level_line = self._header_level_for_line(max_size, header_level_map)

                total_spans = max(len(line.get("spans", [])), 1)
                spans_with_max = 0
                bold_spans = 0

                for span in line.get("spans", []):
                    text = self.clean_text(span.get("text", ""))
                    font_size = span.get("size", 0)
                    flags = span.get("flags", 0)
                    span_rect = span.get("bbox", [0, 0, 0, 0])

                    if self.is_horizontal_line(text):
                        line_text += "\n---\n"
                        continue

                    if text:
                        text = self.apply_formatting(text, flags)
                        if abs(float(font_size) - float(max_size)) <= 0.1:
                            spans_with_max += 1
                        if (flags & (2**4)) != 0:  # bold
                            bold_spans += 1

                    for link in links:
                        if fitz.Rect(span_rect).intersects(link["rect"]):
                            text = f"[{text.strip()}]({link['uri']})"
                            break

                    line_text += text

                # Determine paragraph breaks by y-gap or font size change, but avoid breaking mid-sentence
                if last_y1 is not None:
                    avg_last_font_size = (
                        sum(last_font_size) / len(last_font_size)
                        if last_font_size
                        else 0
                    )
                    avg_current_font_size = sum(curr_font_size) / len(curr_font_size) if curr_font_size else 0
                    font_size_changed = abs(avg_current_font_size - avg_last_font_size) > 1
                    y_gap = abs(line.get("bbox", [0, 0, 0, 0])[3] - last_y1)
                    prev_ended_sentence = bool(re.search(r"[.!?]\s*$", prev_line or ""))
                    if (y_gap > 2 or font_size_changed) and prev_ended_sentence:
                        markdown_content += "\n"

                candidate_line = self.clean_text(line_text)
                is_bullet_like = self.is_bullet_point(candidate_line)
                ordered_match = re.match(r"^(\d+)\s*[\.)]\s*(.*)$", candidate_line)
                ends_with_punct = bool(re.search(r"[.!?:]\s*$", candidate_line))
                contains_link = ("[" in candidate_line and ")" in candidate_line)
                word_count = len(candidate_line.split())
                bold_ratio = bold_spans / float(total_spans)
                uniform_ratio = spans_with_max / float(total_spans)
                size_delta = float(max_size) - float(body_font_size)
                header_eligible = (
                    header_level_line > 0
                    and not is_bullet_like
                    and not ordered_match
                    and not contains_link
                    and not ends_with_punct
                    and word_count <= 12
                    and size_delta >= 2.0  # ensure larger than body size to avoid bold-as-header
                )

                # Heuristic for x-position closeness for sibling list items
                x_close_threshold = max(6.0, indent_step * 0.6)
                x_is_close = (
                    prev_line_left is not None and abs(line_left - prev_line_left) <= x_close_threshold
                )

                if header_eligible:
                    level = min(header_level_line, 3)
                    markdown_content += f"\n{'#' * level} {candidate_line}\n\n"
                    # reset list state when encountering a header
                    list_intro_active = False
                    list_state["active"] = False
                    prev_list_kind = "none"
                    prev_indent_level = 0
                    prev_line_left = None
                elif is_bullet_like:
                    content = self.convert_bullet_to_markdown(candidate_line)
                    content = re.sub(r"^\-\s*", "- ", content)
                    markdown_content += ("  " * indent_level) + content + "\n"
                    list_intro_active = False
                    prev_list_kind = "unordered"
                    prev_indent_level = indent_level
                elif ordered_match:
                    num = ordered_match.group(1)
                    rest = ordered_match.group(2)
                    normalized = f"{num}. {rest}".strip()
                    markdown_content += ("  " * indent_level) + normalized + "\n"
                    list_intro_active = False
                    prev_list_kind = "ordered"
                    prev_indent_level = indent_level
                elif list_intro_active and candidate_line:
                    # Treat lines after a list-intro as bullets at current indent
                    markdown_content += ("  " * indent_level) + "- " + candidate_line + "\n"
                    prev_list_kind = "unordered"
                    prev_indent_level = indent_level
                elif candidate_line and prev_list_kind == "unordered" and (
                    indent_level == prev_indent_level or x_is_close
                ):
                    # Sibling bullet at same indent when previous was an unordered list item
                    markdown_content += ("  " * indent_level) + "- " + candidate_line + "\n"
                    prev_list_kind = "unordered"
                    prev_indent_level = indent_level
                elif candidate_line and prev_list_kind == "ordered" and (
                    indent_level > prev_indent_level or indent_level == prev_indent_level or x_is_close
                ):
                    # Promote lines following an ordered item to nested bullets either when indented
                    # more than the ordered marker, or when aligned to the same x-position.
                    target_indent = indent_level if indent_level > prev_indent_level else (prev_indent_level + 1)
                    markdown_content += ("  " * target_indent) + "- " + candidate_line + "\n"
                    prev_list_kind = "unordered"
                    prev_indent_level = target_indent
                elif candidate_line and prev_list_kind in ("ordered", "unordered") and indent_level > prev_indent_level:
                    # Implicit nested bullet under a preceding list item
                    markdown_content += ("  " * indent_level) + "- " + candidate_line + "\n"
                    prev_list_kind = "unordered"
                    prev_indent_level = indent_level
                else:
                    markdown_content += f"{candidate_line}\n"
                    if candidate_line:
                        prev_list_kind = "none"

                last_font_size = curr_font_size
                last_y1 = line.get("bbox", [0, 0, 0, 0])[3]
                prev_line = candidate_line
                prev_line_left = line_left

                # Manage list-intro activation based on current line
                if candidate_line.endswith(":") and not is_bullet_like and not ordered_match:
                    list_intro_active = True
                elif candidate_line == "" or candidate_line.endswith((".", "!", "?")):
                    list_intro_active = False

            # Persist cross-block state so subsequent blocks can continue the list context
            list_state["active"] = list_intro_active
            list_state["kind"] = prev_list_kind
            list_state["indent_level"] = prev_indent_level
            list_state["prev_line_left"] = prev_line_left

            return markdown_content + "\n"
        except Exception as e:
            self.logger.error(f"Error processing text block: {e}")
            self.logger.exception(traceback.format_exc())
            return ""

    def _estimate_indent_step(self, positions: List[float]) -> float:
        """Estimate a reasonable indentation step from line left positions."""
        if not positions:
            return 18.0
        uniq = sorted(set(round(p, 1) for p in positions))
        diffs = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
        diffs = [d for d in diffs if d > 1.0]
        if not diffs:
            return 18.0
        diffs.sort()
        # use median diff, clamped to a sensible range
        mid = len(diffs) // 2
        step = diffs[mid] if len(diffs) % 2 == 1 else (diffs[mid - 1] + diffs[mid]) / 2.0
        return max(8.0, min(36.0, step))

    def get_header_level(self, font_size):
        """Determine header level based on font size."""
        if font_size > 24:
            return 1
        elif font_size > 20:
            return 2
        elif font_size > 18:
            return 3
        elif font_size > 16:
            return 4
        elif font_size > 14:
            return 5
        elif font_size > 12:
            return 6
        else:
            return 0

    def _build_header_level_map(self, blocks: List[Dict[str, Any]]) -> Dict[float, int]:
        """Build a mapping from font size to header level using the largest sizes on the page.

        The largest unique font size becomes H1, the next H2, and so on up to H6.
        Sizes below or equal to the 6th largest are not headers (mapped to 0).
        """
        sizes: List[float] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        sizes.append(float(size))
        unique_sizes = sorted(set(sizes), reverse=True)
        level_map: Dict[float, int] = {}
        for idx, size in enumerate(unique_sizes[:6]):
            level_map[size] = idx + 1  # 1..6
        return level_map

    def _estimate_body_font_size(self, blocks: List[Dict[str, Any]]) -> float:
        """Estimate the typical body font size on the page (median of span sizes)."""
        sizes: List[float] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        sizes.append(float(size))
        if not sizes:
            return 12.0
        sizes.sort()
        mid = len(sizes) // 2
        if len(sizes) % 2 == 1:
            return sizes[mid]
        return (sizes[mid - 1] + sizes[mid]) / 2.0

    def _header_level_for_line(self, font_size: float, level_map: Dict[float, int]) -> int:
        """Return header level for a line based on its max span size, using nearest match."""
        if not level_map:
            return 0
        # Exact match first
        if font_size in level_map:
            return level_map[font_size]
        # Nearest larger size, else nearest smaller
        larger = [s for s in level_map.keys() if s >= font_size]
        smaller = [s for s in level_map.keys() if s < font_size]
        if larger:
            return level_map[min(larger, key=lambda s: s - font_size)]
        if smaller:
            return level_map[max(smaller, key=lambda s: font_size - s)]
        return 0

    def _normalize_numbered_line(self, text: str) -> str:
        """Normalize numbered list/heading markers like '1 . Foo' -> '1. Foo'."""
        return re.sub(r"^(\d+)\s*[\.)]\s*", r"\1. ", text)

    def post_process_markdown(self, markdown_content):
        """Post-process the markdown content."""
        try:
            markdown_content = re.sub(
                r"\n{3,}", "\n\n", markdown_content
            )  # Remove excessive newlines
            # Remove a single leading newline if present
            markdown_content = re.sub(r"^\n", "", markdown_content)
            markdown_content = re.sub(
                r" +", " ", markdown_content
            )  # Remove multiple spaces
            markdown_content = re.sub(
                r"\s*(---\n)+", "\n\n---\n", markdown_content
            )  # Remove duplicate horizontal lines
            return markdown_content
        except Exception as e:
            self.logger.error(f"Error post-processing markdown: {e}")
            self.logger.exception(traceback.format_exc())
            return markdown_content

    def save_markdown(self, markdown_content):
        """Save the markdown content to a file."""
        try:
            os.makedirs(Path(config["OUTPUT_DIR"]), exist_ok=True)
            with open(
                f"{Path(config['OUTPUT_DIR'])}/{self.pdf_filename}.md",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(markdown_content)
            self.logger.info("Markdown content saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving markdown content: {e}")
            self.logger.exception(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="Extract markdown-formatted content from a PDF file."
    )
    parser.add_argument("--pdf_path", help="Path to the input PDF file", required=True)
    args = parser.parse_args()

    extractor = MarkdownPDFExtractor(args.pdf_path)
    markdown_pages = extractor.extract()
    return markdown_pages


if __name__ == "__main__":
    main()