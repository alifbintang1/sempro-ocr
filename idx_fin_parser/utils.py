from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
AMOUNT_WORD_RE = re.compile(r'^\(?\d{1,3}(?:,\d{3})*\)?$|^\(?\d+\)?$')


def normalize_text(line: str) -> str:
    """Normalize spaced parentheses like '( 7 )' -> '(7)' and collapse spaces."""
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def find_years_in_order(lines: List[str]) -> List[int]:
    """Find the column years in the order they appear in the header."""
    for ln in lines[:50]:
        ln_norm = normalize_text(ln)
        years = [int(m.group(0)) for m in YEAR_RE.finditer(ln_norm)]
        ordered: List[int] = []
        for y in years:
            if y not in ordered:
                ordered.append(y)
        if len(ordered) >= 2:
            return ordered[:2]

    all_years: List[int] = []
    for ln in lines[:50]:
        all_years.extend(int(m.group(0)) for m in YEAR_RE.finditer(ln))
    uniq = sorted(set(all_years), reverse=True)
    return uniq[:2]


def is_amount_word(text: str, years_set: Optional[set] = None) -> bool:
    """Check if a single word token looks like a numeric amount (not a year)."""
    t = text.strip()
    if not AMOUNT_WORD_RE.match(t):
        return False
    inner = t.strip("()")
    try:
        v = int(inner.replace(",", ""))
        if years_set and len(inner.replace(",", "")) == 4 and v in years_set:
            return False
    except ValueError:
        pass
    return True


_AMOUNT_TOKEN_RE = re.compile(r"\(?\s*(?:\d{1,3}(?:,\d{3})+|\d+)\s*\)?")


def extract_amount_tokens(line: str, years: List[int]) -> List[str]:
    """Extract amount-looking tokens from a line, skipping years."""
    years_set = set(years)
    tokens: List[str] = []
    for m in _AMOUNT_TOKEN_RE.finditer(line):
        tok = normalize_text(m.group(0))
        if tok.isdigit() and len(tok) == 4 and int(tok) in years_set:
            continue
        tokens.append(tok)
    return tokens


def token_to_int(tok: str) -> Optional[int]:
    tok = normalize_text(tok)
    if not tok:
        return None
    neg = tok.startswith("(") and tok.endswith(")")
    tok_num = tok.strip("()").replace(",", "").strip()
    if not tok_num:
        return None
    try:
        val = int(tok_num)
    except ValueError:
        return None
    return -val if neg else val


def split_label_amounts(line: str, amount_tokens: List[str]) -> Tuple[str, str]:
    """Split line into (left_label, right_label) around amounts."""
    if not amount_tokens:
        return normalize_text(line), ""
    first = amount_tokens[0]
    last = amount_tokens[-1]
    i = line.find(first)
    j = line.rfind(last)
    left = line[:i].strip()
    right = line[j + len(last):].strip()
    return normalize_text(left), normalize_text(right)


def looks_like_section_header(label_left: str) -> Optional[str]:
    """Map section labels to canonical section names."""
    l = label_left.lower().strip()
    # Financial Position sections
    if l == "aset" or l == "assets" or l.startswith("aset "):
        return "assets"
    if l == "liabilitas" or l == "liabilities" or l.startswith("liabilitas "):
        return "liabilities"
    if l == "ekuitas" or l == "equity" or l.startswith("ekuitas "):
        return "equity"
    if l.startswith("dana syirkah temporer") or "liabilitas, dana syirkah" in l:
        return "temporary_syirkah_funds"
    # Profit & Loss sections
    if "pendapatan dan beban operasional" in l or "operating income and" in l:
        return "operating"
    if "penghasilan komprehensif lain" in l or "other comprehensive income" in l:
        return "other_comprehensive_income"
    if "pendapatan dan beban bukan operasional" in l or "non-operating income" in l:
        return "non_operating"
    if l.startswith("laba per saham") or l.startswith("laba (rugi) per saham") or l.startswith("earnings per share") or l.startswith("earnings (loss) per share"):
        return "earnings_per_share"
    return None


@dataclass
class ItemNode:
    label: str
    label_right: str = ""
    values: Dict[str, Optional[int]] = field(default_factory=dict)
    children: List["ItemNode"] = field(default_factory=list)
    level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"level": self.level, "label": self.label}
        if self.label_right:
            d["label_right"] = self.label_right
        if self.values:
            d["values"] = self.values
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


_EN_STARTERS = [
    "Current", "Placements", "Insurance", "Deferred", "Deposits", "Marketable",
    "Loans", "Investments", "Reinsurance", "Post-employment", "Equipment",
    "Guarantees", "Claims", "Difference", "Reserve", "Subordinated",
    "Obligations", "Securities", "Derivative", "Provisions", "Contract",
    "Other", "Total", "Cash", "Restricted", "Assets", "Liabilities", "Equity",
]


def split_bilingual_if_possible(text: str) -> Tuple[str, str]:
    """Try to split 'Indonesian English' combined labels into (left, right)."""
    t = normalize_text(text)
    for starter in _EN_STARTERS:
        idx = t.find(starter)
        if idx > 0:
            left = t[:idx].strip()
            right = t[idx:].strip()
            if left and right:
                return left, right
    return t, ""


# ── Column-based extraction ────────────────────────────────────────────────

def assign_levels(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign a 'level' field to each row based on x0_first (leftmost x of first word).

    x0 values are clustered into discrete levels with 8pt tolerance:
      smallest x0 cluster → level 0 (main items)
      next cluster        → level 1 (sub-items / pihak ketiga)
      etc.
    """
    # Use x0 from rows with amounts first (actual data rows, not headers/section labels).
    # Fall back to all rows with alphabetic id_labels if no amount rows exist.
    x0_vals = [r["x0_first"] for r in rows if r.get("amounts") and "x0_first" in r]
    if not x0_vals:
        x0_vals = [
            r["x0_first"] for r in rows
            if r.get("id_label") and r["id_label"][:1].isalpha() and "x0_first" in r
        ]
    if not x0_vals:
        return [{**r, "level": 0} for r in rows]

    # Build clusters: each cluster is represented by its first member
    sorted_unique = sorted(set(round(x) for x in x0_vals))
    clusters: List[float] = []
    for x in sorted_unique:
        if clusters and abs(x - clusters[-1]) <= 8:
            pass  # same cluster
        else:
            clusters.append(float(x))

    def x_to_level(x: float) -> int:
        rounded = round(x)
        best_lvl, best_dist = 0, float("inf")
        for lvl, rep in enumerate(clusters):
            d = abs(rounded - rep)
            if d < best_dist:
                best_dist, best_lvl = d, lvl
        return best_lvl

    return [{**r, "level": x_to_level(r.get("x0_first", 0.0))} for r in rows]


def _merge_paren_amounts(words: List[Dict]) -> List[Dict]:
    """Merge split parenthesized amounts: ['(', '7', ')'] → [{'text': '(7)', ...}]."""
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        if (w["text"] == "("
                and i + 2 < len(words)
                and re.match(r"^[\d,]+$", words[i + 1]["text"])
                and words[i + 2]["text"] == ")"):
            merged = {**w, "x1": words[i + 2]["x1"], "text": f"({words[i + 1]['text']})"}
            result.append(merged)
            i += 3
        else:
            result.append(w)
            i += 1
    return result


_CONTINUATION_TAIL_WORDS = {
    "bank", "dan", "pada", "dengan", "atas", "atau", "di", "ke", "dari",
    "the", "and", "of", "in", "at", "on", "for",
}


def merge_continuation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge visual continuation lines into their preceding row.

    A continuation row has NO amounts and satisfies ONE of:
      1. id_label is empty (only English text wrapped to next line)
      2. id_label starts with a lowercase letter (Indonesian label suffix)
      3. previous id_label ends with a continuation word (e.g. "Bank") and
         the current row has the same x0 (within 8pt) — fixes mid-word splits
         like "Penempatan pada Bank" + "Indonesia dan bank lain"
    """
    if not rows:
        return rows

    result: List[Dict[str, Any]] = []
    for row in rows:
        id_label = row.get("id_label", "")
        en_label = row.get("en_label", "")
        amounts = row.get("amounts", [])
        x0 = row.get("x0_first", 0.0)

        if not amounts and result:
            prev = result[-1]
            prev_id = prev.get("id_label", "")
            prev_tail = prev_id.split()[-1].lower() if prev_id.split() else ""
            prev_x0 = prev.get("x0_first", 0.0)

            is_continuation = (
                not id_label                             # 1. empty id (only EN text wrapped)
                or (id_label and id_label[0].islower())  # 2. lowercase label suffix
                or (                                     # 3. prev ends mid-phrase, same indent
                    prev_tail in _CONTINUATION_TAIL_WORDS
                    and abs(x0 - prev_x0) < 8
                )
            )

            if is_continuation:
                if id_label:
                    prev["id_label"] = (prev["id_label"] + " " + id_label).strip()
                if en_label:
                    prev["en_label"] = (prev["en_label"] + " " + en_label).strip()
                continue

        result.append(dict(row))

    return result


def extract_rows_by_column(page: Any, years_set: set) -> List[Dict[str, Any]]:
    """
    Extract structured rows from a pdfplumber page using word x-coordinates.

    Splits each visual row into:
      - id_label: Indonesian text (left of amounts)
      - amounts:  numeric tokens (middle)
      - en_label: English text (right of amounts)

    Returns list of {'id_label': str, 'amounts': List[str], 'en_label': str}.
    """
    words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    if not words:
        return []

    # Group words into visual rows by y-coordinate (3pt tolerance)
    row_map: Dict[int, List[Dict]] = {}
    for word in words:
        y_key = round(word["top"] / 3) * 3
        row_map.setdefault(y_key, []).append(word)

    # First pass: detect where the English column starts (x threshold)
    # Use rows that have amounts: English text starts right after the last amount.
    en_x_starts: List[float] = []
    for y_key in sorted(row_map.keys()):
        row_words = _merge_paren_amounts(sorted(row_map[y_key], key=lambda w: w["x0"]))
        amt_indices = [i for i, w in enumerate(row_words) if is_amount_word(w["text"], years_set)]
        if amt_indices:
            last_amt = amt_indices[-1]
            if last_amt + 1 < len(row_words):
                en_x_starts.append(row_words[last_amt + 1]["x0"])

    en_x_threshold = min(en_x_starts) if en_x_starts else page.width * 0.65

    # Second pass: build structured rows
    result: List[Dict[str, Any]] = []
    for y_key in sorted(row_map.keys()):
        row_words = _merge_paren_amounts(sorted(row_map[y_key], key=lambda w: w["x0"]))
        amt_indices = [i for i, w in enumerate(row_words) if is_amount_word(w["text"], years_set)]
        x0_first = row_words[0]["x0"] if row_words else 0.0

        if not amt_indices:
            # No amounts — split into ID / EN columns by x threshold
            id_words = [w for w in row_words if w["x0"] < en_x_threshold]
            en_words = [w for w in row_words if w["x0"] >= en_x_threshold]
            id_text = normalize_text(" ".join(w["text"] for w in id_words))
            en_text = normalize_text(" ".join(w["text"] for w in en_words))
            if id_text or en_text:
                result.append({"id_label": id_text, "amounts": [], "en_label": en_text, "x0_first": x0_first})
            continue

        first_amt = amt_indices[0]
        last_amt = amt_indices[-1]

        id_label = normalize_text(" ".join(row_words[i]["text"] for i in range(first_amt)))
        amounts = [row_words[i]["text"] for i in amt_indices]
        en_label = normalize_text(
            " ".join(row_words[i]["text"] for i in range(last_amt + 1, len(row_words)))
        )

        result.append({"id_label": id_label, "amounts": amounts, "en_label": en_label, "x0_first": x0_first})

    return merge_continuation_rows(result)
