from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

def normalize_text(line: str) -> str:
    """Normalize spaced parentheses like '( 7 )' -> '(7)' and collapse spaces."""
    line = re.sub(r"\(\s+", "(", line)
    line = re.sub(r"\s+\)", ")", line)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line

def find_years_in_order(lines: List[str]) -> List[int]:
    """
    Find the column years in the order they appear in the header.
    We look for a line containing at least 2 years (often near '31 December').
    """
    for ln in lines[:50]:
        ln_norm = normalize_text(ln)
        years = [int(m.group(0)) for m in YEAR_RE.finditer(ln_norm)]
        ordered: List[int] = []
        for y in years:
            if y not in ordered:
                ordered.append(y)
        if len(ordered) >= 2:
            return ordered[:2]

    # Fallback: pick two most recent
    all_years: List[int] = []
    for ln in lines[:50]:
        all_years.extend(int(m.group(0)) for m in YEAR_RE.finditer(ln))
    uniq = sorted(set(all_years), reverse=True)
    return uniq[:2]

_AMOUNT_TOKEN_RE = re.compile(r"\(?\s*(?:\d{1,3}(?:,\d{3})+|\d+)\s*\)?")

def extract_amount_tokens(line: str, years: List[int]) -> List[str]:
    """
    Extract amount-looking tokens from a line (numbers with optional commas/parentheses),
    skipping header years like 2025/2024.
    """
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
    """
    Split line into (left_label, right_label) around amounts.
    Heuristic: left label is text before first amount token occurrence,
    right label is text after last amount token occurrence.
    """
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
    """
    Map Indonesian/English section labels to canonical section names.
    """
    l = label_left.lower().strip()
    if l == "aset" or l == "assets" or l.startswith("aset "):
        return "assets"
    if l == "liabilitas" or l == "liabilities" or l.startswith("liabilitas "):
        return "liabilities"
    if l == "ekuitas" or l == "equity" or l.startswith("ekuitas "):
        return "equity"
    if "dana syirkah temporer" in l:
        return "temporary_syirkah_funds"
    return None

@dataclass
class ItemNode:
    label: str
    label_right: str = ""
    values: Dict[str, Optional[int]] = field(default_factory=dict)
    children: List["ItemNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label}
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
    """
    Try to split 'Indonesian English' combined labels into (left, right).
    Heuristic: split at the first occurrence of common English starters.
    """
    t = normalize_text(text)
    for starter in _EN_STARTERS:
        idx = t.find(starter)
        if idx > 0:
            left = t[:idx].strip()
            right = t[idx:].strip()
            if left and right:
                return left, right
    return t, ""