"""Deterministic parsers for the cell formats found in shop files.

Every parser takes the cell text produced by `tools.cell_text` and returns a typed
value or None. Column-level helpers decide ambiguous separators from the whole
column, because one cell like "1.500" cannot tell a decimal point from a
thousands separator.
"""

import datetime as dt
import re
import unicodedata
from decimal import Decimal, InvalidOperation

EXCEL_EPOCH = dt.date(1899, 12, 30)
# Excel serial days for 1954..2119; outside this range a bare integer is not a date.
SERIAL_RANGE = (20000, 80000)

CURRENCY = re.compile(r"\s*(?:đ|₫|vnđ|vnd|đồng|dong)\.?\s*$", re.IGNORECASE)
TR_FORM = re.compile(r"^(\d+)\s*(?:tr|triệu|trieu)\s*(\d*)$", re.IGNORECASE)
TR_DECIMAL = re.compile(r"^(\d+)[,.](\d+)\s*(?:tr|triệu|trieu)$", re.IGNORECASE)
K_FORM = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|nghin|ngàn|ngan)$", re.IGNORECASE)


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def fold(text):
    """Accent- and case-insensitive key with collapsed whitespace."""
    return re.sub(r"\s+", " ", strip_accents(str(text)).lower()).strip()


def separator_style(values):
    """Return the decimal mark ('.' or ',') a numeric text column uses, or None.

    Evidence, in order: a value with both marks (the later one is decimal); a mark
    followed by other than exactly three digits (decimal); a mark repeated in one
    value (thousands).
    """
    votes = {".": 0, ",": 0}
    for value in values:
        if value is None:
            continue
        text = CURRENCY.sub("", str(value)).strip().lstrip("-+")
        if re.search(r"tr|k$", text, re.IGNORECASE):
            continue
        if "." in text and "," in text:
            votes[text[max(text.rfind("."), text.rfind(",")) :][0]] += 5
            continue
        for mark in ".,":
            parts = text.split(mark)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                if len(parts[1]) != 3 or parts[0] == "0":
                    votes[mark] += 1
            elif len(parts) > 2:
                other = "," if mark == "." else "."
                votes[other] += 1
    if votes["."] == votes[","]:
        return None
    return "." if votes["."] > votes[","] else ","


def parse_number(value, decimal=None):
    """Parse a number written as text: '560,000', '29.000đ', '45,000.0000',
    '129 000', '41k', '1tr090', '1,45tr', '-49.000đ', '(1,000)'. Returns Decimal.

    `decimal` is the column's decimal mark from `separator_style`; without it a
    lone mark followed by exactly three digits is read as a thousands separator.
    """
    if value is None:
        return None
    if isinstance(value, (int, Decimal)):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    text = str(value).strip().replace(" ", " ")
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    if text[:1] in "-−–":
        negative, text = True, text[1:].strip()
    elif text[:1] == "+":
        text = text[1:].strip()
    text = CURRENCY.sub("", text).strip()
    result = None
    m = TR_FORM.match(text)
    if m:
        frac = m.group(2)
        result = Decimal(m.group(1)) * 1_000_000
        if frac:
            result += Decimal(frac) / (10 ** len(frac)) * 1_000_000
    elif TR_DECIMAL.match(text):
        m = TR_DECIMAL.match(text)
        result = Decimal(f"{m.group(1)}.{m.group(2)}") * 1_000_000
    elif K_FORM.match(text):
        result = _plain(K_FORM.match(text).group(1).replace(",", "."), ".") * 1000
    else:
        result = _plain(text, decimal)
    if result is None:
        return None
    return -result if negative else result


def _plain(text, decimal):
    text = re.sub(r"(?<=\d)[  '](?=\d{3}\b)", "", text)
    if not re.fullmatch(r"\d[\d.,]*|[.,]\d+", text):
        return None
    if "." in text and "," in text:
        decimal = text[max(text.rfind("."), text.rfind(","))]
    if decimal is None:
        for mark in ".,":
            parts = text.split(mark)
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
                decimal = "," if mark == "." else "."
                break
        else:
            decimal = "." if "." in text else "," if "," in text else "."
    thousands = "," if decimal == "." else "."
    text = text.replace(thousands, "")
    if text.count(decimal) > 1:
        return None
    try:
        return Decimal(text.replace(decimal, "."))
    except InvalidOperation:
        return None


QTY = re.compile(r"^\s*([-+−]?\s*\d+(?:[.,]\d+)?)\s*(.*?)\s*$")


def parse_qty(value):
    """'2 hộp 30' -> (Decimal(2), 'hộp 30'); '3' -> (Decimal(3), None)."""
    if value is None:
        return None, None
    m = QTY.match(str(value))
    if not m:
        return None, None
    number = parse_number(m.group(1).replace(" ", ""), decimal=".")
    if number is None:
        number = parse_number(m.group(1).replace(" ", ""))
    unit = m.group(2) or None
    return number, unit


DATE_PATTERNS = [
    # ISO date or datetime (also what cell_text gives for real Excel dates).
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?"), "ymd"),
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$"), "dmy"),
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$"), "dmy2"),
    (re.compile(r"^(\d{1,2})\s*(?:thg|tháng|th)\s*(\d{1,2})[,\s]*(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$", re.IGNORECASE), "dmy"),
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})$"), "dm"),
]


def parse_datetime(value, year=None, month_first=False):
    """Parse a date cell into datetime (time 00:00 when absent), or None.

    Handles ISO dates/datetimes, d/m/yyyy, d-m-yy, '1 thg 6 2025', Excel serial
    days, and d/m without a year when `year` (the sheet's year, rulebook §8) is
    given. `month_first` swaps d/m for columns written m/d/yyyy.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{5}(?:\.\d+)?", text):
        serial = float(text)
        if SERIAL_RANGE[0] <= serial <= SERIAL_RANGE[1]:
            day = EXCEL_EPOCH + dt.timedelta(days=int(serial))
            seconds = round((serial - int(serial)) * 86400)
            return dt.datetime(day.year, day.month, day.day) + dt.timedelta(seconds=seconds)
        return None
    for pattern, kind in DATE_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        g = m.groups()
        try:
            if kind == "ymd":
                y, mo, d = int(g[0]), int(g[1]), int(g[2])
                rest = g[3:]
            elif kind == "dm":
                if year is None:
                    return None
                d, mo, y = int(g[0]), int(g[1]), int(year)
                rest = ()
            else:
                d, mo, y = int(g[0]), int(g[1]), int(g[2])
                if kind == "dmy2":
                    y += 2000
                rest = g[3:]
            if month_first and kind != "ymd":
                d, mo = mo, d
            hh, mm, ss = (int(x) if x else 0 for x in (list(rest) + [None] * 3)[:3])
            return dt.datetime(y, mo, d, hh, mm, ss)
        except ValueError:
            return None
    return None


def parse_date(value, year=None, month_first=False):
    parsed = parse_datetime(value, year, month_first)
    return parsed.date() if parsed else None


TIME_ONLY = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def parse_time(value):
    """Clock time from 'HH:MM[:SS]' or the time part of a datetime cell."""
    if value is None:
        return None
    text = str(value).strip()
    m = TIME_ONLY.match(text)
    if m:
        try:
            return dt.time(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
        except ValueError:
            return None
    parsed = parse_datetime(text)
    if parsed and (parsed.hour or parsed.minute or parsed.second or ":" in text):
        return parsed.time()
    if re.fullmatch(r"0?\.\d+", text):  # Excel time fraction
        seconds = round(float(text) * 86400)
        return (dt.datetime.min + dt.timedelta(seconds=seconds)).time()
    return None


def month_first_column(values):
    """True if a d/m/yyyy-looking column is really m/d/yyyy (some first part > 12
    and no second part > 12)."""
    first = second = False
    for value in values:
        m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}", str(value or ""))
        if m:
            first |= int(m.group(1)) > 12
            second |= int(m.group(2)) > 12
    return second and not first


def to_bool(value):
    if value is None:
        return None
    text = fold(value)
    if text in {"true", "1", "co", "x", "yes", "y", "dung"}:
        return True
    if text in {"false", "0", "khong", "no", "n", "sai"}:
        return False
    return None
