"""Vietnamese time expressions -> date ranges, following rulebook §2.

All ranges are inclusive [start, end] dates. `start=None` means "from the
beginning of the data" (stock / debt questions "đến hết ngày X").
"""

import csv
import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache

from .values import fold

TELEX_PAIRS = [("dd", "d"), ("aw", "a"), ("aa", "a"), ("ee", "e"), ("oo", "o"), ("ow", "o"), ("uw", "u")]


SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?"
    r"(?:a|ai|ao|au|ay|e|eo|i|ia|iu|ie|ieu|o|oa|oai|oay|oe|oi|oo|u|ua|uan|ue|ui|uo|uoi|uou|uy|uya|uye|uyu|y|ye|yeu|eu|ua|uu|ou|oao)"
    r"(?:ng|nh|ch|c|m|n|p|t|i|o|u|y)?$"
)


def untelex(word):
    """Undo common Telex typing slips in a folded word: 'thangs' -> 'thang',
    'dduwowcj' -> 'duoc', 'nawm' -> 'nam'. Vietnamese syllables never end in
    s/f/r/x/j, so a trailing one is a stray tone key. Only applied when the
    result is a Vietnamese syllable and the word is not, so 'order' or 'clear'
    stay unchanged."""
    if not word.isalpha() or len(word) < 3 or SYLLABLE.match(word):
        return word
    fixed = word
    if fixed[-1] in "sfrxj":
        fixed = fixed[:-1]
    for a, b in TELEX_PAIRS:
        fixed = fixed.replace(a, b)
    return fixed if SYLLABLE.match(fixed) else word


def normalize_question(text):
    """Folded, typo-tolerant form used for intent and period matching."""
    text = fold(text)
    text = re.sub(r"[?!.,;:\"“”()]+(?=\s|$)", " ", text)
    text = re.sub(r"[a-z]+", lambda m: untelex(m.group()), text)
    replacements = [
        (r"\bdthu\b|\bdoanh so\b", "doanh thu"),
        (r"\bhnay\b|\bbua nay\b|\bbua ni\b|\bhom ni\b|\bhom nay\b", "hom nay"),
        (r"\bhqua\b|\bbua qua\b|\bhom qua\b|\bbua hom qua\b", "hom qua"),
        (r"\bthang ni\b", "thang nay"),
        (r"\btuan ni\b", "tuan nay"),
        (r"\bnam ni\b", "nam nay"),
        (r"\btrc\b", "truoc"),
        (r"\bvs\b", "voi"),
        (r"\bdc\b|\bdk\b", "duoc"),
        (r"\bbn\b|\bbnhiu\b|\bbnhieu\b|\bbao nhiu\b|\bnhiu\b", "bao nhieu"),
        (r"\bsl\b", "so luong"),
        (r"\bsp\b", "san pham"),
        (r"\bdh\b", "don hang"),
        (r"\bko\b|\bkh\b", "khong"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Period:
    start: dt.date | None
    end: dt.date
    kind: str = "range"
    label: str = ""

    @property
    def days(self):
        return None if self.start is None else (self.end - self.start).days + 1

    def contains(self, day):
        return (self.start is None or self.start <= day) and day <= self.end


def month_end(year, month):
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return nxt - dt.timedelta(days=1)


def month(year, mon):
    return Period(dt.date(year, mon, 1), month_end(year, mon), "month", f"{mon:02d}/{year}")


def quarter(year, q):
    first = 3 * (q - 1) + 1
    return Period(dt.date(year, first, 1), month_end(year, first + 2), "quarter", f"Q{q}/{year}")


def year_period(year):
    return Period(dt.date(year, 1, 1), dt.date(year, 12, 31), "year", str(year))


def day(d):
    return Period(d, d, "day", d.isoformat())


def week_of(d):
    monday = d - dt.timedelta(days=d.weekday())
    return Period(monday, monday + dt.timedelta(days=6), "week", f"week of {monday}")


def this_week(today):
    monday = today - dt.timedelta(days=today.weekday())
    return Period(monday, today, "this_week", "tuần này")


def last_week(today):
    monday = today - dt.timedelta(days=today.weekday() + 7)
    return Period(monday, monday + dt.timedelta(days=6), "week", "tuần trước")


def this_month(today):
    return Period(today.replace(day=1), today, "this_month", "tháng này")


def previous_month_same_days(today):
    """'Cùng kỳ tháng trước' of 'tháng này' (§2): day 1 to the same day number of
    the previous month, or to its end if it has no such day."""
    first_prev = (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
    last = month_end(first_prev.year, first_prev.month)
    end = first_prev.replace(day=min(today.day, last.day))
    return Period(first_prev, end, "range", "cùng kỳ tháng trước")


def last_n_days(n, today):
    return Period(today - dt.timedelta(days=n - 1), today, "range", f"{n} ngày gần nhất")


def year_to_date(today):
    return Period(dt.date(today.year, 1, 1), today, "range", "từ đầu năm")


def resolve_month(mon, today):
    """'Tháng M' without a year (§2): this year unless that is after today."""
    year = today.year if dt.date(today.year, mon, 1) <= today else today.year - 1
    return month(year, mon)


def resolve_quarter(q, today):
    year = today.year if dt.date(today.year, 3 * (q - 1) + 1, 1) <= today else today.year - 1
    return quarter(year, q)


def previous_period(p, today=None):
    """The period right before `p`, as used for 'so với kỳ trước'."""
    if p.kind == "month":
        prev = p.start - dt.timedelta(days=1)
        return month(prev.year, prev.month)
    if p.kind == "quarter":
        prev = p.start - dt.timedelta(days=1)
        return quarter(prev.year, (prev.month - 1) // 3 + 1)
    if p.kind == "year":
        return year_period(p.start.year - 1)
    if p.kind == "this_month":
        return previous_month_same_days(p.end)
    if p.kind in ("week", "this_week"):
        start = p.start - dt.timedelta(days=7)
        return Period(start, start + dt.timedelta(days=p.days - 1), "range", "tuần trước đó")
    if p.kind == "tet":
        return Period(p.start - dt.timedelta(days=p.days), p.start - dt.timedelta(days=1), "range", "trước Tết")
    n = p.days
    return Period(p.start - dt.timedelta(days=n), p.start - dt.timedelta(days=1), "range", "kỳ trước")


def same_period_last_year(p):
    def shift(d):
        try:
            return d.replace(year=d.year - 1)
        except ValueError:  # 29/02
            return d.replace(year=d.year - 1, day=28)

    if p.kind == "month":
        return month(p.start.year - 1, p.start.month)
    if p.kind == "quarter":
        return quarter(p.start.year - 1, (p.start.month - 1) // 3 + 1)
    return Period(shift(p.start), shift(p.end), p.kind if p.kind != "this_month" else "range", "cùng kỳ năm trước")


@lru_cache(maxsize=8)
def lunar_table(calendar_path):
    rows = {}
    with open(calendar_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (int(r["lunar_year"]), int(r["lunar_month"]), int(r["lunar_day"]), r["lunar_leap"].lower() == "true")
            rows.setdefault(key, dt.date.fromisoformat(r["date"]))
    return rows


def tet(year, calendar_path):
    """Tết năm Y (§2): solar date of 23 tháng Chạp of lunar year Y-1 through 10
    tháng Giêng of lunar year Y."""
    table = lunar_table(str(calendar_path))
    start = table.get((year - 1, 12, 23, False))
    end = table.get((year, 1, 10, False))
    if not start or not end:
        return None
    return Period(start, end, "tet", f"Tết {year}")


DATE_RE = r"(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?"
MONTH_YEAR_RE = r"(?:thang|th)\s*(\d{1,2})\s*(?:/|-|nam)\s*(\d{4}|\d{2})\b"


def _year(text):
    y = int(text)
    return y + 2000 if y < 100 else y


def _date(d, m, y, today):
    if y:
        return dt.date(_year(y), int(m), int(d))
    candidate = dt.date(today.year, int(m), int(d))
    return candidate if candidate <= today else candidate.replace(year=today.year - 1)


@dataclass
class Mention:
    period: Period
    span: tuple
    text: str


def find_periods(q, today, calendar_path=None):
    """All period mentions in a normalized question, in text order.

    Overlapping matches keep the earliest, longest one. Returns [Mention].
    """
    found = []

    def add(m, period, group=0):
        if period is not None:
            found.append(Mention(period, m.span(group), m.group(group)))

    for m in re.finditer(r"tu dau thang nay(?: (?:den|toi) (?:het )?(?:ngay )?hom nay)?", q):
        add(m, this_month(today))
    for m in re.finditer(r"cung ky thang truoc", q):
        add(m, previous_month_same_days(today))
    for m in re.finditer(r"tu (?:ngay )?(?:dau nam|01?/01?)(?: nay)?(?: (?:den|toi) (?:het )?(?:ngay )?hom nay)?", q):
        add(m, year_to_date(today))
    for m in re.finditer(r"\btu (?:ngay )?" + DATE_RE + r" (?:den|toi) (?:het )?(?:ngay )?" + DATE_RE, q):
        g = m.groups()
        try:
            add(m, Period(_date(g[0], g[1], g[2] or g[5], today), _date(g[3], g[4], g[5], today), "range", m.group(0)))
        except ValueError:
            pass
    for m in re.finditer(r"\btu " + MONTH_YEAR_RE + r" (?:den|toi) (?:het )?" + MONTH_YEAR_RE, q):
        g = m.groups()
        a, b = month(_year(g[1]), int(g[0])), month(_year(g[3]), int(g[2]))
        add(m, Period(a.start, b.end, "month_range", m.group(0)))
    for m in re.finditer(r"(?:den het|tinh den(?: het)?|toi het|cho den het)(?: ngay)? (?:" + DATE_RE + r"|(hom nay))", q):
        g = m.groups()
        try:
            end = today if g[3] else _date(g[0], g[1], g[2], today)
        except ValueError:
            continue
        add(m, Period(None, end, "asof", m.group(0)))
    for m in re.finditer(r"(\d+) ngay (?:gan nhat|gan day|qua|vua qua|cuoi)", q):
        add(m, last_n_days(int(m.group(1)), today))
    for m in re.finditer(r"(\d+) ngay ke tu (?:ngay )?" + DATE_RE, q):
        try:
            start = _date(m.group(2), m.group(3), m.group(4), today)
        except ValueError:
            continue
        n = int(m.group(1))
        add(m, Period(start, start + dt.timedelta(days=n - 1), "range", m.group(0)))
    for m in re.finditer(MONTH_YEAR_RE, q):
        try:
            add(m, month(_year(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(r"\bth?\s*(\d{1,2})/(\d{2}|\d{4})\b", q):
        try:
            add(m, month(_year(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in re.finditer(r"\bquy (\d)\s*(?:/|-|nam)\s*(\d{4}|\d{2})\b", q):
        if 1 <= int(m.group(1)) <= 4:
            add(m, quarter(_year(m.group(2)), int(m.group(1))))
    for m in re.finditer(r"\bquy (\d)\b", q):
        if 1 <= int(m.group(1)) <= 4:
            add(m, resolve_quarter(int(m.group(1)), today))
    for m in re.finditer(r"\bthang (\d{1,2})\b(?!\s*(?:/|-|nam)\s*\d)", q):
        if 1 <= int(m.group(1)) <= 12:
            add(m, resolve_month(int(m.group(1)), today))
    for m in re.finditer(r"(?:ngay )?\b" + DATE_RE + r"\b", q):
        try:
            add(m, day(_date(m.group(1), m.group(2), m.group(3), today)))
        except ValueError:
            pass
    for m in re.finditer(r"\bthang nay\b", q):
        add(m, this_month(today))
    for m in re.finditer(r"\bthang (?:truoc|roi)(?: do)?\b", q):
        prev = today.replace(day=1) - dt.timedelta(days=1)
        add(m, month(prev.year, prev.month))
    for m in re.finditer(r"\btuan nay\b", q):
        add(m, this_week(today))
    for m in re.finditer(r"\btuan (?:truoc|roi)\b", q):
        add(m, last_week(today))
    for m in re.finditer(r"\bhom nay\b", q):
        add(m, day(today))
    for m in re.finditer(r"\bhom qua\b", q):
        add(m, day(today - dt.timedelta(days=1)))
    for m in re.finditer(r"\bnam nay\b", q):
        add(m, year_to_date(today))
    for m in re.finditer(r"\bnam (?:ngoai|truoc|roi)\b", q):
        add(m, year_period(today.year - 1))
    for m in re.finditer(r"\bnam (\d{4})\b", q):
        add(m, year_period(int(m.group(1))))
    if calendar_path:
        for m in re.finditer(r"\btet(?: am lich| nguyen dan)?(?: nam)? (\d{4}|nay|ngoai|truoc)\b|\b(?:dip|mua) tet\b", q):
            word = m.group(1)
            if word is None or word == "nay":
                y = today.year
            elif word in ("ngoai", "truoc"):
                y = today.year - 1
            else:
                y = int(word)
            add(m, tet(y, calendar_path))
    # Keep earliest-starting, then longest, non-overlapping mentions.
    found.sort(key=lambda x: (x.span[0], -(x.span[1] - x.span[0])))
    result, last_end = [], -1
    for mention in found:
        if mention.span[0] >= last_end:
            result.append(mention)
            last_end = mention.span[1]
    return result

