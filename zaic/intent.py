"""Rule-based question parser for the common question families.

`parse(question, shop, calendar_path)` returns an Intent (metric name, periods,
filters) or None when the question is outside the families it knows; the
caller then falls back to the LLM. Entities (items, groups, channels) are
resolved against the shop's own values, never invented.
"""

import datetime as dt
import re
from dataclasses import dataclass, field
from decimal import Decimal

from .lexicon import FAMILY_WORDS, KNOWN_WORDS
from .metrics import NoData, Unsupported, number, pct, person_name
from .periods import (
    Period,
    accented,
    find_periods,
    normalize_folded,
    previous_period,
    same_period_last_year,
)
from .values import fold, strip_accents

RELATIVE_BASE = re.compile(
    r"\b(?:thang|tuan|quy|nam|ky) (?:truoc|roi|lien truoc|ngay truoc)(?: do)?\b|\bcung ky\b|\btruoc tet\b"
)
PAYMENT_PHRASES = [
    ("cash", r"tien mat|tien tuoi|\btm\b"),
    ("transfer", r"chuyen khoan|\bck\b"),
    ("cod", r"\bcod\b|giao hang thu tien"),
    ("ewallet", r"vi dien tu|\bvi\b"),
    ("debt", r"ghi no|\bno\b"),
]


CHANNEL_SYNONYMS = [
    (r"store|quay|cua hang|tiem|truc tiep|offline|tai cho", r"tai quay|tai cua hang|tai tiem|truc tiep|den mua|tai cho"),
    (r"phone|dien thoai|hotline", r"dien thoai|goi dien|hotline"),
    (r"zalo", r"zalo"),
    (r"facebook|\bfb\b|messenger", r"facebook|\bfb\b|messenger"),
    (r"shopee|lazada|tiki|tiktok|\bsan\b|tmdt|marketplace|ecommerce", r"\bsan\b|tmdt|shopee|lazada|tiktok"),
    (r"grab|baemin|gojek|befood|shopeefood|\bapp\b", r"grab|baemin|gojek|befood|\bapp\b"),
    (r"web|online", r"website|\bweb\b|online"),
]


# Masked entity tokens: the name found in the question is replaced by its token
# before typo repair and intent matching, so words inside a product name
# ("kho quẹt", "lọ 13ml", "19L") never look like question vocabulary.
TOKENS = {
    "item": "xitem",
    "group": "xgroup",
    "ingredient": "xingredient",
    "channel": "xchannel",
    "customer_group": "xcgroup",
    "staff": "xstaff",
    "customer": "xcustomer",
}
# Metrics whose question legitimately carries free numbers (top k, thresholds,
# month counts, day windows); elsewhere a number nobody consumed is a
# condition the parser did not understand.
NUMERIC_METRICS = {
    "top_items",
    "top_customers",
    "count_days",
    "items_in_orders",
    "customers_with_orders",
    "customers_share",
    "count_orders_by_value",
    "best_month",
    "increasing",
    "month_values",
}
# Attributes a question can filter on, and how the question names them. When the
# shop records no such attribute at all, the question is unanswerable (§7).
ATTRIBUTE_PHRASES = [
    ("channel", r"\bkenh\b|\b(?:qua|tren) (?:zalo|dien thoai|hotline|facebook|fb|messenger|shopee|lazada|tiktok|grab|app|web|website|san)\b|\btai quay\b|\bonline\b"),
    ("payment_method", r"\btra bang\b|\bthanh toan bang\b|\bhinh thuc thanh toan\b|\bcod\b|\bchuyen khoan\b|\bvi dien tu\b|\btien mat\b|\bghi no\b"),
    ("customer_group", r"\bkhach (?:nhom|si|le|than thiet|vip)\b|\bnhom khach\b"),
    ("staff", r"\bnhan vien\b|\bnguoi ban\b"),
    ("gender", r"\bkhach (?:hang )?(?:nu|nam)\b(?! (?:nay|ngoai|truoc|\d))|\bgioi tinh\b|\b(?:nam|nu) gioi\b|\bphu nu\b|\bdan ong\b"),
    ("address", r"\bo (?:duong|quan|phuong|huyen|khu)\b|\bdia chi\b"),
    ("cancel_reason", r"\bhuy (?:vi|do)\b|\bly do huy\b|\btra (?:hang )?(?:vi|do)\b|\bly do tra\b"),
]


@dataclass
class Intent:
    metric: str
    period: Period
    base: Period | None = None
    filters: dict = field(default_factory=dict)
    k: int | None = None
    measure: str = "revenue"
    values: list = field(default_factory=list)  # for list answers: [(filter key, value)]
    numeric: bool = False  # the question's free numbers were consumed


class Matcher:
    """Finds any of many folded names in a text, longest name first."""

    def __init__(self, mapping):
        self.mapping = mapping
        keys = sorted((k for k in mapping if len(k) >= 2), key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, keys)) + r")(?![\w])") if keys else None

    def find(self, text):
        if self.rx is None:
            return []
        return [(m.start(), m.end(), self.mapping[m.group()]) for m in self.rx.finditer(text)]


def entity_index(shop):
    """Folded value -> original value for each entity kind the shop records."""
    index = {"item": {}, "group": {}, "customer_group": {}, "channel": {}, "ingredient": {}, "staff": {}, "unit": {}}
    for r in shop.lines or []:
        if r.get("kind") != "item":
            continue
        if r.get("item_name"):
            index["item"].setdefault(fold(r["item_name"]), r["item_name"])
        if r.get("group"):
            index["group"].setdefault(fold(r["group"]), r["group"])
        if r.get("ingredient"):
            index["ingredient"].setdefault(fold(r["ingredient"]), r["ingredient"])
        if r.get("base_unit"):
            index["unit"].setdefault(fold(r["base_unit"]), r["base_unit"])
    for o in shop.orders or []:
        for k in ("customer_group", "channel", "staff"):
            if o.get(k):
                index[k].setdefault(fold(o[k]).replace("_", " "), o[k])
    index["customer"] = {}
    for o in shop.orders or []:
        name = o.get("customer_name")
        if name and len(person_name(name).split()) >= 2:
            index["customer"].setdefault(person_name(name), name)
    index["_matchers"] = {k: Matcher(v) for k, v in index.items()}
    # Every accented spelling of a name, to tell "trả" from the item "Trà".
    spellings = {}
    for kind, mapping in index.items():
        if kind != "_matchers":
            for folded in mapping:
                spellings.setdefault(folded, set())
    for rows, fields in ((shop.lines or [], ("item_name", "group", "ingredient", "base_unit")), (shop.orders or [], ("customer_group", "customer_name", "staff"))):
        for r in rows:
            for f in fields:
                v = r.get(f)
                if v:
                    text = accented(v)
                    for name in {fold(text), person_name(v) if f == "customer_name" else fold(text)}:
                        if name in spellings:
                            spellings[name].add(text[: len(name)] if f == "customer_name" else text)
    index["_spellings"] = spellings
    return index


def find_all(text, mapping):
    """All (start, end, value) of mapping keys in text, longest match wins per span."""
    matcher = mapping if isinstance(mapping, Matcher) else Matcher(mapping)
    return matcher.find(text)


def channels_in(q, channel_index):
    """(start, end, raw channel value) for channel phrases in q. Raw values may be
    codes ('store', 'zalo_oa'), so known synonyms are matched too."""
    hits = find_all(q, channel_index)
    mapping = channel_index.mapping if isinstance(channel_index, Matcher) else channel_index
    for value_key, raw in mapping.items():
        for value_rx, question_rx in CHANNEL_SYNONYMS:
            if re.search(value_rx, value_key):
                for m in re.finditer(question_rx, q):
                    hits.append((m.start(), m.end(), raw))
                break
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    seen, out = set(), []
    for start, end, raw in hits:
        if raw not in seen:
            seen.add(raw)
            out.append((start, end, raw))
    return out


def payments_in(q):
    hits = []
    for kind, rx in PAYMENT_PHRASES:
        for m in re.finditer(rx, q):
            hits.append((m.start(), kind))
    hits.sort()
    seen, out = set(), []
    for pos, kind in hits:
        if kind not in seen:
            seen.add(kind)
            out.append((pos, kind))
    return out


MONEY_IN_Q = r"(\d[\d.,]*)\s*(trieu|tr|k|nghin|ngan|dong|d|vnd)?\b"


def threshold(q):
    """('>=', Decimal) from phrases like 'từ 55.000 đồng trở lên', 'ít nhất 4 đơn',
    'trên 10 triệu', 'dưới 3'. None when absent."""
    from .values import parse_number

    pats = [
        (r"(?:tu|dat|it nhat|toi thieu|>=|≥)\s*" + MONEY_IN_Q + r"(?:[^\d]{0,25}tro len)?", ">="),
        (r"" + MONEY_IN_Q + r"[^\d]{0,25}tro len", ">="),
        (r"(?:tren|hon|qua|>)\s*" + MONEY_IN_Q, ">"),
        (r"(?:duoi|it hon|<)\s*" + MONEY_IN_Q, "<"),
        (r"(?:toi da|khong qua|<=|≤)\s*" + MONEY_IN_Q, "<="),
    ]
    for rx, op in pats:
        m = re.search(rx, q)
        if not m:
            continue
        number_text, unit = m.group(1), m.group(2)
        value = parse_number(number_text.rstrip(".,"))
        if value is None:
            continue
        if unit in ("trieu", "tr"):
            value *= 1_000_000
        elif unit in ("k", "nghin", "ngan"):
            value *= 1000
        return op, value
    return None


def month_list(q, mentions, today):
    """Months a multi-month question names: '3 tháng liền nhau kết thúc ở tháng X',
    'từ tháng A đến tháng B', or explicit 'tháng A, B và C'."""
    from .periods import month as month_period

    m = re.search(r"(\d+) thang (?:lien nhau|lien tiep|gan nhat)?.*?ket thuc (?:o|vao|tai)? ?thang (\d{1,2})\s*(?:/|nam)\s*(\d{4})", q)
    if m:
        n, mon, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out = []
        for i in range(n - 1, -1, -1):
            y, mo = year, mon - i
            while mo < 1:
                mo += 12
                y -= 1
            out.append(month_period(y, mo))
        return out
    ranges = [x.period for x in mentions if x.period.kind == "month_range"]
    if ranges:
        r = ranges[0]
        out, d = [], r.start
        while d <= r.end:
            out.append(month_period(d.year, d.month))
            d = (out[-1].end + dt.timedelta(days=1))
        return out
    months = [x.period for x in mentions if x.period.kind == "month"]
    found = re.findall(r"(?<![\d/])(\d{1,2})/(\d{4})", q)
    if len(found) >= 2:
        return [month_period(int(y), int(mo)) for mo, y in found]
    return months if len(months) >= 2 else None


def same_word(written, folded, index):
    """A name typed with accents must carry the accents of a spelling the shop
    uses ("trả" is not the item "Trà"); a name typed without accents matches
    by its folded form."""
    if written == folded:
        return True
    spellings = (index.get("_spellings") or {}).get(folded)
    if not spellings:
        return True
    return written in spellings or all(s == strip_accents(s) for s in spellings)


def entity_hits(plain, q0, index, written=None):
    """Entity mentions in the folded question as non-overlapping
    (start, end, kind, value), longest first. Kinds other than items are only
    looked for when the question has the word that introduces them."""
    matchers = index.get("_matchers") or {k: Matcher(v) for k, v in index.items() if isinstance(v, dict)}
    kinds = ["item"]
    if re.search(r"\bnhom\b", q0):
        kinds.append("group")
    if re.search(r"hoat chat", q0):
        kinds.append("ingredient")
    if re.search(r"\bkhach\b", q0):
        kinds.append("customer_group")
    if re.search(r"\bnhan vien\b|\bnguoi ban\b|\bnv\b", q0):
        kinds.append("staff")
    if re.search(r"\bkhach\b|\bno\b", q0):
        kinds.append("customer")
    hits = []
    for kind in kinds:
        if kind in matchers:
            for s, e, v in matchers[kind].find(plain):
                if written is None or same_word(written[s:e], plain[s:e], index):
                    hits.append((s, e, kind, v))
    if re.search(r"\bkenh\b|\bqua\b|\btai\b|\bban\b", q0) and "channel" in matchers:
        hits += [(s, e, "channel", v) for s, e, v in channels_in(plain, matchers["channel"])]

    def rank(h):
        s, e, kind, _ = h
        after_nhom = plain[max(0, s - 10) : s].rstrip().endswith(("nhom", "nhom hang", "hoat chat"))
        order = {"ingredient": 1, "item": 2, "group": 3, "customer_group": 4, "staff": 5, "channel": 6, "customer": 7}[kind]
        if kind in ("group", "ingredient") and after_nhom:
            order = 0
        return (-(e - s), order, s)

    kept = []
    for h in sorted(hits, key=rank):
        if all(h[1] <= k[0] or h[0] >= k[1] for k in kept):
            kept.append(h)
    return sorted(kept)


def mask(plain, hits):
    out, last = [], 0
    for s, e, kind, _ in hits:
        out.append(plain[last:s])
        out.append(f" {TOKENS[kind]} ")
        last = e
    out.append(plain[last:])
    return "".join(out)


def leftover(q, mentions, intent):
    """Words of the question no rule accounts for. A known-vocabulary word is
    fine; anything else (an unmatched product, a condition, a number nobody
    consumed) means the parser did not understand the whole question."""
    text, last = [], 0
    for m in mentions:
        text.append(q[last : m.span[0]])
        last = m.span[1]
    text.append(q[last:])
    rest = re.sub(r"\d+ ngay (?:ngay )?truoc\b", " ", " ".join(text))
    unknown = []
    for token in re.findall(r"[a-z]+\d*|\d[\d.,/]*", rest):
        if token in TOKENS.values() or token in KNOWN_WORDS or token in FAMILY_WORDS.get(intent.metric, ()):
            continue
        if token[0].isdigit():
            if intent.numeric or intent.metric in NUMERIC_METRICS:
                continue
            unknown.append(token)
            continue
        unknown.append(token)
    return unknown


def parse(question, shop, calendar_path, today=None, index=None, explain=None):
    """Intent for a question, or None when it is outside the known families or
    has words the rules do not account for. `explain` (a dict) receives the
    candidate intent and the leftover words for diagnostics."""
    written = accented(question["question"])
    plain = strip_accents(written)  # entity names: no typo repair
    if len(plain) != len(written):
        written = None
    q0 = normalize_folded(plain)
    today = today or dt.date.fromisoformat(question["as_of_date"])
    index = index or entity_index(shop)
    hits = entity_hits(plain, q0, index, written)
    q = normalize_folded(mask(plain, hits))
    mentions = find_periods(q, today, calendar_path)
    if explain is not None:
        explain.update({"masked": q, "reason": "no period" if not mentions else "no family"})
    if not mentions:
        return None
    intent = classify(q, hits, mentions, today, shop, index)
    if intent is None:
        return None
    extra = [] if intent.metric == "unanswerable" else leftover(q, mentions, intent)
    if explain is not None:
        explain.update({"intent": intent, "leftover": extra, "masked": q, "reason": "leftover" if extra else None})
    return None if extra else intent


# Header words that may hold an attribute even when no role was assigned; if
# any header of the shop looks like one, "not recorded" cannot be claimed.
ATTRIBUTE_HEADERS = {
    "channel": r"kenh|nguon|ban qua",
    "payment_method": r"thanh toan|pttt|httt|hinh thuc|phuong thuc",
    "customer_group": r"nhom kh|^hang$|^hang (?:kh|thanh vien)|loai kh|phan loai kh|doi tuong kh|kieu khach",
    "staff": r"nhan vien|\bnv\b|nguoi ban|thu ngan|ma nv",
    "gender": r"gioi tinh|nam ?nu|\bgioi\b|\bphai\b",
    "address": r"dia chi|duong|quan|phuong|khu vuc",
    "cancel_reason": r"ly do|nguyen nhan|ghi chu",
}


def missing_attribute(q, shop):
    """Name of an attribute the question filters on that the shop never
    records (§7), "?" when a column might hold it but was not understood."""
    for name, rx in ATTRIBUTE_PHRASES:
        if re.search(rx, q) and not shop_has(shop, name):
            return name if not may_have(shop, name) else "?"
    return None


def shop_has(shop, attribute):
    cache = shop.__dict__.setdefault("_has", {})
    if attribute not in cache:
        cache[attribute] = any(o.get(attribute) for o in shop.orders or []) or any(
            r.get(attribute) for r in (shop.lines or [])[:50000]
        )
    return cache[attribute]


def may_have(shop, attribute):
    cache = shop.__dict__.setdefault("_may_have", {})
    if attribute not in cache:
        rx = re.compile(ATTRIBUTE_HEADERS[attribute])
        headers = [h for spec in shop.readme.files.values() for h in spec.columns]
        cache[attribute] = any(rx.search(fold(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(h)).replace("_", " "))) for h in headers)
    return cache[attribute]


DEBT_WORDS = r"\bno\b|\bcong no\b|\bcon thieu\b|\bchua tra\b|\bmua chiu\b|\bthu hoi\b|\bcon phai thu\b"


def age_buckets(q):
    """Debt-age ranges named in the question, in text order: '0–7 ngày',
    'trên 15 ngày', 'quá 7 ngày', 'từ 30 ngày trở lên', 'dưới 8 ngày'."""
    found = []
    for m in re.finditer(r"(?<![\d/])(\d{1,3})\s*(?:-|–)\s*(\d{1,3})(?![\d/])|(\d+) den (\d+) ngay", q):
        lo, hi = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        found.append((m.start(), (int(lo), int(hi))))
    for m in re.finditer(r"(?:tren|qua|hon) (\d+) ngay", q):
        found.append((m.start(), (int(m.group(1)) + 1, None)))
    for m in re.finditer(r"tu (\d+) ngay tro len", q):
        found.append((m.start(), (int(m.group(1)), None)))
    for m in re.finditer(r"duoi (\d+) ngay", q):
        found.append((m.start(), (None, int(m.group(1)) - 1)))
    return [b for _, b in sorted(found)]


def debt_intent(q, period, customers):
    """§4 intents. None: not a debt question; False: a debt question the rules
    do not cover."""
    if not re.search(DEBT_WORDS, q) or re.search(r"\bdoanh thu\b|\btien hang\b", q):
        return None
    asof = period.kind == "asof" or period.kind == "day"
    if re.search(r"no them|mua chiu them|ghi no them", q):
        if customers and re.search(r"\btra\b", q):
            return Intent("debt_change", period, filters={"customer": customers[0][1]})
        return False
    if customers:
        return False
    if re.search(r"thu hoi|tra (?:bot )?no|thu no|thu duoc .*no", q):
        return Intent("debt_collected", period) if not asof else False
    if not asof:
        return False
    snapshot = Period(None, period.end, "asof", period.label)
    buckets = age_buckets(q)
    if buckets or re.search(r"tuoi no", q):
        return Intent("debt_aging", snapshot, values=buckets, numeric=True) if buckets else False
    if re.search(r"(?:nhieu|lon|cao) nhat", q):
        topk = re.search(r"\btop (\d+)\b|(?<![\d/])\b(\d{1,2}) (?:khach|nguoi)\b", q)
        return Intent("debt_top", snapshot, k=int(topk.group(1) or topk.group(2)), numeric=True) if topk else False
    if re.search(r"(?:so|may|bao nhieu) (?:khach|nguoi)", q):
        return Intent("debt_customers", snapshot)
    if re.search(r"bao nhieu|\btong\b", q) and not re.search(r"\bdon\b", q):
        return Intent("debt_total", snapshot)
    return False


def stock_intent(q, period, item, index):
    """§5 closing stock. None: not a stock question; False: one the rules do not cover."""
    if not re.search(r"\bton\b|\btrong kho\b|\bkho con\b|\bcon (?:bao nhieu|may)\b|\bcon hang\b", q):
        return None
    if re.search(r"han dung|het han|\blo\b.*\bhan\b|het hang|\bno\b", q):
        return False
    if period.kind not in ("day", "asof"):
        return False
    snapshot = Period(None, period.end, "asof", period.label)
    if re.search(r"gia von|gia tri", q):
        return Intent("stock_value", snapshot) if not item else False
    if not item:
        return False
    units = [m.group(1) for m in re.finditer(r"(?:bao nhieu|\bmay) ([a-z]+\d?)\b", q)]
    if re.search(r"\bchan\b|\bnguyen\b", q) and re.search(r"\ble\b", q) and len(units) >= 2:
        return Intent("stock_qty", snapshot, filters={"item": item}, values=[units[0]], numeric=True)
    if len(units) == 1:
        return Intent("stock_qty", snapshot, filters={"item": item})
    return False


def asks_amount(q):
    return bool(re.search(r"bao nhieu|\bmay\b|\btong\b|\bcho (?:toi|minh|em) biet\b|\bxem\b|\btinh\b", q))


def classify(q, hits, mentions, today, shop, index):
    def of(kind):
        return [(s, v) for s, _, k, v in hits if k == kind]

    filters = {}
    has_morning, has_afternoon = bool(re.search(r"\bca sang\b", q)), bool(re.search(r"\bca chieu\b", q))
    if has_morning != has_afternoon:
        filters["shift"] = "morning" if has_morning else "afternoon"
    groups, items, channels, cgroups, staff = of("group"), of("item"), of("channel"), of("customer_group"), of("staff")
    pays = payments_in(q) if re.search(r"thanh toan|tra bang|\bdon\b|ban bang|tra tien|hinh thuc", q) else []
    ing = of("ingredient")
    if ing:
        filters["ingredient"] = ing[0][1]
        items = []
    if re.search(r"\b(?:khong|k|ko|hok|hong|chang) (?:can )?(?:ke|can|theo) (?:don|toa)|khong can toa|\botc\b|tru (?:thuoc |hang )?ke don", q):
        filters["rx"] = False
    elif re.search(r"\bke don\b|\btheo don\b|\bke toa\b", q):
        filters["rx"] = True
    if cgroups:
        filters["customer_group"] = cgroups[0][1]
    elif groups:
        filters["group"] = groups[0][1]
    if items and "group" not in filters:
        filters["item"] = items[0][1]
    if staff:
        filters["staff"] = staff[0][1]
    multi = None
    if len(channels) >= 2:
        multi = [("channel", v) for _, v in channels]
    elif len(pays) >= 2:
        multi = [("payment_method", v) for _, v in pays]
    else:
        if channels:
            filters["channel"] = channels[0][1]
        if pays and re.search(r"thanh toan|tra bang|ban bang|tra tien|\bdon\b", q):
            filters["payment_method"] = pays[0][1]

    # Primary period: the first mention that is not a relative comparison base.
    primary = None
    for m in mentions:
        if not RELATIVE_BASE.fullmatch(m.text.strip()) and not RELATIVE_BASE.match(m.text):
            primary = m
            break
    primary = primary or mentions[0]
    period = primary.period
    others = [m for m in mentions if m is not primary]
    percent = bool(re.search(r"%|phan tram", q))
    change = bool(re.search(r"tang|giam|thay doi|hon hay kem|chenh|so voi|so cung|\bso\b", q))

    # §7: a filter on something the shop never records cannot be answered.
    lacking = missing_attribute(q, shop)
    if lacking == "?":
        return None
    if lacking:
        return Intent("unanswerable", period, values=[f"shop does not record {lacking}"], numeric=True)
    # A named group / ingredient / staff member that matches nothing the shop
    # has is not understood: never answer for the whole shop instead.
    item = filters.get("item")
    customers = of("customer")
    debt = debt_intent(q, period, customers)
    if debt is not None:
        return debt or None
    if customers:
        return None  # a named customer outside the debt family is not handled
    if re.search(r"\bnhom\b", q) and not (groups or cgroups):
        return None
    if re.search(r"\bhoat chat\b", q) and not ing:
        return None
    if re.search(r"\bkenh\b", q) and not channels:
        return None

    months = month_list(q, mentions, today) if re.search(r"\bthang\b", q) else None
    limit = threshold(q)
    if months and len(months) >= 2 and re.search(r"doanh thu|ban duoc|tien", q):
        if re.search(r"thang nao", q):
            return Intent("best_month", period, filters=filters, values=months, measure="low" if re.search(r"it nhat|thap nhat|kem nhat", q) else "high")
        if re.search(r"tang lien tuc|giam lien tuc|luon cao hon|luon thap hon|tang deu|giam deu", q):
            return Intent("increasing", period, filters=filters, values=months, measure="down" if re.search(r"giam lien tuc|luon thap hon|giam deu", q) else "up")
        if re.search(r"moi thang|lan luot|tung thang|theo thu tu|tu thang cu", q):
            return Intent("month_values", period, filters=filters, values=months)
    if limit and re.search(r"\bmay ngay\b|bao nhieu ngay|so ngay", q) and re.search(r"doanh thu|ban duoc", q):
        return Intent("count_days", period, filters=filters, values=[limit])
    if limit and re.search(r"(?:bao nhieu|may|so) (?:mat hang|san pham|mon|hang)\b|so (?:mat hang|san pham|mon)", q) and re.search(r"\bdon\b", q):
        filters.pop("item", None)
        return Intent("items_in_orders", period, filters=filters, values=[limit])
    if limit and re.search(r"\bkhach\b", q) and re.search(r"\bdon\b", q):
        filters.pop("item", None)
        if percent:
            return Intent("customers_share", period, filters=filters, values=[limit])
        return Intent("customers_with_orders", period, filters=filters, values=[limit])
    if limit and re.search(r"(?:bao nhieu|may|so) don", q) and re.search(r"doanh thu|tien hang|gia tri|tri gia|tien", q):
        return Intent("count_orders_by_value", period, filters=filters, values=[limit])
    effect = re.search(r"(\d+) ngay (?:ke tu|tinh tu|tu)", q)
    if effect and re.search(r"tang gia|len gia|doi gia|giam gia|ha gia", q) and filters.get("item"):
        n = int(effect.group(1))
        days = [x.period for x in mentions if x.period.kind == "day"]
        start = days[0].start if days else (period.start if period.start else None)
        if start:
            cur = Period(start, start + dt.timedelta(days=n - 1), "range")
            base = Period(start - dt.timedelta(days=n), start - dt.timedelta(days=1), "range")
            return Intent("growth", cur, base, filters={"item": filters["item"]}, measure="qty", numeric=True)

    def base_period():
        if re.search(r"cung ky nam (?:truoc|ngoai)|cung thang nam (?:truoc|ngoai)|so voi nam (?:truoc|ngoai)|cung ky nam ngoai", q):
            return same_period_last_year(period)
        m2 = re.search(r"(\d+) ngay (?:ngay )?truoc tet", q)
        if m2 and period.kind == "tet":
            n = int(m2.group(1))
            return Period(period.start - dt.timedelta(days=n), period.start - dt.timedelta(days=1), "range")
        for m in others:
            if m.period != period and (m.period.kind == "tet" or not RELATIVE_BASE.search(m.text)):
                return m.period
        return previous_period(period, today)

    cust_rank = re.search(r"\bkhach\b|\bai\b", q) and re.search(
        r"nhieu tien nhat|chi tieu nhieu nhat|mua nhieu nhat|mua nhieu lan nhat|nhieu don nhat|nhieu don hang nhat|tong tien mua cao nhat|mua (?:hang )?nhieu nhat", q
    )
    if cust_rank:
        filters.pop("item", None)
        measure = "orders" if re.search(r"nhieu lan|nhieu don|so don|so lan", q) else "revenue"
        topk = re.search(r"\btop (\d+)\b|(?<![\d/])\b(\d{1,2}) (?:khach|nguoi)\b", q)
        if topk:
            return Intent("top_customers", period, filters=filters, k=int(topk.group(1) or topk.group(2)), measure=measure)
        return Intent("best_customer", period, filters=filters, measure=measure)
    if re.search(r"\bnhan vien\b|\bnguoi ban\b", q) and re.search(r"\bnao\b|\bai\b", q) and re.search(r"gioi nhat|doanh thu cao nhat|ban (?:duoc )?nhieu (?:tien )?nhat|nhieu doanh thu nhat", q):
        filters.pop("item", None)
        filters.pop("staff", None)
        return Intent("best_staff", period, filters=filters)
    if re.search(r"\bnhan vien\b|\bnguoi ban\b", q) and "staff" not in filters:
        return None
    if re.search(r"\bnhap\b", q):  # §5 purchases
        if re.search(r"nha cung cap|\bncc\b", q):
            if re.search(r"\bnao\b", q) and re.search(r"nhieu|cao|lon", q):
                return Intent("top_supplier", period)
            return None
        if re.search(r"nhap tru|tru (?:di )?(?:so )?ban|chenh", q):
            return Intent("purchase_minus_sales", period, filters={"item": item}) if item else None
        if re.search(r"binh quan|trung binh", q):
            return Intent("purchase_avg", period, filters={"item": item}) if item else None
        if not item and asks_amount(q) and re.search(r"\btien\b|gia tri|tong", q):
            return Intent("purchase_total", period)
        return None
    stock = stock_intent(q, period, item, index)
    if stock is not None:
        return stock or None
    if re.search(r"\b(?:mat hang|mon|san pham|hang) (?:bi )?e\b|\be (?:han|den muc)\b|khong ban (?:ra )?(?:duoc|noi) (?:cai|mon|lan|san pham) nao|khong ai mua|ban cham den muc", q):
        if re.search(r"bao nhieu|\bmay\b|\bdem\b|\bso (?:mat hang|mon|san pham)\b", q) and not item:
            return Intent("unsold_count", period)
        return None
    if re.search(r"lai gop|loi gop|loi nhuan gop|\bloi nhat\b|\blai nhat\b", q) and not re.search(r"lai rong|loi nhuan rong|chi phi", q):
        if re.search(r"\b(?:nao|gi)\b", q) and re.search(r"(?:cao|lon|nhieu|loi|lai) nhat", q):
            filters.pop("item", None)
            return Intent("most_profitable_item", period, filters=filters)
        if not item and not percent and not re.search(r"nhat\b", q):
            return Intent("gross_profit", period, filters=filters)
        return None
    if re.search(r"\bkhach\b.*\b(?:nao|ai)\b|\bai\b.*\bkhach\b|\bla ai\b|\bton\b|\bkho\b|\bnhap\b|cong no|\bno\b|\bloi\b|lai gop|lai rong|chi phi|het han|giao hang|dot giao", q):
        return None
    share = percent and re.search(r"\b(?:ti|ty) trong\b|\bchiem\b|\bphan cua\b|dong gop|\bphan\b.*\btrong tong\b|\bbang\b.*\btong doanh thu\b|\btren tong\b", q)
    if share:
        part = {k: v for k, v in filters.items() if k in ("group", "item", "channel", "customer_group", "payment_method", "rx", "ingredient")}
        if "group" in part:
            part.pop("item", None)
        if part:
            rest = {k: v for k, v in filters.items() if k not in part}
            return Intent("share", period, filters=rest, values=list(part.items()))
        return None
    if re.search(r"\bhuy\b", q) and (percent or re.search(r"\b(?:ti|ty) le\b", q)):
        filters.pop("item", None)
        return Intent("cancel_rate", period, filters=filters)
    if percent and change:
        if has_morning and has_afternoon:
            measure = "orders" if re.search(r"\bso don\b|\bdon\b", q) and "doanh thu" not in q else "revenue"
            return Intent("shift_growth", period, filters=filters, measure=measure)
        measure = "orders" if re.search(r"\bso don\b", q) and "doanh thu" not in q else "revenue"
        if re.search(r"so luong|ban (?:duoc|ra)", q) and filters.get("item") and "doanh thu" not in q:
            measure = "qty"
        base = base_period()
        return Intent("growth", period, base, filters=filters, measure=measure) if base else None
    if percent:
        return None
    if re.search(r"\bdon gia\b|\bgia ban\b|\bgia bao nhieu\b|\bvoi gia\b|\bgia (?:mot|1)\b", q) and filters.get("item") and period.kind == "day":
        return Intent("price_on_day", period, filters={"item": filters["item"]})
    if re.search(r"\bngay nao\b", q):
        measure = "orders" if re.search(r"nhieu don|dong khach|dat khach|dong don|dong nhat", q) else "revenue"
        return Intent("best_day", period, filters=filters, measure=measure)
    if re.search(r"\bhuy\b", q) and re.search(r"\bdon\b", q):
        filters.pop("item", None)
        return Intent("cancelled_count", period, filters=filters)
    if re.search(r"trung binh|binh quan|gia tri trung binh", q) and re.search(r"\bdon\b", q):
        return Intent("aov", period, filters=filters)
    rank = re.search(r"ban chay|chay nhat|ban nhieu nhat|ban duoc nhieu nhat|ban ra nhieu nhat|hot nhat", q)
    top = re.search(r"\btop (\d+)\b|(?<![\d/])\b(\d{1,2}) (?:mat hang|mon|san pham|hang)\b", q)
    if rank:
        filters.pop("item", None)
        if top:
            return Intent("top_items", period, filters=filters, k=int(top.group(1) or top.group(2)), measure="qty")
        return Intent("best_item", period, filters=filters, measure="qty")
    asks_count = re.search(r"bao nhieu|\bmay\b", q)
    size = re.search(r"\bsize ([a-z]{1,3})\b", q)
    if size and asks_count and re.search(r"\bban\b", q) and "item" not in filters:
        return Intent("qty", period, filters={**filters, "size": size.group(1)}, measure="qty")
    if re.search(r"\bdon\b", q) and re.search(r"bao nhieu|\bmay\b|\bso don\b|tong so", q) and not re.search(r"doanh thu|tien", q):
        return Intent("order_count", period, filters=filters)
    # "bán được bao nhiêu <đơn vị>", "số <đơn vị> ... bán được" (§3).
    unit = None
    for m in re.finditer(r"(?:bao nhieu|\bmay|\bso|tinh theo) ([a-z]+\d?)\b", q):
        if m.group(1) in index["unit"]:
            unit = index["unit"][m.group(1)]
            break
    sold = re.search(r"\bban\b", q) and "doanh thu" not in q and not re.search(r"\btien\b", q)
    if unit and sold and asks_count:
        if filters.get("item"):
            return Intent("qty", period, filters=filters, measure="qty")
        return Intent("qty", period, filters={**filters, "unit": unit}, measure="qty")
    if re.search(r"doanh thu|bao nhieu tien|ban duoc .*tien|thu ve|tien hang|tien ban|ban duoc tong cong", q) and not re.search(r"\bloi\b|lai gop|nhap|cong no|\bton\b|\bno\b", q):
        if multi:
            return Intent("revenue_list", period, filters=filters, values=multi)
        if change and others and re.search(r"chenh|hon hay kem|tang hay giam|bao nhieu tien", q) and not percent:
            base = base_period()
            return Intent("difference", period, base, filters=filters) if base else None
        return Intent("revenue", period, filters=filters)
    if filters.get("item") and re.search(r"ban (?:duoc|ra)? ?(?:bao nhieu|may)|so luong", q):
        return Intent("qty", period, filters=filters, measure="qty")
    return None


def run(intent, metrics):
    """Execute an intent. Missing information in the shop's data is §7
    unanswerable; other gaps raise Unsupported for the caller's fallback."""
    try:
        return _run(intent, metrics)
    except NoData as exc:
        return {"type": "unanswerable", "reason": str(exc)}


def _run(intent, metrics):
    p, kw = intent.period, dict(intent.filters)
    if intent.metric == "unanswerable":
        return {"type": "unanswerable", "reason": str(intent.values[0])}
    if not metrics.check_period(p):
        return {"type": "unanswerable", "reason": "period outside the data range"}
    m = intent.metric
    if m == "revenue":
        return metrics.revenue(p, **kw)
    if m == "revenue_list":
        values = []
        for key, value in intent.values:
            values.append(metrics.revenue(p, **{**kw, key: value})["answer"])
        return {"type": "list", "answer": values}
    if m == "top_customers":
        return metrics.top_customers(p, intent.k, intent.measure, **kw)
    if m == "best_customer":
        return metrics.best_customer(p, intent.measure, **kw)
    if m == "best_month":
        return metrics.best_month(intent.values, lowest=intent.measure == "low", **kw)
    if m == "increasing":
        return metrics.increasing(intent.values, decreasing=intent.measure == "down", **kw)
    if m == "month_values":
        return metrics.month_values(intent.values, **kw)
    if m == "count_days":
        return metrics.count_days(p, *intent.values[0], **kw)
    if m == "items_in_orders":
        return metrics.items_in_orders(p, *intent.values[0], **kw)
    if m == "customers_with_orders":
        return metrics.customers_with_orders(p, *intent.values[0], **kw)
    if m == "customers_share":
        return metrics.customers_with_orders_share(p, *intent.values[0], **kw)
    if m == "count_orders_by_value":
        return metrics.count_orders_by_value(p, *intent.values[0], **kw)
    if m == "share":
        return metrics.share(p, dict(intent.values), **kw)
    if m == "price_on_day":
        return metrics.price_on_day(p, **kw)
    if m == "order_count":
        return metrics.order_count(p, **kw)
    if m == "cancelled_count":
        return metrics.cancelled_count(p, **kw)
    if m == "cancel_rate":
        return metrics.cancel_rate(p, **kw)
    if m == "aov":
        return metrics.aov(p, **kw)
    if m == "growth":
        return metrics.growth(p, intent.base, intent.measure, **kw)
    if m == "difference":
        return metrics.difference(p, intent.base, intent.measure, **kw)
    if m == "best_day":
        return metrics.best_day(p, intent.measure, **kw)
    if m == "top_items":
        return metrics.top_items(p, intent.k, intent.measure, **kw)
    if m == "stock_qty":
        return metrics.stock_qty(p, kw["item"], intent.values[0] if intent.values else None)
    if m == "stock_value":
        return metrics.stock_value(p)
    if m == "unsold_count":
        return metrics.unsold_count(p)
    if m == "gross_profit":
        return metrics.gross_profit(p, **kw)
    if m == "most_profitable_item":
        return metrics.most_profitable_item(p, **kw)
    if m in ("growth", "difference") and intent.base is not None and not metrics.check_period(intent.base):
        return {"type": "unanswerable", "reason": "comparison period outside the data range"}
    if m == "debt_total":
        return metrics.debt_total(p)
    if m == "debt_customers":
        return metrics.debt_customers(p)
    if m == "debt_top":
        return metrics.debt_top(p, intent.k)
    if m == "debt_aging":
        return metrics.debt_aging(p, intent.values)
    if m == "debt_collected":
        return metrics.debt_collected(p)
    if m == "debt_change":
        return metrics.debt_change(p, kw["customer"])
    if m == "purchase_total":
        return metrics.purchase_total(p)
    if m == "purchase_avg":
        return metrics.purchase_avg_price(p, kw["item"])
    if m == "top_supplier":
        return metrics.top_supplier(p)
    if m == "purchase_minus_sales":
        return metrics.purchase_minus_sales(p, kw["item"])
    if m == "best_staff":
        return metrics.best_staff(p, **kw)
    if m == "best_item":
        return metrics.best_item(p, intent.measure, **kw)
    if m == "qty":
        return {"type": "number", "answer": number(metrics.qty_value(p, **kw))}
    if m == "shift_growth":
        cur = metrics.measure(intent.measure, p, shift="afternoon", **kw)
        old = metrics.measure(intent.measure, p, shift="morning", **kw)
        if old == 0:
            raise Unsupported("morning is zero")
        return {"type": "percent", "answer": pct((Decimal(cur) - Decimal(old)) * 100 / Decimal(old))}
    raise Unsupported(m)
