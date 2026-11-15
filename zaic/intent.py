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

from .metrics import Unsupported, number, pct
from .periods import (
    Period,
    find_periods,
    normalize_question,
    previous_period,
    same_period_last_year,
)
from .values import fold

RELATIVE_BASE = re.compile(
    r"\b(?:thang|tuan|quy|nam|ky) (?:truoc|roi|lien truoc|ngay truoc)(?: do)?\b|\bcung ky\b|\btruoc tet\b"
)
PAYMENT_PHRASES = [
    ("cash", r"tien mat|tien tuoi"),
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


@dataclass
class Intent:
    metric: str
    period: Period
    base: Period | None = None
    filters: dict = field(default_factory=dict)
    k: int | None = None
    measure: str = "revenue"
    values: list = field(default_factory=list)  # for list answers: [(filter key, value)]


def entity_index(shop):
    """Folded value -> original value for each entity kind the shop records."""
    index = {"item": {}, "group": {}, "customer_group": {}, "channel": {}, "ingredient": {}}
    for r in shop.lines or []:
        if r.get("kind") != "item":
            continue
        if r.get("item_name"):
            index["item"].setdefault(fold(r["item_name"]), r["item_name"])
        if r.get("group"):
            index["group"].setdefault(fold(r["group"]), r["group"])
        if r.get("ingredient"):
            index["ingredient"].setdefault(fold(r["ingredient"]), r["ingredient"])
    for o in shop.orders or []:
        for k in ("customer_group", "channel"):
            if o.get(k):
                index[k].setdefault(fold(o[k]).replace("_", " "), o[k])
    return index


def channels_in(q, channel_index):
    """(position, raw channel value) for channel phrases in q. Raw values may be
    codes ('store', 'zalo_oa'), so known synonyms are matched too."""
    hits = find_all(q, channel_index)
    for value_key, raw in channel_index.items():
        for value_rx, question_rx in CHANNEL_SYNONYMS:
            if re.search(value_rx, value_key):
                for m in re.finditer(question_rx, q):
                    hits.append((m.start(), raw))
                break
    hits.sort()
    seen, out = set(), []
    for pos, raw in hits:
        if raw not in seen:
            seen.add(raw)
            out.append((pos, raw))
    return out


def find_all(text, mapping):
    """All (position, value) of mapping keys in text, longest match wins per span."""
    hits = []
    for k, v in mapping.items():
        if len(k) < 2:
            continue
        for m in re.finditer(r"(?<![\w])" + re.escape(k) + r"(?![\w])", text):
            hits.append((m.start(), m.end(), v))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    out, last = [], -1
    for start, end, v in hits:
        if start >= last:
            out.append((start, v))
            last = end
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


def parse(question, shop, calendar_path, today=None, index=None):
    q = normalize_question(question["question"])
    plain = fold(question["question"])  # entity names: no typo repair
    today = today or dt.date.fromisoformat(question["as_of_date"])
    mentions = find_periods(q, today, calendar_path)
    if not mentions:
        return None
    index = index or entity_index(shop)
    filters = {}
    has_morning, has_afternoon = bool(re.search(r"\bca sang\b", q)), bool(re.search(r"\bca chieu\b", q))
    if has_morning != has_afternoon:
        filters["shift"] = "morning" if has_morning else "afternoon"
    groups = find_all(plain, index["group"]) if re.search(r"\bnhom\b", q) else []
    items = find_all(plain, index["item"])
    channels = channels_in(plain, index["channel"]) if re.search(r"\bkenh\b|\bqua\b|\btai\b|\bban\b", q) else []
    cgroups = find_all(plain, index["customer_group"]) if re.search(r"\bkhach\b", q) else []
    pays = payments_in(q) if re.search(r"thanh toan|tra bang|\bdon\b|ban bang|tra tien|hinh thuc", q) else []
    if re.search(r"hoat chat", q) and index["ingredient"]:
        ing = find_all(plain, index["ingredient"])
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
            return Intent("growth", cur, base, filters={"item": filters["item"]}, measure="qty")

    def base_period():
        if re.search(r"cung ky nam (?:truoc|ngoai)|cung thang nam (?:truoc|ngoai)|so voi nam (?:truoc|ngoai)|cung ky nam ngoai", q):
            return same_period_last_year(period)
        m2 = re.search(r"(\d+) ngay (?:ngay )?truoc tet", q)
        if m2 and period.kind == "tet":
            n = int(m2.group(1))
            return Period(period.start - dt.timedelta(days=n), period.start - dt.timedelta(days=1), "range")
        for m in others:
            if not RELATIVE_BASE.search(m.text) and m.period != period:
                return m.period
        return previous_period(period, today)

    cust_rank = re.search(r"\bkhach\b|\bai\b", q) and re.search(r"nhieu tien nhat|chi tieu nhieu nhat|mua nhieu nhat|mua nhieu lan nhat|nhieu don nhat|tong tien mua cao nhat|mua (?:hang )?nhieu nhat", q)
    if cust_rank:
        filters.pop("item", None)
        measure = "orders" if re.search(r"nhieu lan|nhieu don|so don|so lan", q) else "revenue"
        topk = re.search(r"\btop (\d+)\b|(?<![\d/])\b(\d{1,2}) (?:khach|nguoi)\b", q)
        if topk:
            return Intent("top_customers", period, filters=filters, k=int(topk.group(1) or topk.group(2)), measure=measure)
        return Intent("best_customer", period, filters=filters, measure=measure)
    if re.search(r"\bkhach\b.*\b(?:nao|ai)\b|\bai\b.*\bkhach\b|\bla ai\b|\bnhan vien\b|\bton\b|\bkho\b|\bnhap\b|cong no|\bno\b|\bloi\b|lai gop|lai rong|chi phi|het han|\blo\b|giao hang|dot giao", q):
        return None
    share = percent and re.search(r"\b(?:ti|ty) trong\b|\bchiem\b|\bphan cua\b|dong gop|\bphan\b.*\btrong tong\b|\bbang\b.*\btong doanh thu\b|\btren tong\b", q)
    if share:
        part = {k: v for k, v in filters.items() if k in ("group", "item", "channel", "customer_group", "payment_method", "rx", "ingredient")}
        if "group" in part:
            part.pop("item", None)
        if part:
            rest = {k: v for k, v in filters.items() if k not in part}
            return Intent("share", period, filters=rest, values=list(part.items()))
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
        return Intent("growth", period, base_period(), filters=filters, measure=measure)
    if re.search(r"\bdon gia\b|\bgia ban\b|\bgia bao nhieu\b|\bvoi gia\b|\bgia (?:mot|1)\b", q) and filters.get("item") and period.kind == "day":
        return Intent("price_on_day", period, filters={"item": filters["item"]})
    if re.search(r"\bngay nao\b", q):
        measure = "orders" if re.search(r"nhieu don|dong khach|dat khach|dong don|dong nhat", q) else "revenue"
        return Intent("best_day", period, filters=filters, measure=measure)
    if re.search(r"\bhuy\b", q) and re.search(r"\bdon\b", q):
        filters.pop("item", None)
        return Intent("cancelled_count", period, filters=filters)
    if re.search(r"trung binh|binh quan|gia tri trung binh", q) and re.search(r"\bdon\b|\border\b|\bbill\b|hoa don", q):
        return Intent("aov", period, filters=filters)
    rank = re.search(r"ban chay|ban nhieu nhat|ban duoc nhieu nhat|hot nhat|ban dc nhieu nhat", q)
    top = re.search(r"\btop (\d+)\b|(?<![\d/])\b(\d{1,2}) (?:mat hang|mon|san pham|hang|sp)\b", q)
    if rank:
        filters.pop("item", None)
        if top:
            return Intent("top_items", period, filters=filters, k=int(top.group(1) or top.group(2)), measure="qty")
        return Intent("best_item", period, filters=filters, measure="qty")
    if re.search(r"\bdon\b", q) and re.search(r"bao nhieu|\bmay\b|\bso don\b|tong so", q) and not re.search(r"doanh thu|tien", q):
        return Intent("order_count", period, filters=filters)
    unit_q = re.search(r"(?:bao nhieu|may) (?!tien\b|don\b|phan tram\b|%)([a-z]+)\b", q)
    if filters.get("item") and unit_q and re.search(r"ban (?:duoc|ra)|so luong|tong cong", q) and "doanh thu" not in q:
        return Intent("qty", period, filters=filters, measure="qty")
    if re.search(r"doanh thu|bao nhieu tien|ban duoc .*tien|thu ve|tien hang|tien ban|ban duoc tong cong", q) and not re.search(r"\bloi\b|lai gop|nhap|cong no|\bton\b|\bno\b", q):
        if multi:
            return Intent("revenue_list", period, filters=filters, values=multi)
        if change and others and re.search(r"chenh|hon hay kem|tang hay giam|bao nhieu tien", q) and not percent:
            return Intent("difference", period, base_period(), filters=filters)
        return Intent("revenue", period, filters=filters)
    if filters.get("item") and re.search(r"ban (?:duoc|ra)? ?(?:bao nhieu|may)|so luong", q):
        return Intent("qty", period, filters=filters, measure="qty")
    return None


def run(intent, metrics):
    """Execute an intent. Missing information in the shop's data is §7
    unanswerable; other gaps raise Unsupported for the caller's fallback."""
    from .metrics import NoData

    try:
        return _run(intent, metrics)
    except NoData as exc:
        return {"type": "unanswerable", "reason": str(exc)}


def _run(intent, metrics):
    p, kw = intent.period, dict(intent.filters)
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
