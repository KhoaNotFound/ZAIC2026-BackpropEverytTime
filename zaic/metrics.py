"""Rulebook metrics over a `semantic.Shop`.

Each metric returns a final answer dict ({"type": ..., "answer": ...}) or raises
`Unsupported` when the shop's data cannot compute it with confidence, so the
caller can fall back to the agent. Rounding follows rulebook §9: money to whole
đồng (half up), percentages and other averages to 2 decimals.
"""

import datetime as dt
import re
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise

from .values import fold

COUNTED = ("done", "shipping")  # §1: not cancelled, not returned


class Unsupported(Exception):
    """The semantic layer cannot answer this; use another path."""


class NoData(Exception):
    """The shop does not record the information the question needs (§7)."""


def compare(value, op, target):
    value, target = Decimal(value), Decimal(target)
    return {
        ">=": value >= target,
        ">": value > target,
        "<=": value <= target,
        "<": value < target,
        "=": value == target,
    }[op]


def money(value):
    return int(Decimal(value).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def pct(value):
    return float(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def number(value):
    value = Decimal(value)
    if value == value.to_integral_value():
        return int(value)
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class Metrics:
    def __init__(self, shop):
        self.shop = shop
        self.lines = shop.lines or []
        self.orders = shop.orders or []
        lo, hi = shop.date_range()
        self.first_day = lo
        self.last_day = shop.data_end or hi

    # ----------------------------------------------------------- coverage
    def check_period(self, period):
        """§7: a period entirely outside the data is unanswerable, not zero."""
        if self.first_day is None:
            raise Unsupported("no dated sales")
        start = period.start or self.first_day
        return not (period.end < self.first_day or start > self.last_day)

    def need_orders(self):
        if not self.orders:
            raise Unsupported("shop has no order ids")

    # ------------------------------------------------------------ filters
    def order_rows(self, period, statuses=COUNTED, shift=None, **filters):
        self.need_orders()
        out = []
        for o in self.orders:
            day = o.get("day")
            if day is None or not period.contains(day):
                continue
            if statuses and o["status"] not in statuses:
                continue
            if shift and not in_shift(o.get("time"), shift):
                continue
            if not match_filters(o, filters):
                continue
            out.append(o)
        return out

    def line_rows(self, period, statuses=COUNTED, item=None, group=None, kinds=("item", "service"), shift=None, **filters):
        if getattr(self.shop, "combined_items", False):
            raise Unsupported("items share cells with toppings")
        out = []
        for r in self.lines:
            day = r.get("day")
            if day is None or not period.contains(day):
                continue
            if statuses and r.get("status") not in statuses:
                continue
            if kinds and r.get("kind") not in kinds:
                continue
            if item is not None and not item_matches(r, item):
                continue
            if group is not None and fold(r.get("group") or "") != fold(group):
                continue
            if shift and not in_shift(r.get("time"), shift):
                continue
            if not match_filters(r, filters):
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------ metrics
    def revenue_value(self, period, item=None, group=None, shift=None, **filters):
        if item is not None or group is not None or "rx" in filters or "ingredient" in filters:
            rows = self.line_rows(period, item=item, group=group, shift=shift, **filters)
            if not rows and not self.lines:
                raise Unsupported("no line items")
            return sum((r["line_amount"] for r in rows if r.get("line_amount") is not None), Decimal(0))
        if self.orders:
            rows = self.order_rows(period, shift=shift, **filters)
            return sum((o["revenue"] for o in rows if o.get("revenue") is not None), Decimal(0))
        # No order ids: every line is its own record (§8 "bảng không ghi theo đơn").
        rows = self.line_rows(period, shift=shift, **filters)
        return sum((r["line_amount"] for r in rows if r.get("line_amount") is not None), Decimal(0))

    def revenue(self, period, **kw):
        return {"type": "number", "answer": money(self.revenue_value(period, **kw))}

    def order_count_value(self, period, statuses=COUNTED, **kw):
        if kw.get("item") is not None:
            item = kw.pop("item")
            rows = self.line_rows(period, statuses=statuses, item=item, **kw)
            return len({r["order_id"] for r in rows if r.get("order_id")})
        return len(self.order_rows(period, statuses=statuses, **kw))

    def order_count(self, period, **kw):
        return {"type": "number", "answer": self.order_count_value(period, **kw)}

    def cancelled_count(self, period, **kw):
        return {"type": "number", "answer": self.order_count_value(period, statuses=("cancelled",), **kw)}

    def cancel_rate(self, period, **kw):
        total = self.order_count_value(period, statuses=None, **kw)
        if total == 0:
            raise Unsupported("no orders in period")
        cancelled = self.order_count_value(period, statuses=("cancelled",), **kw)
        return {"type": "percent", "answer": pct(Decimal(cancelled) * 100 / total)}

    def aov(self, period, **kw):
        n = self.order_count_value(period, **kw)
        if n == 0:
            raise Unsupported("no orders in period")
        return {"type": "number", "answer": money(self.revenue_value(period, **kw) / n)}

    def growth(self, period, base, measure="revenue", **kw):
        cur = self.measure(measure, period, **kw)
        old = self.measure(measure, base, **kw)
        if old == 0:
            raise Unsupported("base period is zero")
        return {"type": "percent", "answer": pct((Decimal(cur) - Decimal(old)) * 100 / Decimal(old))}

    def difference(self, period, base, measure="revenue", **kw):
        cur = self.measure(measure, period, **kw)
        old = self.measure(measure, base, **kw)
        value = Decimal(cur) - Decimal(old)
        return {"type": "number", "answer": money(value) if measure == "revenue" else number(value)}

    def measure(self, measure, period, **kw):
        if measure == "revenue":
            return self.revenue_value(period, **kw)
        if measure == "orders":
            return self.order_count_value(period, **kw)
        if measure == "qty":
            return self.qty_value(period, **kw)
        raise Unsupported(measure)

    def best_day(self, period, measure="revenue", **kw):
        totals = defaultdict(Decimal)
        if measure == "revenue":
            if self.orders and not kw.get("item"):
                for o in self.order_rows(period, **kw):
                    totals[o["day"]] += o.get("revenue") or 0
            else:
                for r in self.line_rows(period, statuses=COUNTED, **kw):
                    totals[r["day"]] += r.get("line_amount") or 0
        elif measure == "orders":
            for o in self.order_rows(period, **kw):
                totals[o["day"]] += 1
        else:
            raise Unsupported(measure)
        if not totals:
            raise Unsupported("no sales in period")
        best = max(totals.values())
        day = min(d for d, v in totals.items() if v == best)
        return {"type": "date", "answer": day.isoformat()}

    def qty_value(self, period, item=None, unit=None, **kw):
        rows = self.line_rows(period, statuses=COUNTED, item=item, kinds=("item",), **kw)
        total = Decimal(0)
        for r in rows:
            if r.get("base_qty") is None:
                continue
            total += r["base_qty"]
        return total

    def item_ranking(self, period, measure="qty", **kw):
        totals = defaultdict(Decimal)
        codes = {}
        rows = self.line_rows(period, statuses=COUNTED, kinds=("item",), **kw)
        for r in rows:
            name = r.get("item_name") or r.get("item_code")
            if name is None:
                continue
            if measure == "qty":
                totals[name] += r.get("base_qty") or 0
            elif measure == "revenue":
                totals[name] += r.get("line_amount") or 0
            elif measure == "profit":
                cost = r.get("unit_cost")
                if cost is None or r.get("qty") is None:
                    raise Unsupported("no line cost")
                totals[name] += (r.get("line_amount") or 0) - r["qty"] * cost
            codes.setdefault(name, r.get("item_code") or name)
        if not totals:
            raise Unsupported("no items in period")
        # §9: ties by code ascending (name when there is no code).
        return sorted(totals, key=lambda n: (-totals[n], str(codes[n]))), totals

    def share(self, period, part, **kw):
        """§1: revenue of a part (group, item, channel, customer group, payment
        method) as a percentage of the whole shop's revenue in the same period,
        walk-in orders included in the denominator."""
        total = self.revenue_value(period, **kw)
        if total == 0:
            raise Unsupported("no revenue in period")
        value = self.revenue_value(period, **{**kw, **part})
        return {"type": "percent", "answer": pct(value * 100 / total)}

    def price_on_day(self, period, item, **kw):
        """§5: unit price on the sale lines of `item` on day D, per base unit,
        before line discount, including cancelled and returned orders."""
        rows = self.line_rows(period, statuses=None, item=item, kinds=("item",))
        prices = Counter()
        for r in rows:
            price = r.get("unit_price")
            if price is None and r.get("line_amount") is not None and r.get("qty"):
                price = (r["line_amount"] + (r.get("line_discount") or 0)) / r["qty"]
            if price is None:
                continue
            qty, base = r.get("qty"), r.get("base_qty")
            factor = (base / qty) if qty and base else 1
            prices[money(price / factor) if factor else money(price)] += 1
        if not prices:
            raise Unsupported("no sale line for item on that day")
        return {"type": "number", "answer": prices.most_common(1)[0][0]}

    def month_values(self, months, **kw):
        return {"type": "list", "answer": [money(self.revenue_value(m, **kw)) for m in months]}

    def best_month(self, months, lowest=False, **kw):
        values = [(self.revenue_value(m, **kw), m) for m in months]
        target = min(v for v, _ in values) if lowest else max(v for v, _ in values)
        month = next(m for v, m in values if v == target)
        return {"type": "date", "answer": f"{month.start.year:04d}-{month.start.month:02d}"}

    def increasing(self, months, decreasing=False, **kw):
        values = [self.revenue_value(m, **kw) for m in months]
        pairs = list(pairwise(values))
        ok = all(b < a for a, b in pairs) if decreasing else all(b > a for a, b in pairs)
        return {"type": "boolean", "answer": ok}

    def daily_revenue(self, period, **kw):
        totals = defaultdict(Decimal)
        if self.orders and not kw.get("item") and not kw.get("group"):
            for o in self.order_rows(period, **kw):
                totals[o["day"]] += o.get("revenue") or 0
        else:
            for r in self.line_rows(period, statuses=COUNTED, **kw):
                totals[r["day"]] += r.get("line_amount") or 0
        return totals

    def count_days(self, period, op, value, **kw):
        totals = self.daily_revenue(period, **kw)
        return {"type": "number", "answer": sum(1 for v in totals.values() if compare(v, op, value))}

    def count_orders_by_value(self, period, op, value, **kw):
        rows = self.order_rows(period, **kw)
        return {"type": "number", "answer": sum(1 for o in rows if compare(o.get("revenue") or 0, op, value))}

    def items_in_orders(self, period, op, value, **kw):
        self.need_orders()
        orders = defaultdict(set)
        for r in self.line_rows(period, kinds=("item",), **kw):
            name = r.get("item_name") or r.get("item_code")
            if name and r.get("order_id"):
                orders[name].add(r["order_id"])
        return {"type": "number", "answer": sum(1 for v in orders.values() if compare(len(v), op, value))}

    def customer_key(self, row):
        """§3: customer code when the shop has codes, else the name ignoring
        spelling (case/accents/spaces); walk-in customers are not customers."""
        cid = row.get("customer_id")
        if cid:
            return ("id", fold(cid))
        name = row.get("customer_name")
        if not name:
            return None
        key = fold(name)
        if re.fullmatch(r"(khach )?(le|vang lai)|kl|khach|walk ?in|guest", key):
            return None
        return ("name", re.sub(r"[^a-z0-9]", "", key))

    def has_customers(self):
        return any(o.get("customer_id") or o.get("customer_name") for o in self.orders)

    def customer_orders(self, period, **kw):
        if not self.orders or not self.has_customers():
            raise NoData("shop does not record customers")
        counts = Counter()
        for o in self.order_rows(period, **kw):
            k = self.customer_key(o)
            if k:
                counts[k] += 1
        return counts

    def customers_with_orders(self, period, op, value, **kw):
        counts = self.customer_orders(period, **kw)
        return {"type": "number", "answer": sum(1 for v in counts.values() if compare(v, op, value))}

    def customers_with_orders_share(self, period, op, value, **kw):
        counts = self.customer_orders(period, **kw)
        if not counts:
            raise Unsupported("no buyers in period")
        n = sum(1 for v in counts.values() if compare(v, op, value))
        return {"type": "percent", "answer": pct(Decimal(n) * 100 / len(counts))}

    def customer_ranking(self, period, measure="revenue", **kw):
        if not self.orders or not self.has_customers():
            raise NoData("shop does not record customers")
        totals, names, codes = defaultdict(Decimal), {}, {}
        for o in self.order_rows(period, **kw):
            k = self.customer_key(o)
            if not k:
                continue
            totals[k] += (o.get("revenue") or 0) if measure == "revenue" else 1
            names.setdefault(k, o.get("customer_name") or o.get("customer_id"))
            codes.setdefault(k, o.get("customer_id") or o.get("customer_name"))
        if not totals:
            raise Unsupported("no customers in period")
        ranked = sorted(totals, key=lambda k: (-totals[k], str(codes[k])))
        return [names[k] for k in ranked], totals

    def top_customers(self, period, k, measure="revenue", **kw):
        names, _ = self.customer_ranking(period, measure, **kw)
        return {"type": "list", "answer": names[:k]}

    def best_customer(self, period, measure="revenue", **kw):
        names, _ = self.customer_ranking(period, measure, **kw)
        return {"type": "string", "answer": names[0]}

    def top_items(self, period, k, measure="qty", **kw):
        names, _ = self.item_ranking(period, measure, **kw)
        return {"type": "list", "answer": names[:k]}

    def best_item(self, period, measure="qty", **kw):
        names, _ = self.item_ranking(period, measure, **kw)
        return {"type": "string", "answer": names[0]}


def in_shift(time, shift):
    if time is None:
        return False
    if shift == "morning":
        return dt.time(6, 0) <= time <= dt.time(13, 59, 59)
    if shift == "afternoon":
        return dt.time(14, 0) <= time <= dt.time(21, 59, 59)
    return True


def item_matches(row, item):
    target = fold(item)
    return any(fold(row.get(k) or "") == target for k in ("item_name", "item_code"))


def match_filters(row, filters):
    for k, v in filters.items():
        if v is None:
            continue
        if k == "payment_method":
            if payment_kind(row.get("payment_method")) != v:
                return False
        elif k == "rx":
            if bool(row.get("rx")) != v:
                return False
        elif fold(row.get(k) or "") != fold(v):
            return False
    return True


PAYMENT_WORDS = [
    ("cod", ("cod", "giao hang thu tien", "thu ho")),
    ("debt", ("debt", "ghi no", "no", "cong no")),
    ("transfer", ("transfer", "chuyen khoan", "ck", "bank")),
    ("ewallet", ("ewallet", "vi dien tu", "momo", "zalopay", "vnpay", "vi")),
    ("card", ("card", "the", "quet the", "pos")),
    ("cash", ("cash", "tien mat", "tm", "tien tuoi")),
]


def payment_kind(value):
    text = fold(value or "")
    for kind, words in PAYMENT_WORDS:
        if text in words or any(w in text for w in words if len(w) > 3):
            return kind
    return text or None


__all__ = ["Counter", "Metrics", "Unsupported", "money", "payment_kind", "pct"]
