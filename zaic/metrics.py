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
    def need_times(self, rows):
        if not any(r.get("time") for r in rows):
            raise Unsupported("no order times for shifts")

    def order_rows(self, period, statuses=COUNTED, shift=None, **filters):
        self.need_orders()
        if shift:
            self.need_times(self.orders)
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

    def line_rows(self, period, statuses=COUNTED, item=None, group=None, kinds=("item", "service"), shift=None, size=None, unit=None, **filters):
        if getattr(self.shop, "combined_items", False):
            raise Unsupported("items share cells with toppings")
        if size is not None:
            # §6 cà phê: size M / L = item names ending in "(M)" / "(L)".
            suffix = f"({fold(size)})"
            if not any(fold(r.get("item_name") or "").endswith(suffix) for r in self.lines):
                raise NoData(f"no item has size {size}")
        if unit is not None:
            unit = fold(unit)
        if shift:
            self.need_times(self.lines)
        out = []
        for r in self.lines:
            day = r.get("day")
            if day is None or not period.contains(day):
                continue
            if statuses and r.get("status") not in statuses:
                continue
            if kinds and r.get("kind") not in kinds:
                continue
            if item is not None and not self.is_item(r, item):
                continue
            if group is not None and fold(r.get("group") or "") != fold(group):
                continue
            if size is not None and not fold(r.get("item_name") or "").endswith(suffix):
                continue
            if unit is not None and fold(r.get("base_unit") or "") != unit:
                continue
            if shift and not in_shift(r.get("time"), shift):
                continue
            if not match_filters(r, filters):
                continue
            out.append(r)
        return out

    def is_item(self, row, item):
        """§8: names that point to the same code are the same item."""
        name, codes = self.item_keys(item)
        return fold(row.get("item_name") or "") == name or (row.get("item_code") is not None and fold(row["item_code"]) in codes)

    def item_keys(self, item):
        cache = self.__dict__.setdefault("_item_keys", {})
        if item not in cache:
            name = fold(item)
            codes = {fold(r["item_code"]) for r in self.lines if r.get("item_code") and fold(r.get("item_name") or "") == name}
            codes |= {fold(r["item_code"]) for r in self.lines if r.get("item_code") and fold(r["item_code"]) == name}
            cache[item] = (name, codes)
        return cache[item]

    # ------------------------------------------------------------ metrics
    def revenue_value(self, period, item=None, group=None, shift=None, **filters):
        if item is not None or group is not None or any(k in filters for k in ("rx", "ingredient", "size", "unit")):
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

    def qty_value(self, period, item=None, **kw):
        """§3: quantity in base units; `unit` keeps only items whose base unit
        is that unit, `size` only items of that size."""
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
            name = self.item_label(r)
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

    def item_label(self, row):
        """One name per item: lines of the same code share the first name seen
        for that code (§8), so spelling variants are not ranked separately."""
        labels = self.__dict__.get("_labels")
        if labels is None:
            labels = self.__dict__.setdefault("_labels", {})
            for r in self.lines:
                if r.get("item_code") and r.get("item_name"):
                    labels.setdefault(fold(r["item_code"]), r["item_name"])
        code = row.get("item_code")
        if code:
            return labels.get(fold(code)) or row.get("item_name") or code
        return row.get("item_name")

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
            name = self.item_label(r)
            if name and r.get("order_id"):
                orders[name].add(r["order_id"])
        return {"type": "number", "answer": sum(1 for v in orders.values() if compare(len(v), op, value))}

    def customer_key(self, row):
        return customer_key(row)

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

    def best_staff(self, period, **kw):
        """§3: the staff member with the highest revenue in the period."""
        if not self.orders or not any(o.get("staff") for o in self.orders):
            raise NoData("shop does not record staff")
        totals = defaultdict(Decimal)
        for o in self.order_rows(period, **kw):
            if o.get("staff"):
                totals[o["staff"]] += o.get("revenue") or 0
        if not totals:
            raise Unsupported("no staff sales in period")
        return {"type": "string", "answer": min(totals, key=lambda s: (-totals[s], str(s)))}

    # ------------------------------------------------------- stock, profit
    def stock_table(self, period):
        """§5 closing stock per item key (code, else name) in base units at the
        end of `period.end`. Only the stock report is trusted, and it states
        the stock of the last day of data; rebuilding stock from purchases,
        sales and returns is left to another path."""
        shop = self.shop
        if not shop.stock and not shop.purchases:
            raise NoData("shop does not record stock or purchases")
        if not shop.stock or self.last_day is None or period.end != self.last_day:
            raise Unsupported("no stock report for that day")
        return dict(shop.stock)

    def stock_of(self, period, item):
        table = self.stock_table(period)
        name, codes = self.item_keys(item)
        hits = [table[k] for k in codes | {name} if k in table]
        if not hits:
            raise Unsupported("item has no stock row")
        return hits[0]

    def stock_qty(self, period, item, big_unit=None):
        """Stock of an item in base units, or [whole big units, loose base units] (§5)."""
        qty = self.stock_of(period, item)
        if big_unit is None:
            return {"type": "number", "answer": number(qty)}
        name, codes = self.item_keys(item)
        factor = None
        for k in codes | {name}:
            hit = self.shop.conversions.get((k, fold(big_unit)))
            if hit:
                factor = hit[0]
        if not factor or factor <= 1:
            raise Unsupported("no conversion for that unit")
        whole = int(qty // factor)
        return {"type": "list", "answer": [whole, number(qty - whole * factor)]}

    def stock_value(self, period):
        """§5: stock of each item times the catalog cost of one base unit, items in stock only."""
        table = self.stock_table(period)
        products = self.shop.products or {}
        total, seen = Decimal(0), set()
        for k, qty in table.items():
            prod = products.get(k)
            ident = prod and fold(prod.get("item_code") or prod.get("item_name") or "")
            if qty <= 0 or prod is None or ident in seen:
                continue
            if prod.get("unit_cost") is None:
                raise Unsupported("catalog has no cost for an item in stock")
            seen.add(ident)
            total += qty * prod["unit_cost"]
        if not seen:
            raise Unsupported("no catalog cost")
        return {"type": "number", "answer": money(total)}

    def unsold_count(self, period):
        """§5: items sold before the period (in counted orders) with no counted sale in it."""
        if period.start is None:
            raise Unsupported("unsold items need a period")
        before, during = set(), set()
        for r in self.line_rows(Period_all(period.end), kinds=("item",)):
            label = self.item_label(r)
            if label is None:
                continue
            (during if r["day"] >= period.start else before).add(fold(label))
        return {"type": "number", "answer": len(before - during)}

    def gross_profit_value(self, period, **kw):
        """§5: revenue minus quantity x the cost written on each sale line."""
        rows = self.line_rows(period, **kw)
        if not any(r.get("unit_cost") is not None for r in self.lines[:5000]):
            raise NoData("sale lines carry no cost")
        total = Decimal(0)
        for r in rows:
            total += r.get("line_amount") or 0
            if r.get("kind") == "item":
                if r.get("unit_cost") is None or r.get("qty") is None:
                    raise Unsupported("sale line without cost")
                total -= r["qty"] * r["unit_cost"]
        return total

    def gross_profit(self, period, **kw):
        return {"type": "number", "answer": money(self.gross_profit_value(period, **kw))}

    def most_profitable_item(self, period, **kw):
        if not any(r.get("unit_cost") is not None for r in self.lines[:5000]):
            raise NoData("sale lines carry no cost")
        names, _ = self.item_ranking(period, "profit", **kw)
        return {"type": "string", "answer": names[0]}

    # --------------------------------------------------------------- debt
    def debt_book(self):
        """§4: per order, what the customer owes and when it was paid.
        Cancelled orders, COD orders and marketplace orders are never debts."""
        book = self.__dict__.get("_debt_book")
        if book is not None:
            return book
        if not self.shop.payments:
            headers = [fold(h) for spec in self.shop.readme.files.values() for h in spec.columns]
            known = any(o.get("paid") is not None or o.get("debt") is not None for o in self.orders)
            if known or any(re.search(r"\bno\b|chua tra|da tra|da thu|con thieu", h) for h in headers):
                raise Unsupported("debts are not kept as a payment list")
            raise NoData("shop does not record payments or debts")
        self.need_orders()
        paid = defaultdict(list)
        for p in self.shop.payments:
            if p.get("day") is not None:
                paid[p["order_id"]].append((p["day"], p["amount"]))
        book = []
        for o in self.orders:
            if o["status"] == "cancelled" or o.get("day") is None:
                continue
            kind = payment_kind(o.get("payment_method"))
            if kind == "cod" or MARKETPLACE.search(fold(o.get("channel") or "")):
                continue
            total = o.get("order_total")
            if total is None:
                raise Unsupported("order without a total to pay")
            book.append({"order": o, "total": total, "payments": sorted(paid.get(o["order_id"], [])), "on_credit": kind == "debt"})
        self.__dict__["_debt_book"] = book
        return book

    def open_debts(self, asof):
        """(order, amount still owed at the end of day `asof`) for orders owing > 0."""
        out = []
        for e in self.debt_book():
            day = e["order"]["day"]
            if day > asof:
                continue
            left = e["total"] - sum((a for d, a in e["payments"] if d <= asof), Decimal(0))
            if left > 0:
                out.append((e["order"], left))
        return out

    def debt_total(self, period):
        return {"type": "number", "answer": money(sum((left for _, left in self.open_debts(period.end)), Decimal(0)))}

    def debt_by_customer(self, asof):
        totals, names, codes = defaultdict(Decimal), {}, {}
        for o, left in self.open_debts(asof):
            k = self.customer_key(o)
            if not k:
                continue
            totals[k] += left
            names.setdefault(k, o.get("customer_name") or o.get("customer_id"))
            codes.setdefault(k, o.get("customer_id") or o.get("customer_name"))
        return totals, names, codes

    def debt_customers(self, period):
        totals, _, _ = self.debt_by_customer(period.end)
        return {"type": "number", "answer": len(totals)}

    def debt_top(self, period, k):
        totals, names, codes = self.debt_by_customer(period.end)
        ranked = sorted(totals, key=lambda c: (-totals[c], str(codes[c])))[:k]
        return {
            "type": "table",
            "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [[names[c], money(totals[c])] for c in ranked]},
        }

    def debt_aging(self, period, buckets):
        """§4: debt age counts days from the order date. `buckets` are
        (low, high) inclusive day ranges, None for an open end."""
        asof, sums = period.end, [Decimal(0)] * len(buckets)
        for o, left in self.open_debts(asof):
            age = (asof - o["day"]).days
            for i, (lo, hi) in enumerate(buckets):
                if (lo is None or age >= lo) and (hi is None or age <= hi):
                    sums[i] += left
        values = [money(v) for v in sums]
        return {"type": "number", "answer": values[0]} if len(values) == 1 else {"type": "list", "answer": values}

    def customer_keys(self, name):
        """Customer keys whose name is `name` (a phone number beside the name is ignored)."""
        target = person_name(name)
        return {
            self.customer_key(o)
            for o in self.orders
            if o.get("customer_name") and person_name(o["customer_name"]) == target and self.customer_key(o)
        }

    def debt_flows(self, period, customer=None):
        """§4: (new debt, debt repaid) in the period, over orders sold on
        credit. New debt is what an order placed in the period left unpaid on
        its own day; repayments are payments dated after the order day."""
        keys = None
        if customer is not None:
            keys = self.customer_keys(customer)
            if len(keys) != 1:
                raise Unsupported("customer name is not unique")
        added = repaid = Decimal(0)
        for e in self.debt_book():
            o = e["order"]
            if not e["on_credit"] or (keys is not None and self.customer_key(o) not in keys):
                continue
            if period.contains(o["day"]):
                unpaid = e["total"] - sum((a for d, a in e["payments"] if d <= o["day"]), Decimal(0))
                added += max(unpaid, Decimal(0))
            repaid += sum((a for d, a in e["payments"] if d > o["day"] and period.contains(d)), Decimal(0))
        return added, repaid

    def debt_collected(self, period):
        return {"type": "number", "answer": money(self.debt_flows(period)[1])}

    def debt_change(self, period, customer):
        added, repaid = self.debt_flows(period, customer)
        return {"type": "list", "answer": [money(added), money(repaid)]}

    # ---------------------------------------------------------- purchases
    def purchase_rows(self, period, item=None):
        """§5: purchase lines of slips dated in the period, cancelled slips left out."""
        if not self.shop.purchases:
            raise NoData("shop does not record purchases")
        name, codes = self.item_keys(item) if item is not None else (None, set())
        out = []
        for r in self.shop.purchases:
            if r["cancelled"] or r.get("day") is None or not period.contains(r["day"]):
                continue
            if item is not None and not (
                fold(r.get("item_name") or "") == name or (r.get("item_code") and fold(r["item_code"]) in codes)
            ):
                continue
            out.append(r)
        return out

    def purchase_slips(self, period):
        if not self.shop.purchase_slips:
            raise NoData("shop does not record purchases")
        return [
            s
            for s in self.shop.purchase_slips.values()
            if not s["cancelled"] and s.get("day") is not None and period.contains(s["day"])
        ]

    def purchase_total(self, period):
        total = sum((s["purchase_total"] for s in self.purchase_slips(period) if s.get("purchase_total") is not None), Decimal(0))
        return {"type": "number", "answer": money(total)}

    def purchase_avg_price(self, period, item):
        """§5: average purchase price weighted by purchased quantity, per purchase unit."""
        rows = [r for r in self.purchase_rows(period, item) if r.get("qty") and r.get("price") is not None]
        if not any(r.get("price") is not None for r in self.shop.purchases):
            raise NoData("shop does not record purchase prices")
        if not rows:
            raise Unsupported("item not purchased in period")
        if len({fold(r.get("unit") or "") for r in rows}) > 1:
            raise Unsupported("item purchased in several units")
        qty = sum((r["qty"] for r in rows), Decimal(0))
        return {"type": "number", "answer": money(sum((r["qty"] * r["price"] for r in rows), Decimal(0)) / qty)}

    def top_supplier(self, period):
        totals = defaultdict(Decimal)
        for s in self.purchase_slips(period):
            if s.get("supplier") and s.get("purchase_total") is not None:
                totals[s["supplier"]] += s["purchase_total"]
        if not totals:
            raise Unsupported("no supplier purchases in period")
        return {"type": "string", "answer": min(totals, key=lambda s: (-totals[s], str(s)))}

    def purchase_minus_sales(self, period, item):
        """§5: purchased minus sold quantity of an item, both in base units."""
        rows = self.purchase_rows(period, item)
        if any(r.get("base_qty") is None for r in rows):
            raise Unsupported("purchase line without quantity")
        bought = sum((r["base_qty"] for r in rows), Decimal(0))
        return {"type": "number", "answer": number(bought - self.qty_value(period, item=item))}

    def top_items(self, period, k, measure="qty", **kw):
        names, _ = self.item_ranking(period, measure, **kw)
        return {"type": "list", "answer": names[:k]}

    def best_item(self, period, measure="qty", **kw):
        names, _ = self.item_ranking(period, measure, **kw)
        return {"type": "string", "answer": names[0]}


class Period_all:
    """Everything up to and including `end` (duck-typed Period)."""

    start = None

    def __init__(self, end):
        self.end = end

    def contains(self, day):
        return day <= self.end


def customer_key(row):
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


def person_name(name):
    """Folded customer name without a phone number written beside it (§9)."""
    text = fold(name)
    text = re.sub(r"\s*[-–(]?\s*\+?\d[\d .]{7,}\)?\s*$", "", text)
    return re.sub(r"^\+?\d[\d .]{7,}\s*[-–]?\s*", "", text).strip(" -–")


MARKETPLACE = re.compile(r"shopee|lazada|tiki|tiktok|sendo|tmdt|\bsan\b|marketplace|ecommerce")


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
