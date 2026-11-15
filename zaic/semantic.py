"""Semantic layer: turn one business's files into clean canonical tables.

`Shop(split_dir, business_id)` reads the README (zaic.readme), finds the files of
each README section, parses cells (zaic.values) and applies the data rules of
the rulebook (§1, §8, §10) once, deterministically:

- header row found by matching README headers; repeated header rows, total and
  note rows dropped;
- grouped rows filled down (§8), identical rows of one order counted once;
- dates / money / quantities typed, statuses mapped to done / cancelled /
  returned / shipping, base-unit quantities derived from conversion tables;
- orders, products and customers joined by code.

Result: pandas frames `lines` (one row per sold line) and `orders` (one row per
order), plus `products`, `customers`, `payments` when the shop has them, and a
list of `issues` explaining anything that could not be built. Nothing here is
keyed on a business id; everything follows the README vocabulary.
"""

import datetime as dt
import re
from collections import Counter
from decimal import Decimal

from .readme import (
    CONTEXT_ROLES,
    FileSpec,
    column_role,
    header_role,
    note_descriptions,
    parse_readme,
)
from .tools import Tools, cell_text, detect_header
from .values import (
    fold,
    month_first_column,
    parse_datetime,
    parse_number,
    parse_qty,
    parse_time,
    separator_style,
    to_bool,
)

ORDER_LEVEL = [
    "order_id",
    "order_seq",
    "date",
    "datetime",
    "time",
    "status",
    "note",
    "customer_id",
    "customer_name",
    "customer_group",
    "phone",
    "payment_method",
    "channel",
    "staff",
    "order_total",
    "goods_total",
    "order_discount",
    "shipping",
    "paid",
    "debt",
    "due_date",
    "due_days",
]
MONEY_ROLES = {
    "line_amount",
    "unit_price",
    "unit_cost",
    "line_discount",
    "discount",
    "order_total",
    "goods_total",
    "order_discount",
    "shipping",
    "paid",
    "debt",
    "pay_amount",
    "current_price",
    "new_price",
    "old_price",
    "purchase_price",
    "purchase_line_amount",
    "purchase_total",
}
NUMBER_ROLES = {"base_qty", "return_qty", "factor", "stock_qty", "purchase_base_qty", "batch_qty", "due_days"}
DATE_ROLES = {"date", "datetime", "purchase_date", "price_date", "expiry", "due_date", "delivery_date", "pay_time"}
SUMMARY_ROW = re.compile(r"^(tong|cong|ghi chu|luu y|note|total)\b")
YEAR = re.compile(r"(?<!\d)(20\d\d)(?!\d)")

# Order matters: "hoàn thành" (done) must win over the "hoàn" of "hoàn trả".
STATUS_RULES = [
    ("cancelled", r"\bhuy\b|cancel|\bvoid\b|\bbo don\b|khong giao duoc|giao (?:khong )?that bai|\bfail"),
    ("done", r"hoan thanh|thanh cong|\bxong\b|\bok\b|done|complete|\bda lay\b|\bda giao\b|\bda nhan\b|da thanh toan|\bpaid\b|success|\bfinish"),
    ("returned", r"\btra\b|\bhoan\b|return|refund|\bdoi\b"),
    ("shipping", r"dang giao|dang ship|shipping|deliver|cho giao|van chuyen|chua giao|giao dich|dang xu ly|pending|processing"),
]


def normalize_status(value):
    if value is None:
        return None
    text = fold(value)
    for status, rx in STATUS_RULES:
        if re.search(rx, text):
            return status
    return None


def fix_mojibake(name):
    try:
        fixed = name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    return fixed


def key(text):
    return fold(text) if text is not None else None


def camel_words(name):
    """'NgayBan' -> 'ngay ban', 'TongThanhToan' -> 'tong thanh toan'."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name)).replace("_", " ")
    return fold(spaced)


class Shop:
    def __init__(self, split_dir, business_id):
        self.tools = Tools(split_dir, business_id)
        self.business_id = business_id
        self.readme = parse_readme(self.tools.resolve("README.md").read_text(encoding="utf-8"))
        self.issues = []
        self.frames = {}  # logical file -> list of row dicts keyed by role
        self.files = [f["path"] for f in self.tools.list_files()]
        self.lines = None
        self.orders = None
        self.products = None
        self.customers = None
        self.payments = None
        self.conversions = {}
        self.data_end = self.readme.data_end
        # Toppings written in the same cell as the drink ("Trà sữa + 2 Trân châu")
        # are separate items in the rulebook; item-level metrics are not reliable
        # until such cells are split, so the metrics decline them.
        self.combined_items = any(
            re.search(r"topping", fold(n)) and re.search(r"chung o|cung o|noi bang", fold(n)) for n in self.readme.notes
        )
        try:
            self._build()
        finally:
            self.tools.close()

    # ------------------------------------------------------------------ files
    def physical_files(self, logical):
        stem, _, ext = logical.rpartition(".")
        target = fold(stem)
        out = []
        if "<" in target:  # README file-name template, e.g. BanHang_T<tháng>_<năm>.xlsx
            rx = re.compile("^" + re.sub(r"\\<[^>]*\\>|<[^>]*>", r"[a-z0-9]+", re.escape(target).replace("\\<", "<").replace("\\>", ">")) + "$")
            for path in self.files:
                if path.lower().endswith("." + ext.lower()) and rx.match(fold(fix_mojibake(path.rpartition(".")[0]))):
                    out.append(path)
            return sorted(out)
        for path in self.files:
            if not path.lower().endswith("." + ext.lower()):
                continue
            pstem = fold(fix_mojibake(path.rpartition(".")[0]))
            if pstem == target or (
                pstem.startswith(target + "_")
                and re.fullmatch(r"(?:_[a-z]*\d+)+", pstem[len(target) :])
            ):
                out.append(path)
        return sorted(out)

    def read_spec(self, spec):
        """Rows of every physical file/sheet of one README section, as dicts
        role -> cell text, plus _file/_sheet/_row/_year."""
        paths = self.physical_files(spec.path)
        if not paths:
            self.issues.append(f"no file for {spec.path}")
            return []
        headers = {key(h): h for h in spec.columns}
        records = []
        for path in paths:
            sheets = self.sheets_of(path)
            for sheet, rows in sheets.items():
                if not rows:
                    continue
                if spec.sheet is not None and fold(sheet or "") != fold(spec.sheet):
                    continue
                best, best_hits = None, 0
                for i, row in enumerate(rows[:30]):
                    hits = sum(1 for v in row if v is not None and key(v) in headers)
                    if hits > best_hits:
                        best, best_hits = i, hits
                if best is None or best_hits < max(1, len(headers) // 2):
                    self.issues.append(f"header not found in {path}:{sheet}")
                    best = detect_header(rows)
                header = rows[best]
                cols = []
                for v in header:
                    h = headers.get(key(v)) if v is not None else None
                    cols.append(h)
                year = None
                for text in (sheet or "", path):
                    m = YEAR.search(text)
                    if m:
                        year = int(m.group(1))
                        break
                header_keys = [key(v) for v in header]
                for i, row in enumerate(rows[best + 1 :], best + 1):
                    if not any(v is not None for v in row):
                        continue
                    if [key(v) for v in row[: len(header_keys)]] == header_keys:
                        continue  # repeated header
                    rec = {"_file": path, "_sheet": sheet, "_row": i, "_year": year}
                    for j, h in enumerate(cols):
                        if h is not None and j < len(row):
                            rec[h] = row[j]
                    records.append(rec)
        return records

    def sheets_of(self, path):
        """{sheet or table: rows} for Excel/CSV (shared reader) and SQLite."""
        if path.lower().endswith(".sqlite"):
            import sqlite3
            from contextlib import closing

            p = self.tools.resolve(path)
            out = {}
            with closing(sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)) as conn:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                for t in tables:
                    cur = conn.execute(f'SELECT * FROM "{t}"')
                    rows = [[c[0] for c in cur.description]]
                    rows += [[cell_text(v) for v in row] for row in cur.fetchall()]
                    out[t] = rows
            return out
        _, sheets = self.tools.read_rows(self.tools.resolve(path))
        return sheets

    def synthesize_specs(self):
        """Specs for files/sheets/tables the README does not describe, with roles
        from header names (and descriptions in the README conventions)."""
        covered = set()
        for spec in self.readme.files.values():
            if not spec.columns:
                continue
            for path in self.physical_files(spec.path):
                covered.add((path, fold(spec.sheet) if spec.sheet else "*"))
        notes = note_descriptions(self.readme.notes)
        groups = {}
        for path in self.files:
            if not path.lower().endswith((".xlsx", ".csv", ".sqlite")) or (path, "*") in covered:
                continue
            try:
                sheets = self.sheets_of(path)
            except Exception as exc:  # noqa: BLE001
                self.issues.append(f"cannot read {path}: {exc}")
                continue
            for sheet, rows in sheets.items():
                if not rows or (path, fold(sheet or "")) in covered:
                    continue
                h = detect_header(rows)
                header = [v for v in rows[h] if v is not None]
                if len(header) < 2:
                    continue
                stem, _, ext = fix_mojibake(path).rpartition(".")
                stem = re.sub(r"(?:_[A-Za-z]*\d+)+$", "", stem)
                multi = len(sheets) > 1 and not path.lower().endswith(".csv")
                group_key = (stem, ext, tuple(key(x) for x in header), sheet if multi else None)
                groups.setdefault(group_key, (f"{stem}.{ext}", sheet if multi else None, header))
        for logical, sheet, header in groups.values():
            spec = FileSpec(logical, sheet)
            for h in header:
                role = header_role(h)
                desc = notes.get(h)
                if role is None and desc:
                    role = column_role(desc)
                spec.columns[h] = desc or h
                spec.roles[h] = role
            spec.synthetic = True
            self.readme.files[f"{logical}#{sheet}" if sheet else logical] = spec

    # ------------------------------------------------------------------ build
    def classify(self):
        kinds = {}
        for spec in self.readme.files.values():
            if not spec.columns:
                continue
            r = set(spec.roles.values())
            item = bool(r & {"item_name", "item_code"})
            if r & {"purchase_id", "purchase_date", "purchase_price", "purchase_qty", "purchase_line_amount"}:
                kind = "purchases"
            elif "batch_qty" in r or ("batch_id" in r and "order_id" in r):
                kind = "batch_sales"
            elif "batch_id" in r:
                kind = "batches"
            elif "factor" in r:
                kind = "conversions"
            elif "price_date" in r or "new_price" in r:
                kind = "prices"
            elif "payment_id" in r or ("pay_amount" in r and "order_id" in r):
                kind = "payments"
            elif item and ("order_id" in r or "date" in r) and r & {"line_amount", "qty", "unit_price"}:
                kind = "lines"
            elif "order_id" in r and r & {"date", "datetime", "status", "order_total", "goods_total"}:
                kind = "orders"
            elif "stock_qty" in r:
                kind = "stock"
            elif "rx" in r or "ingredient" in r:
                kind = "drugs"
            elif "customer_id" in r and not r & {"item_code", "unit_price", "current_price", "unit_cost", "group"}:
                kind = "customers"
            elif item or ("name" in r and r & {"group", "unit_base", "current_price", "unit_cost"}):
                kind = "products"
            else:
                kind = "other"
            # Roles whose meaning depends on the file kind ("Trạng thái" of a
            # purchase slip, "Thời gian" of a payment, "Giảm giá" of an order).
            for header, role in list(spec.roles.items()):
                ctx = CONTEXT_ROLES.get(kind, {})
                if role in ctx:
                    spec.roles[header] = ctx[role]
            kinds.setdefault(kind, []).append(spec)
        return kinds

    def _build(self):
        self.synthesize_specs()
        kinds = self.classify()
        self.kinds = {k: [s.path for s in v] for k, v in kinds.items()}
        self._build_conversions(kinds.get("conversions", []))
        self.products = self._build_products(kinds.get("products", []))
        self.customers = self._build_customers(kinds.get("customers", []))
        self.drugs = self._build_drugs(kinds.get("drugs", []))
        orders_specs = kinds.get("orders", [])
        lines_specs = kinds.get("lines", [])
        if not lines_specs and not orders_specs:
            self.issues.append("no sales file recognized")
            return
        order_rows = []
        for spec in orders_specs:
            order_rows += self.typed(self.read_spec(spec), spec)
        line_rows = []
        for spec in lines_specs:
            line_rows += self.typed(self.read_spec(spec), spec)
        self._assemble(line_rows, order_rows)

    def typed(self, records, spec):
        """Rename columns to roles and parse values."""
        roles = spec.roles
        by_role = {}
        for header, role in roles.items():
            if role and role not in by_role:
                by_role[role] = header
        seq_orders = "order_seq" in by_role and "order_id" not in by_role
        if seq_orders:
            by_role["order_id"] = by_role.pop("order_seq")
        out = []
        styles = {}
        month_first = {}
        for role, header in by_role.items():
            values = [r.get(header) for r in records[:20000]]
            if role in MONEY_ROLES or role in NUMBER_ROLES:
                styles[role] = separator_style(values)
            if role in DATE_ROLES:
                month_first[role] = month_first_column(values)
        anchors = [by_role[k] for k in ("order_id", "date", "datetime", "purchase_id", "purchase_date") if k in by_role]
        dropped = 0
        for r in records:
            if anchors and all(r.get(h) is None for h in anchors) and any(
                isinstance(v, str) and SUMMARY_ROW.match(fold(v)) for k, v in r.items() if not k.startswith("_")
            ):
                dropped += 1  # total / note row, e.g. "Tổng tháng 4/2026", "Ghi chú: ..."
                continue
            rec = {"_file": r["_file"], "_sheet": r["_sheet"], "_row": r["_row"], "_year": r["_year"]}
            raw = {}
            for role, header in by_role.items():
                v = r.get(header)
                raw[role] = v
                rec[role] = v
            rec["_raw"] = tuple(r.get(h) for h in spec.columns)
            if seq_orders and rec.get("order_id") is not None:
                # Order numbers in a hand-kept book may restart on each sheet.
                rec["order_id"] = f"{rec['_file']}#{rec['_sheet']}#{rec['order_id']}"
            out.append(rec)
        if dropped:
            self.issues.append(f"dropped {dropped} total/note rows in {spec.path}")
        self._styles = getattr(self, "_styles", {})
        self._styles[spec.path] = (styles, month_first)
        for rec in out:
            for role in list(rec):
                if role.startswith("_"):
                    continue
                v = rec[role]
                if v is None:
                    continue
                if role in MONEY_ROLES or role in NUMBER_ROLES:
                    rec[role] = parse_number(v, styles.get(role))
                elif role in DATE_ROLES:
                    rec[role] = parse_datetime(v, rec["_year"], month_first.get(role, False))
                elif role == "time":
                    rec[role] = parse_time(v)
                elif role == "qty":
                    number, unit = parse_qty(v)
                    rec["qty"] = number
                    if unit and not rec.get("unit"):
                        rec["unit"] = unit
                    if unit:
                        rec["_qty_unit"] = unit
        return out

    def _build_conversions(self, specs):
        conv = {}
        for item, big, factor, base in self.readme.conversions:
            conv[(key(item), key(big))] = (Decimal(str(factor)), base)
        for spec in specs:
            rows = self.typed(self.read_spec(spec), spec)
            for r in rows:
                name = r.get("item_name") or r.get("name") or r.get("item_code")
                big = r.get("unit_big")
                if name and big and r.get("factor") is not None:
                    conv[(key(name), key(big))] = (r["factor"], r.get("unit_base"))
                    if r.get("item_code"):
                        conv[(key(r["item_code"]), key(big))] = (r["factor"], r.get("unit_base"))
        self.conversions = conv

    def _build_products(self, specs):
        products = {}
        for spec in specs:
            for r in self.typed(self.read_spec(spec), spec):
                name = r.get("item_name") or r.get("name")
                code = r.get("item_code")
                if not (name or code):
                    continue
                entry = {
                    "item_code": code,
                    "item_name": name,
                    "group": r.get("group"),
                    "unit_base": r.get("unit_base") or r.get("unit"),
                    "unit_cost": r.get("unit_cost"),
                    "current_price": r.get("current_price"),
                }
                for k in filter(None, (key(code), key(name))):
                    products.setdefault(k, {}).update({a: b for a, b in entry.items() if b is not None})
        return products

    def _build_drugs(self, specs):
        """Drug catalog: code/name -> {rx, ingredient} (§6 nhà thuốc)."""
        drugs = {}
        for spec in specs:
            for r in self.typed(self.read_spec(spec), spec):
                entry = {"rx": to_bool(r.get("rx")), "ingredient": r.get("ingredient")}
                for k in filter(None, (key(r.get("item_code")), key(r.get("item_name") or r.get("name")))):
                    drugs[k] = entry
        return drugs

    def _build_customers(self, specs):
        customers = {}
        for spec in specs:
            for r in self.typed(self.read_spec(spec), spec):
                cid = r.get("customer_id")
                if cid:
                    customers[key(cid)] = {
                        "customer_id": cid,
                        "customer_name": r.get("customer_name") or r.get("name"),
                        "customer_group": r.get("customer_group"),
                        "phone": r.get("phone"),
                    }
        return customers

    # --------------------------------------------------------------- assemble
    def _fill_down(self, rows):
        """Rulebook §8: blank order-level cells in later rows of a group repeat the
        group's first row. Groups are runs of rows in source order; a new group
        starts where the order id (or, without ids, the date) is present."""
        if not rows:
            return rows
        has_id = any(r.get("order_id") for r in rows)
        anchor = "order_id" if has_id else ("date" if any(r.get("date") for r in rows) else None)
        if anchor is None:
            return rows
        grouped = any(r.get(anchor) is None for r in rows)
        rows.sort(key=lambda r: (r["_file"], str(r["_sheet"]), r["_row"]))
        if grouped:
            current = {}
            prev_file = None
            for r in rows:
                if (r["_file"], r["_sheet"]) != prev_file:
                    current, prev_file = {}, (r["_file"], r["_sheet"])
                if r.get(anchor) is not None:
                    current = {k: r.get(k) for k in ORDER_LEVEL if r.get(k) is not None}
                    if anchor == "date":
                        current = {"date": r["date"]}
                else:
                    for k, v in current.items():
                        if r.get(k) is None:
                            r[k] = v
        if has_id:
            first = {}
            for r in rows:
                oid = r.get("order_id")
                if oid is None:
                    continue
                seen = first.setdefault(oid, {})
                for k in ORDER_LEVEL:
                    if r.get(k) is not None:
                        seen.setdefault(k, r[k])
            for r in rows:
                oid = r.get("order_id")
                if oid is None:
                    continue
                for k, v in first[oid].items():
                    if r.get(k) is None:
                        r[k] = v
        return rows

    def _line_kind(self, name):
        text = fold(name or "")
        if re.match(r"phi phuc vu", text):
            return "service"
        if re.match(r"(phi ship|phi giao|ship\b|van chuyen)", text):
            return "shipping"
        if re.match(r"voucher|ma giam|giam gia", text):
            return "voucher"
        if re.match(r"dung \d+ diem|dung diem|doi diem", text):
            return "points"
        return "item"

    def _factor(self, code, name, unit):
        if unit is None:
            return Decimal(1)
        u = key(unit)
        for k in filter(None, (key(code), key(name))):
            hit = self.conversions.get((k, u))
            if hit:
                return hit[0]
            prod = self.products.get(k) if self.products else None
            if prod and prod.get("unit_base") and key(prod["unit_base"]) == u:
                return Decimal(1)
        m = re.match(r"^(.*?)\s+(\d+(?:[.,]\d+)?)\s*([a-zà-ỹ]*)$", str(unit))
        if m:  # "bao 25kg", "hộp 30": the number in the unit name
            for k in filter(None, (key(code), key(name))):
                for (ck, cu), (factor, _) in self.conversions.items():
                    if ck == k and cu.startswith(key(m.group(1))) and str(m.group(2)) in cu:
                        return factor
        return Decimal(1)

    def _assemble(self, line_rows, order_rows):
        line_rows = self._fill_down(line_rows)
        order_rows = self._fill_down(order_rows)
        # §8: identical rows of the same order are one export error.
        if any(r.get("order_id") for r in line_rows):
            seen, kept = set(), []
            for r in line_rows:
                sig = (r.get("order_id"), r["_raw"])
                if r.get("order_id") is not None and sig in seen:
                    continue
                seen.add(sig)
                kept.append(r)
            if len(kept) != len(line_rows):
                self.issues.append(f"dropped {len(line_rows) - len(kept)} duplicate order lines")
            line_rows = kept
        orders = {}
        for r in order_rows:
            oid = r.get("order_id")
            if oid is None:
                continue
            o = orders.setdefault(oid, {"order_id": oid})
            for k in ORDER_LEVEL:
                if r.get(k) is not None and o.get(k) is None:
                    o[k] = r[k]
        # Lines get order attributes from the order file.
        for r in line_rows:
            o = orders.get(r.get("order_id"))
            if o:
                for k, v in o.items():
                    if r.get(k) is None:
                        r[k] = v
        # Products and customers by code.
        for r in line_rows:
            prod = None
            for k in filter(None, (key(r.get("item_code")), key(r.get("item_name")))):
                prod = (self.products or {}).get(k)
                if prod:
                    break
            if prod:
                for a in ("item_name", "item_code", "group", "unit_base"):
                    if r.get(a) is None and prod.get(a) is not None:
                        r[a] = prod[a]
                if r.get("catalog_cost") is None:
                    r["catalog_cost"] = prod.get("unit_cost")
            if self.drugs:
                drug = None
                for k in filter(None, (key(r.get("item_code")), key(r.get("item_name")))):
                    drug = self.drugs.get(k)
                    if drug:
                        break
                # §6: everything not marked prescription in the catalog is OTC.
                r["rx"] = bool(drug and drug.get("rx"))
                r["ingredient"] = drug.get("ingredient") if drug else None
            cust = (self.customers or {}).get(key(r.get("customer_id")))
            if cust:
                for a in ("customer_name", "customer_group", "phone"):
                    if r.get(a) is None and cust.get(a) is not None:
                        r[a] = cust[a]
        for o in orders.values():
            cust = (self.customers or {}).get(key(o.get("customer_id")))
            if cust:
                for a in ("customer_name", "customer_group"):
                    if o.get(a) is None and cust.get(a) is not None:
                        o[a] = cust[a]
        # Line values.
        for r in line_rows:
            r["kind"] = self._line_kind(r.get("item_name"))
            if r.get("line_amount") is None and r.get("qty") is not None and r.get("unit_price") is not None:
                r["line_amount"] = r["qty"] * r["unit_price"] - (r.get("line_discount") or 0)
            if r.get("base_qty") is None and r.get("qty") is not None:
                r["base_qty"] = r["qty"] * self._factor(r.get("item_code"), r.get("item_name"), r.get("unit"))
            day_value = r.get("date") or r.get("datetime")
            r["day"] = day_value.date() if isinstance(day_value, dt.datetime) else day_value
            ts = r.get("datetime")
            if r.get("time") is None and isinstance(ts, dt.datetime) and (ts.hour or ts.minute):
                r["time"] = ts.time()
            if r.get("time") is None and isinstance(r.get("date"), dt.datetime):
                d = r["date"]
                if d.hour or d.minute or d.second:
                    r["time"] = d.time()
        self.lines = line_rows
        self._build_orders(line_rows, orders)

    def _build_orders(self, line_rows, orders):
        """One record per order with rulebook status and revenue (§1)."""
        by_order = {}
        no_id = not any(r.get("order_id") for r in line_rows) and line_rows
        if no_id:
            self.issues.append("lines have no order id: order counts are not available")
        for r in line_rows:
            oid = r.get("order_id")
            if oid is None:
                continue
            by_order.setdefault(oid, []).append(r)
        records = []
        all_ids = list(dict.fromkeys(list(orders) + list(by_order)))
        for oid in all_ids:
            head = orders.get(oid, {})
            lines = by_order.get(oid, [])
            statuses = Counter()
            for src in [head] + lines:
                s = normalize_status(src.get("status"))
                if s:
                    statuses[s] += 1
                note = normalize_status(src.get("note"))
                if note in ("cancelled", "returned", "shipping"):
                    statuses[note] += 1
            returned_line = any(
                (r.get("qty") is not None and r["qty"] < 0)
                or (r.get("return_qty") is not None and r["return_qty"] > 0)
                for r in lines
            )
            if statuses.get("cancelled"):
                status = "cancelled"
            elif statuses.get("returned") or returned_line:
                status = "returned"
            elif statuses.get("shipping"):
                status = "shipping"
            else:
                status = "done"
            first = head if head else (lines[0] if lines else {})
            day_value = first.get("date") or first.get("datetime")
            if day_value is None:
                for r in lines:
                    day_value = r.get("date") or r.get("datetime")
                    if day_value:
                        break
            day = day_value.date() if isinstance(day_value, dt.datetime) else day_value
            if lines:
                revenue = sum(
                    (r["line_amount"] for r in lines if r["kind"] in ("item", "service") and r.get("line_amount") is not None),
                    Decimal(0),
                )
                if all(r.get("line_amount") is None for r in lines) and head.get("goods_total") is not None:
                    revenue = head["goods_total"]
            else:
                revenue = head.get("goods_total")
                if revenue is None and head.get("order_total") is not None:
                    revenue = head["order_total"] - (head.get("shipping") or 0) + (head.get("order_discount") or 0)
            time = first.get("time")
            ts = first.get("datetime")
            if time is None and isinstance(ts, dt.datetime):
                time = ts.time()
            rec = {
                "order_id": oid,
                "day": day,
                "time": time,
                "status": status,
                "raw_status": first.get("status"),
                "revenue": revenue,
                "n_lines": len(lines),
            }
            for k in ("customer_id", "customer_name", "customer_group", "payment_method", "channel", "staff",
                      "order_total", "goods_total", "order_discount", "shipping", "paid", "debt", "note"):
                rec[k] = first.get(k)
                if rec[k] is None:
                    for r in lines:
                        if r.get(k) is not None:
                            rec[k] = r[k]
                            break
            records.append(rec)
        self.orders = records
        status_of = {o["order_id"]: o["status"] for o in records}
        for r in line_rows:
            r["status"] = status_of.get(r.get("order_id"), normalize_status(r.get("status")) or "done")

    # ---------------------------------------------------------------- summary
    def date_range(self):
        days = [r["day"] for r in (self.lines or []) if r.get("day")]
        days += [o["day"] for o in (self.orders or []) if o.get("day")]
        if not days:
            return None, None
        return min(days), max(days)

    def summary(self):
        lo, hi = self.date_range()
        return {
            "kinds": self.kinds,
            "lines": len(self.lines or []),
            "orders": len(self.orders or []),
            "products": len({id(v) for v in (self.products or {}).values()}),
            "customers": len(self.customers or {}),
            "date_range": [str(lo), str(hi)],
            "issues": self.issues,
        }


def sqlite_roles(columns):
    """Role guess for SQLite columns without README descriptions, from the
    column name words ('NgayBan' -> 'ngay ban')."""
    return {c: column_role(camel_words(c)) for c in columns}


__all__ = ["FileSpec", "Shop", "normalize_status", "sqlite_roles"]
