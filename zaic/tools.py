"""Read-only tools. No arbitrary Python execution or external SQL file access.

Excel and CSV files are read by one reader (`read_rows`) so that inspect_table
previews show exactly the text that load_table puts into DuckDB, with the same
zero-based row numbers.
"""

import csv
import datetime as dt
import json
import math
import re
import sqlite3
import threading
import time
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from .formats import TOOL_NAMES, TOOL_PARAMS

TOOL_HELP = """Tools (one call per turn):
list_files {} -> paths relative to current business directory.
read_document {"path":"README.md","offset":0,"limit":12000} -> text; rulebook.md is split rulebook.
inspect_table {"path":"file.xlsx","sheet":"optional sheet name"} -> every sheet with its row count,
  plus the first 6 rows (numbered from 0) of each sheet while space allows; pass "sheet" to see
  one sheet. CSV -> delimiter, row count, first rows. SQLite -> table schemas and row counts.
load_table {"path":"file.xlsx","name":"sales","sheet":"Sheet1","header_row":0} ->
  register a sheet/CSV as DuckDB table "name". header_row is the ZERO-based row index shown by
  inspect_table (-1 = no header). sheet: a name or index (default first sheet), a list of
  sheet names, or "*" for all sheets, stacked into one table (columns matched by header name).
  All cells are text exactly as inspect_table shows them; blank cells are NULL; fully blank
  rows are skipped. Added columns: _row (source row index, same numbering as header_row) and,
  for Excel, _sheet. Nothing is filled down, deduplicated or converted. calendar.csv is also
  available as path. Returns columns, row counts and sample rows.
run_sql {"sql":"SELECT ... FROM sales"} -> DuckDB query on loaded tables.
query_sqlite {"path":"pos.sqlite","sql":"SELECT ..."} -> read-only SQLite query.
Queries return at most 100 rows and have a time limit. Aggregate over full tables, not previews.
DuckDB SQL can join loaded tables. SQL has no external file/network access.
Use quoted column identifiers, normalize date/number formats based on observed data.
Numbers stored as text: REPLACE(text, search, replacement) takes the column FIRST, e.g.
  SUM(CAST(REPLACE("Tổng tiền", ',', '') AS DECIMAL(18,2))). TRY_CAST(x AS ...) gives NULL
  instead of failing on bad cells; check previews for thousand separators and date formats.
Source row order is _row (within _sheet). Example: a group starts where "Mã đơn" is not NULL;
  copy the group's first value to the rows below it with
  WITH g AS (SELECT *, COUNT("Mã đơn") OVER (PARTITION BY _sheet ORDER BY _row) AS grp FROM t)
  SELECT FIRST_VALUE("Mã đơn") OVER (PARTITION BY _sheet, grp ORDER BY _row) AS "Mã đơn", ... FROM g
"""

SOURCE_SUFFIXES = {".md", ".xlsx", ".csv", ".sqlite"}
PREVIEW_ROWS = 6
INSPECT_PREVIEW_CHARS = 6000
RESERVED_COLUMNS = {"_row", "_sheet"}


def serial(value):
    if isinstance(value, Decimal):
        # SQL DECIMAL results become JSON numbers, not strings like "280998600.00".
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=serial, allow_nan=False)


def cell_text(value):
    """One text form per Excel/CSV cell, shared by previews and loaded tables."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, dt.datetime):
        if value.time() == dt.time(0):
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    return str(value)


def trim(row):
    """Drop trailing blank cells (for compact previews)."""
    row = list(row)
    while row and row[-1] is None:
        row.pop()
    return row


class Tools:
    def __init__(self, split_dir, business_id, timeout=30):
        self.split = Path(split_dir).resolve()
        if not re.fullmatch(r"[\w-]+", business_id):
            raise ValueError("Invalid business_id")
        self.root = (self.split / "data" / business_id).resolve()
        if not self.root.is_dir() or not self.root.is_relative_to(self.split / "data"):
            raise ValueError(f"Unknown business: {business_id}")
        self.timeout = timeout
        self.db = None
        self.tables = {}  # load_table name -> column list, for error feedback
        self._rows = {}  # resolved path -> (delimiter or None, {sheet: rows})

    def resolve(self, path):
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if path in ("rulebook.md", "calendar.csv"):
            p = (self.split / path).resolve()
            if p.parent != self.split:
                raise ValueError("Invalid shared file")
        else:
            p = (self.root / path).resolve()
            if not p.is_relative_to(self.root):
                raise ValueError("Path outside current business")
        if not p.is_file() or p.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError(
                f"Unsupported or missing source file {path!r}; use a path from list_files"
            )
        return p

    def call(self, name, args):
        if name not in TOOL_NAMES:
            raise ValueError("Unknown tool")
        if not isinstance(args, dict):
            raise TypeError("arguments must be an object")
        required, optional = TOOL_PARAMS[name]
        unknown = set(args) - set(required) - set(optional)
        missing = [k for k in required if k not in args]
        if unknown or missing:
            raise TypeError(
                f"{name} takes required {list(required)} and optional {list(optional)}; "
                f"got {sorted(args)}"
            )
        return getattr(self, name)(**args)

    def describe_tables(self):
        """Tables loaded so far in this question and their columns."""
        return dict(self.tables)

    def list_files(self):
        return [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.resolve().is_relative_to(self.root)
        ]

    def read_document(self, path, offset=0, limit=12000):
        p = self.resolve(path)
        if (
            p.suffix.lower() != ".md"
            or type(offset) is not int
            or type(limit) is not int
            or offset < 0
            or not 1 <= limit <= 20000
        ):
            raise ValueError("Expected markdown, offset >= 0, limit 1..20000")
        text = p.read_text(encoding="utf-8")
        return {
            "text": text[offset : offset + limit],
            "total_chars": len(text),
            "next_offset": offset + limit if offset + limit < len(text) else None,
        }

    def read_rows(self, p):
        """Return (csv delimiter or None, {sheet name: rows of cell_text values})."""
        if p in self._rows:
            return self._rows[p]
        suffix = p.suffix.lower()
        if suffix == ".xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            try:
                sheets = {
                    ws.title: [[cell_text(v) for v in row] for row in ws.values]
                    for ws in wb
                }
            finally:
                wb.close()
            result = (None, sheets)
        elif suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                sample = f.read(8192)
                try:
                    delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    delim = ","
                f.seek(0)
                rows = [[cell_text(v) for v in row] for row in csv.reader(f, delimiter=delim)]
            result = (delim, {None: rows})
        else:
            raise ValueError("Expected Excel or CSV")
        self._rows[p] = result
        return result

    def inspect_table(self, path, sheet=None):
        p = self.resolve(path)
        suffix = p.suffix.lower()
        if suffix == ".sqlite":
            with closing(sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)) as conn:
                out = []
                for name, kind, sql in conn.execute(
                    "SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view')"
                ):
                    entry = {"name": name, "sql": sql}
                    if kind == "table":
                        quoted = '"' + name.replace('"', '""') + '"'
                        entry["rows"] = conn.execute(
                            f"SELECT COUNT(*) FROM {quoted}"
                        ).fetchone()[0]
                    out.append(entry)
                return out
        if suffix not in (".xlsx", ".csv"):
            raise ValueError("Expected Excel, CSV or SQLite")
        delim, sheets = self.read_rows(p)
        if suffix == ".csv":
            rows = sheets[None]
            return {
                "delimiter": delim,
                "rows": len(rows),
                "preview": [trim(r) for r in rows[:PREVIEW_ROWS]],
                "note": "preview rows are numbered from 0; use that number as header_row",
            }
        names = list(sheets)
        if sheet is not None:
            names = [self.sheet_names(p, sheets, sheet)[0]]
        out, used = [], 0
        for name in sheets:
            rows = sheets[name]
            entry = {
                "sheet": name,
                "rows": len(rows),
                "columns": max((len(trim(r)) for r in rows), default=0),
            }
            if name in names:
                preview = [trim(r) for r in rows[:PREVIEW_ROWS]]
                size = len(encode(preview))
                if sheet is not None or used == 0 or used + size <= INSPECT_PREVIEW_CHARS:
                    entry["preview"] = preview
                    used += size
                else:
                    entry["preview_omitted"] = True
            out.append(entry)
        result = {
            "sheets": out,
            "note": "preview rows are numbered from 0; use that number as header_row",
        }
        if any(e.get("preview_omitted") for e in out):
            result["note"] += (
                '; some previews were omitted for space: inspect_table with "sheet" shows one'
            )
        return result

    @staticmethod
    def sheet_names(p, sheets, sheet):
        """Resolve a sheet argument (name, index, list, or "*") to sheet names."""
        available = list(sheets)
        if sheet == "*":
            return available
        requested = sheet if isinstance(sheet, list) else [sheet]
        if not requested:
            raise ValueError("sheet list is empty")
        names = []
        for item in requested:
            if isinstance(item, str) and item in sheets:
                names.append(item)
            elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
                item = int(item)
            if isinstance(item, int) and not isinstance(item, bool):
                if not 0 <= item < len(available):
                    raise ValueError(
                        f"Sheet index {item} out of range; sheets: {available}"
                    )
                names.append(available[item])
            elif not (isinstance(item, str) and item in sheets):
                raise ValueError(f"Unknown sheet {item!r} in {p.name}; sheets: {available}")
        if len(set(names)) != len(names):
            raise ValueError("A sheet is listed twice")
        return names

    def connection(self):
        if self.db is None:
            import duckdb

            self.db = duckdb.connect(
                config={
                    "enable_external_access": False,
                    "threads": 2,
                    "memory_limit": "2GB",
                }
            )
        return self.db

    @staticmethod
    def header_names(header, width):
        cols, used = [], set(RESERVED_COLUMNS)
        for i in range(width):
            raw = header[i] if header is not None and i < len(header) else None
            base = (raw or "").strip() or f"column_{i}"
            col, n = base, 1
            while col.casefold() in used:
                col, n = f"{base}_{n}", n + 1
            used.add(col.casefold())
            cols.append(col)
        return cols

    def load_table(self, path, name, sheet=0, header_row=0):
        import pandas as pd

        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name)
            or type(header_row) is not int
            or header_row < -1
        ):
            raise ValueError(
                "Invalid table name (letters, digits, underscore) or header_row (>= -1)"
            )
        p = self.resolve(path)
        if p.suffix.lower() not in (".xlsx", ".csv"):
            raise ValueError("Expected CSV or Excel; use query_sqlite for SQLite")
        _, sheets = self.read_rows(p)
        excel = p.suffix.lower() == ".xlsx"
        names = self.sheet_names(p, sheets, sheet) if excel else [None]
        columns, data, counts, headers = [], {}, {}, {}
        row_ids, sheet_ids = [], []
        for sheet_name in names:
            rows = sheets[sheet_name]
            if header_row >= len(rows):
                raise ValueError(
                    f"header_row {header_row} beyond the {len(rows)} rows of "
                    f"{sheet_name or p.name}"
                )
            start = header_row + 1
            body = [(i, r) for i, r in enumerate(rows[start:], start) if any(
                v is not None for v in r
            )]
            header = rows[header_row] if header_row >= 0 else None
            width = max(
                [len(trim(header or []))] + [len(trim(r)) for _, r in body]
            )
            cols = self.header_names(header, width)
            headers[sheet_name] = cols
            counts[sheet_name] = len(body)
            offset = len(row_ids)
            for col in cols:
                if col not in data:
                    columns.append(col)
                    data[col] = [None] * offset
            for j, col in enumerate(cols):
                data[col].extend(r[j] if j < len(r) else None for _, r in body)
            for col in columns:
                if col not in cols:
                    data[col].extend([None] * len(body))
            row_ids.extend(i for i, _ in body)
            sheet_ids.extend([sheet_name] * len(body))
        frame = pd.DataFrame(
            {col: pd.array(data[col], dtype="string") for col in columns}
        )
        frame["_row"] = row_ids
        if excel:
            frame["_sheet"] = pd.array(sheet_ids, dtype="string")
        self.connection().register(name, frame)
        self.tables[name] = list(frame.columns)
        result = {
            "table": name,
            "rows": len(frame),
            "columns": list(frame.columns),
        }
        if excel:
            result["rows_per_sheet"] = counts
        distinct = {tuple(c) for c in headers.values()}
        if len(distinct) > 1:
            result["warning"] = (
                "Sheets have different headers; missing columns are NULL. Headers: "
                + encode(headers)
            )
        result["preview"] = [
            [None if pd.isna(v) else v for v in row]
            for row in frame.head(3).itertuples(index=False)
        ]
        return result

    def run_sql(self, sql):
        if not isinstance(sql, str):
            raise TypeError("sql must be a string")
        db = self.connection()
        statements = db.extract_statements(sql)
        if len(statements) != 1 or str(statements[0].type).split(".")[-1] != "SELECT":
            raise ValueError("Only one SELECT/WITH query is permitted")
        timer = threading.Timer(self.timeout, db.interrupt)
        timer.start()
        try:
            cur = db.execute(sql)
            rows = cur.fetchmany(101)
            return {
                "columns": [c[0] for c in cur.description],
                "rows": rows[:100],
                "truncated": len(rows) > 100,
            }
        finally:
            timer.cancel()
            timer.join()

    def query_sqlite(self, path, sql):
        p = self.resolve(path)
        if p.suffix.lower() != ".sqlite":
            raise ValueError("Expected SQLite")
        if not isinstance(sql, str):
            raise TypeError("sql must be a string")
        deadline = time.monotonic() + self.timeout
        with closing(sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }

            def authorize(action, a, b, db, source):
                if action == sqlite3.SQLITE_FUNCTION and (b or "").lower() in {
                    "load_extension",
                    "writefile",
                    "readfile",
                }:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

            conn.set_authorizer(authorize)
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            cur = conn.execute(sql)
            if cur.description is None:
                raise ValueError("Expected read query")
            rows = cur.fetchmany(101)
            return {
                "columns": [c[0] for c in cur.description],
                "rows": rows[:100],
                "truncated": len(rows) > 100,
            }

    def close(self):
        if self.db is not None:
            self.db.close()
