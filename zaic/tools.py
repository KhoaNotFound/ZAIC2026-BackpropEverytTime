"""Read-only tools. No arbitrary Python execution or external SQL file access."""

import csv
import datetime as dt
import json
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

TOOL_HELP = """Tools (one call per turn):
list_files {} -> paths relative to current business directory.
read_document {"path":"README.md","offset":0,"limit":12000} -> text; rulebook.md is split rulebook.
inspect_table {"path":"file.xlsx"} -> sheet names, dimensions and first 6 raw rows;
  CSV -> delimiter, header and first rows; SQLite -> table schemas.
load_table {"path":"file.xlsx","name":"sales","sheet":"Sheet1","header_row":0} ->
  register a whole sheet/CSV in DuckDB, preserving blank cells. header_row is ZERO-based,
  use -1 for no header. sheet defaults to first sheet. All cells loaded as text; CAST as needed.
  Empty/duplicate headers become unique. No automatic fill-forward or deduplication.
  calendar.csv is also available as path. Returns columns and sample rows.
run_sql {"sql":"SELECT ... FROM sales"} -> DuckDB query on loaded tables.
query_sqlite {"path":"pos.sqlite","sql":"SELECT ..."} -> read-only SQLite query.
Queries return at most 100 rows and have a time limit. Aggregate over full tables, not previews.
DuckDB SQL can join loaded tables. SQL has no external file/network access.
Use quoted column identifiers, normalize date/number formats based on observed data.
"""


def serial(value):
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def encode(value):
    return json.dumps(value, ensure_ascii=False, default=serial, allow_nan=False)


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

    def resolve(self, path):
        if path in ("rulebook.md", "calendar.csv"):
            p = (self.split / path).resolve()
            if p.parent != self.split:
                raise ValueError("Invalid shared file")
        else:
            p = (self.root / path).resolve()
            if not p.is_relative_to(self.root):
                raise ValueError("Path outside current business")
        if not p.is_file() or p.suffix.lower() not in {
            ".md",
            ".xlsx",
            ".csv",
            ".sqlite",
        }:
            raise ValueError("Unsupported or missing source file")
        return p

    def call(self, name, args):
        if name not in {
            "list_files",
            "read_document",
            "inspect_table",
            "load_table",
            "run_sql",
            "query_sqlite",
        }:
            raise ValueError("Unknown tool")
        if not isinstance(args, dict):
            raise TypeError("arguments must be an object")
        return getattr(self, name)(**args)

    def list_files(self):
        return [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.resolve().is_relative_to(self.root)
        ]

    def read_document(self, path, offset=0, limit=12000):
        p = self.resolve(path)
        if p.suffix != ".md" or offset < 0 or not 1 <= limit <= 20000:
            raise ValueError("Expected markdown, offset >= 0, limit 1..20000")
        text = p.read_text(encoding="utf-8")
        return {
            "text": text[offset : offset + limit],
            "total_chars": len(text),
            "next_offset": offset + limit if offset + limit < len(text) else None,
        }

    def inspect_table(self, path):
        p = self.resolve(path)
        if p.suffix == ".xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            try:
                from itertools import islice

                return [
                    {
                        "sheet": s.title,
                        "declared_rows": s.max_row,
                        "declared_columns": s.max_column,
                        "preview": list(islice(s.values, 6)),
                    }
                    for s in wb
                ]
            finally:
                wb.close()
        if p.suffix == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                sample = f.read(8192)
                try:
                    delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                except csv.Error:
                    delim = ","
                f.seek(0)
                from itertools import islice

                return {
                    "delimiter": delim,
                    "preview": list(islice(csv.reader(f, delimiter=delim), 6)),
                }
        if p.suffix == ".sqlite":
            with closing(sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)) as conn:
                return [
                    {"name": name, "sql": sql}
                    for name, sql in conn.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type IN ('table','view')"
                    )
                ]
        raise ValueError("Expected Excel, CSV or SQLite")

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

    def load_table(self, path, name, sheet=0, header_row=0):
        import pandas as pd

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) or header_row < -1:
            raise ValueError("Invalid table name/header_row")
        p = self.resolve(path)
        header = None if header_row == -1 else header_row
        if p.suffix == ".csv":
            delim = self.inspect_table(path)["delimiter"]
            frame = pd.read_csv(
                p,
                sep=delim,
                header=header,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        elif p.suffix == ".xlsx":
            frame = pd.read_excel(
                p,
                sheet_name=sheet,
                header=header,
                dtype=str,
                keep_default_na=False,
                engine="openpyxl",
            )
        else:
            raise ValueError("Expected CSV or Excel; use query_sqlite for SQLite")
        cols, used = [], set()
        for i, raw in enumerate(frame.columns):
            base = str(raw).strip() or f"column_{i}"
            col, n = base, 1
            while col.casefold() in used:
                col, n = f"{base}_{n}", n + 1
            used.add(col.casefold())
            cols.append(col)
        frame.columns = cols
        self.connection().register(name, frame)
        return {
            "table": name,
            "rows": len(frame),
            "columns": cols,
            "preview": frame.head(3).values.tolist(),
        }

    def run_sql(self, sql):
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
        if p.suffix != ".sqlite":
            raise ValueError("Expected SQLite")
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
