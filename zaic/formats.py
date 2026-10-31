import datetime as dt
import json
import math
import re


def read_jsonl(path):
    records, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise TypeError(f"{path}:{line_no}: missing string id")
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {record['id']}")
            seen.add(record["id"])
            records.append(record)
    return records


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_answer(record):
    kind, value = record.get("type"), record.get("answer")
    if kind in ("number", "percent"):
        return number(value)
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        return isinstance(value, str) and bool(value.strip())
    if kind == "date":
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}(-\d{2})?", value
        ):
            return False
        try:
            dt.date.fromisoformat(value if len(value) == 10 else value + "-01")
            return True
        except ValueError:
            return False
    if kind == "list":
        return isinstance(value, list) and all(scalar(v) for v in value)
    if kind == "table":
        if not isinstance(value, dict):
            return False
        cols, rows = value.get("columns"), value.get("rows")
        return (
            isinstance(cols, list)
            and bool(cols)
            and all(isinstance(c, str) for c in cols)
            and len(set(cols)) == len(cols)
            and isinstance(rows, list)
            and all(
                isinstance(r, list)
                and len(r) == len(cols)
                and all(scalar(v) for v in r)
                for r in rows
            )
        )
    if kind == "unanswerable":
        return isinstance(record.get("reason"), str) and bool(record["reason"].strip())
    return kind == "abstain"


def scalar(value):
    return isinstance(value, str) or type(value) is bool or number(value)


def parse_action(text):
    # Discard explicit reasoning only if the model emitted a closing tag.
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    action = json.loads(text)
    if not isinstance(action, dict):
        raise TypeError("Expected one JSON object")
    return action


TOOL_NAMES = frozenset(
    {
        "list_files",
        "read_document",
        "inspect_table",
        "load_table",
        "run_sql",
        "query_sqlite",
    }
)


def normalize_action(action):
    """Accept only known tools in wrapped or flat JSON, without guessing arguments."""
    kind = action.get("action")
    if kind == "final":
        return action
    if kind == "tool":
        name = action.get("name")
        arguments = action.get("arguments", {})
        if set(action) - {"action", "name", "arguments"}:
            raise ValueError("Put tool parameters inside arguments")
    elif isinstance(kind, str) and kind in TOOL_NAMES:
        name = kind
        if "arguments" in action:
            if set(action) - {"action", "arguments"}:
                raise ValueError("Use either flat parameters or arguments, not both")
            arguments = action["arguments"]
        else:
            # name is the destination table name for load_table, not the tool name.
            arguments = {k: v for k, v in action.items() if k != "action"}
    else:
        raise ValueError("Unknown action; choose a documented tool or final")
    if not isinstance(name, str) or name not in TOOL_NAMES:
        raise ValueError("Unknown tool")
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be an object")
    arguments = dict(arguments)
    if name == "list_files":
        # Listing is always scoped to the current business. A path cannot broaden it.
        arguments.pop("path", None)
    return {"action": "tool", "name": name, "arguments": arguments}
