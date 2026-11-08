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
        # Rulebook: "reason" is optional and unscored, but must be a string if present.
        return "reason" not in record or isinstance(record["reason"], str)
    return kind == "abstain"


def scalar(value):
    return isinstance(value, str) or type(value) is bool or number(value)


class ProtocolError(ValueError):
    """The model reply is not a usable action (bad JSON, unknown tool, bad arguments)."""


def strip_reasoning(text):
    # Discard explicit reasoning only if the model emitted a closing tag.
    return text.rsplit("</think>", 1)[-1].strip()


def first_json_object(text):
    """Return (object, end_index) for the first complete top-level JSON object, else None.

    Tolerates prose or a markdown fence around the object. Only spans whose outer
    braces balance are tried, so a truncated reply never yields one of its nested
    objects (e.g. the "arguments" of an unfinished call).
    """
    depth, start, in_string, escape = 0, None, False, False
    for i, ch in enumerate(text):
        if depth and in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif depth and ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value, i + 1
    return None


def json_problem(text):
    """Explain why text holds no complete JSON object, precisely enough to fix it."""
    start = text.find("{")
    if start < 0:
        return "Reply contains no JSON object (no '{' found)"
    body = text[start:]
    tail = body[-120:] if len(body) > 120 else body
    depth, in_string, escape, end = 0, False, False, len(body)
    for i, ch in enumerate(body):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if in_string:
        return (
            "Invalid JSON: the reply ends inside a string, so a closing double quote is "
            "missing. Inside a string, escape double quotes as \\\" (e.g. SQL identifiers). "
            f"Reply ended with: <<<{tail}>>>"
        )
    if depth > 0:
        return (
            f"Invalid JSON: {depth} '{{' never closed. Braces inside a string do not count: "
            "check that each string value ends with an unescaped double quote BEFORE the "
            f"closing braces. Reply ended with: <<<{tail}>>>"
        )
    try:
        json.loads(body[:end])
    except json.JSONDecodeError as exc:
        around = body[max(0, exc.pos - 60) : exc.pos + 20]
        return f"Invalid JSON: {exc.msg} at character {exc.pos}, near <<<{around}>>>"
    return "Reply contains no complete JSON object"


def parse_action(text):
    text = strip_reasoning(text)
    found = first_json_object(text)
    if found is None:
        raise ProtocolError(json_problem(text))
    return found[0]


# Parameters accepted by each tool: required, optional. Mirrors zaic.tools.Tools.
TOOL_PARAMS = {
    "list_files": ((), ()),
    "read_document": (("path",), ("offset", "limit")),
    "inspect_table": (("path",), ("sheet",)),
    "load_table": (("path", "name"), ("sheet", "header_row")),
    "run_sql": (("sql",), ()),
    "query_sqlite": (("path", "sql"), ()),
}
TOOL_NAMES = frozenset(TOOL_PARAMS)
INTEGER_PARAMS = {"offset", "limit", "header_row"}


def normalize_action(action):
    """Canonicalize wrapped or flat tool calls to {"action","name","arguments"}.

    Only whitelisted tools are accepted. Parameters may sit inside "arguments" or
    beside it; unknown keys (e.g. "thought") are dropped, conflicting duplicates and
    missing required parameters are rejected with an explicit message.
    """
    kind = action.get("action")
    if kind == "final":
        return {
            k: action[k] for k in ("action", "type", "answer", "reason") if k in action
        }
    if kind == "tool":
        name = action.get("name")
        extras = {
            k: v for k, v in action.items() if k not in {"action", "name", "arguments"}
        }
    elif isinstance(kind, str) and kind in TOOL_NAMES:
        # Flat form: "name" is the load_table destination table, not the tool.
        name = kind
        extras = {k: v for k, v in action.items() if k not in {"action", "arguments"}}
    else:
        raise ProtocolError(
            f'Unknown action {kind!r}; use "tool" (with a name from {sorted(TOOL_NAMES)}) or "final"'
        )
    if not isinstance(name, str) or name not in TOOL_NAMES:
        raise ProtocolError(
            f"Unknown tool {name!r}; choose one of {sorted(TOOL_NAMES)}"
        )
    arguments = action.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments must be a JSON object")
    merged = dict(arguments)
    for key, value in extras.items():
        if key in merged and merged[key] != value:
            raise ProtocolError(f"Parameter {key!r} given twice with different values")
        merged[key] = value
    required, optional = TOOL_PARAMS[name]
    merged = {k: v for k, v in merged.items() if k in required or k in optional}
    for key in INTEGER_PARAMS & set(merged):
        value = merged[key]
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            merged[key] = int(value)
    missing = [k for k in required if merged.get(k) in (None, "")]
    if missing:
        raise ProtocolError(f"{name} requires {list(required)}; missing {missing}")
    return {"action": "tool", "name": name, "arguments": merged}
