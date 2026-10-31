import json
import sqlite3

import pytest

from evaluate import correct, score
from inference import select_questions
from zaic.agent import answer_question
from zaic.formats import parse_action, read_jsonl, valid_answer
from zaic.tools import Tools


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "train"
    shop = root / "data" / "biz_1"
    shop.mkdir(parents=True)
    (root / "rulebook.md").write_text("Exclude canceled orders.")
    (root / "answers.jsonl").write_text("SECRET LABEL")
    (root / "calendar.csv").write_text("date,weekday\n2026-01-01,4\n")
    (shop / "README.md").write_text("Each row is one order.")
    (shop / "sales.csv").write_text(
        "id,status,amount\n1,done,100\n2,cancelled,900\n3,done,200\n"
    )
    with sqlite3.connect(shop / "pos.sqlite") as db:
        db.execute("create table sales(id integer, amount integer)")
        db.executemany("insert into sales values (?,?)", [(1, 100), (2, 200)])
    return root


def test_tools_full_aggregation_and_isolation(source):
    import duckdb

    tools = Tools(source, "biz_1")
    try:
        assert tools.load_table("sales.csv", "sales")["rows"] == 3
        assert (
            tools.run_sql(
                "select sum(cast(amount as decimal(18,0))) from sales where status='done'"
            )["rows"][0][0]
            == 300
        )
        assert tools.query_sqlite("pos.sqlite", "select sum(amount) from sales")[
            "rows"
        ] == [(300,)]
        for path in [
            "../../answers.jsonl",
            str(source / "answers.jsonl"),
            "answers.jsonl",
        ]:
            with pytest.raises(ValueError):
                tools.resolve(path)
        with pytest.raises((ValueError, duckdb.Error)):
            tools.run_sql(f"select * from read_csv('{source}/answers.jsonl')")
        for sql in [
            "drop table sales",
            "select 1; select 2",
            "attach ':memory:' as other",
        ]:
            with pytest.raises((ValueError, duckdb.Error)):
                tools.run_sql(sql)
        for sql in [
            "delete from sales",
            "attach ':memory:' as other",
            "select load_extension('foo')",
        ]:
            with pytest.raises(sqlite3.DatabaseError):
                tools.query_sqlite("pos.sqlite", sql)
        link = tools.root / "outside.csv"
        link.symlink_to(source / "calendar.csv")
        with pytest.raises(ValueError):
            tools.resolve("outside.csv")
    finally:
        tools.close()


def test_excel_raw_header_selection(source):
    import openpyxl

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Tháng 1"
    sheet.append(["Báo cáo"])
    sheet.append(["Mã", "Tiền"])
    sheet.append(["A", 125])
    wb.save(source / "data/biz_1/sales.xlsx")
    tools = Tools(source, "biz_1")
    try:
        assert tools.inspect_table("sales.xlsx")[0]["preview"][0][0] == "Báo cáo"
        result = tools.load_table("sales.xlsx", "sales", sheet="Tháng 1", header_row=1)
        assert result["columns"] == ["Mã", "Tiền"]
        assert (
            tools.run_sql('select sum(cast("Tiền" as integer)) from sales')["rows"][0][
                0
            ]
            == 125
        )
    finally:
        tools.close()


def test_strict_numbers_and_formats():
    gold = {"type": "number", "answer": 1_000_000_000}
    assert correct({"type": "number", "answer": 1_000_000_000}, gold)
    assert not correct({"type": "number", "answer": 1_000_000_001}, gold)
    assert not correct(
        {"type": "number", "answer": 10**18 + 1}, {"type": "number", "answer": 10**18}
    )
    assert not correct({"type": "number", "answer": "1000000000"}, gold)
    assert not correct(
        {"type": "number", "answer": True}, {"type": "number", "answer": 1}
    )
    assert not valid_answer({"type": "number", "answer": float("nan")})
    assert not valid_answer({"type": "date", "answer": "2026-02-30"})
    assert not valid_answer({"type": "unanswerable"})
    assert not correct(
        {"type": "percent", "answer": 0.124}, {"type": "percent", "answer": 12.4}
    )


def test_aliases_accents_and_list_order():
    gold = {"type": "string", "answer": "Cà phê sữa (M)", "alias": ["SP0003"]}
    assert correct({"type": "string", "answer": " CÀ PHÊ SỮA M "}, gold)
    assert correct({"type": "string", "answer": "sp0003"}, gold)
    assert not correct({"type": "string", "answer": "Ca phe sua M"}, gold)
    gold = {"type": "list", "answer": ["A", "B"], "alias": [["SP1"], ["SP2"]]}
    assert correct({"type": "list", "answer": ["SP1", "SP2"]}, gold)
    assert not correct({"type": "list", "answer": ["B", "A"]}, gold)


def test_table_rows_and_column_mapping():
    gold = {
        "type": "table",
        "answer": {"columns": ["customer", "debt"], "rows": [["An", 10], ["Bình", 20]]},
        "alias": [["KH1"], ["KH2"]],
    }
    assert correct(
        {
            "type": "table",
            "answer": {
                "columns": ["debt", "customer"],
                "rows": [[20, "KH2"], [10, "KH1"]],
            },
        },
        gold,
    )
    assert not correct(
        {
            "type": "table",
            "answer": {
                "columns": ["customer", "debt"],
                "rows": [["KH1", 10], ["KH1", 10]],
            },
        },
        gold,
    )


def test_score_scope_and_abstention():
    answers = [{"id": str(i), "type": "number", "answer": i} for i in range(3)]
    predictions = [
        {"id": "0", "type": "number", "answer": 0},
        {"id": "1", "type": "number", "answer": 99},
    ]
    report = score(predictions, answers)
    assert report["overall"]["score"] == 0
    assert report["overall"]["abstained"] == 1
    assert report["overall"]["accuracy"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        score([{"id": "unknown"}], answers)


def test_sampling_reproducible():
    questions = [
        {"id": f"{b}_{i}", "business_id": b} for b in ["a", "b", "c"] for i in range(4)
    ]
    selected = select_questions(questions, 3, 42, "stratified")
    assert len({q["business_id"] for q in selected}) == 3
    assert selected == select_questions(questions, 3, 42, "stratified")
    assert len(select_questions(questions, 0, 42, "stratified")) == 12


class ScriptedBackend:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(json.dumps(messages))
        return next(self.actions), {"input_tokens": 10, "output_tokens": 5}


def test_agent_end_to_end_without_gold(source):
    backend = ScriptedBackend(
        [
            json.dumps(
                {
                    "action": "tool",
                    "name": "query_sqlite",
                    "arguments": {
                        "path": "pos.sqlite",
                        "sql": "select sum(amount) from sales",
                    },
                }
            ),
            '{"action":"final","type":"number","answer":300}',
        ]
    )
    pred, trace = answer_question(
        backend,
        source,
        {
            "id": "q1",
            "business_id": "biz_1",
            "question": "Tổng tiền?",
            "as_of_date": "2026-01-01",
        },
    )
    assert pred == {"id": "q1", "type": "number", "answer": 300}
    assert trace["status"] == "ok"
    assert "SECRET LABEL" not in "".join(backend.prompts)


def test_agent_repairs_errors_and_exhaustion(source):
    backend = ScriptedBackend(["not json", '{"action":"final","type":"abstain"}'])
    pred, detail = answer_question(
        backend, source, {"id": "q", "business_id": "biz_1"}, max_steps=2
    )
    assert pred["type"] == "abstain" and detail["status"] == "ok"
    assert "tool_error" in detail["trace"][0]
    backend = ScriptedBackend(["not json"])
    pred, detail = answer_question(
        backend, source, {"id": "q", "business_id": "biz_1"}, max_steps=1
    )
    assert detail["status"] == "max_steps" and pred["type"] == "abstain"


def test_parse_and_duplicate_ids(tmp_path):
    assert parse_action('```json\n{"action":"final"}\n```')["action"] == "final"
    assert (
        parse_action('<think>reasoning</think>{"action":"final"}')["action"] == "final"
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text('{"id":"a"}\n{"id":"a"}\n')
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_cli_checkpoint_resume_and_evaluation(source, tmp_path, monkeypatch):
    import importlib.metadata
    import sys

    import evaluate
    import inference
    import zaic.model

    q = {
        "id": "q1",
        "business_id": "biz_1",
        "question": "Tổng tiền?",
        "as_of_date": "2026-01-01",
    }
    (source / "questions.jsonl").write_text(json.dumps(q) + "\n")
    (source / "answers.jsonl").write_text('{"id":"q1","type":"number","answer":300}\n')

    class FakeBackend(ScriptedBackend):
        def __init__(self, *args):
            self.device_map = {"": "cpu"}
            super().__init__(
                [
                    '{"action":"tool","name":"query_sqlite","arguments":{"path":"pos.sqlite","sql":"select sum(amount) from sales"}}',
                    '{"action":"final","type":"number","answer":300}',
                ]
            )

        def reset_peak(self):
            pass

        def memory(self):
            return {}

    monkeypatch.setattr(zaic.model, "QwenBackend", FakeBackend)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test")
    output = tmp_path / "out" / "predictions.jsonl"
    argv = [
        "inference.py",
        "--data-dir",
        str(source),
        "--limit",
        "1",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    inference.main()
    assert read_jsonl(output)[0]["answer"] == 300
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    inference.main()
    assert len(read_jsonl(output)) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--predictions",
            str(output),
            "--answers",
            str(source / "answers.jsonl"),
        ],
    )
    evaluate.main()
    assert (
        json.loads(output.with_suffix(".evaluation.json").read_text())["overall"][
            "accuracy"
        ]
        == 1
    )


def test_explicit_model_split_uses_both_gpus_and_keeps_blocks_whole():
    from zaic.model import make_split_map

    prefix = "model.language_model.layers"
    names = {
        prefix,
        "model.language_model.norm",
        "model.language_model.embed_tokens",
        "lm_head",
    }
    names.update(f"{prefix}.{i}" for i in range(32))
    mapping = make_split_map(names, 32, 2)
    assert mapping["model.language_model.embed_tokens"] == 0
    assert all(mapping[f"{prefix}.{i}"] == 0 for i in range(16))
    assert all(mapping[f"{prefix}.{i}"] == 1 for i in range(16, 32))
    assert mapping["model.language_model.norm"] == 1
    assert mapping["lm_head"] == 0
    assert make_split_map(names, 32, 2, tied_embeddings=True)["lm_head"] == 0
    with pytest.raises(ValueError, match="two visible GPUs"):
        make_split_map(names, 32, 1)
    with pytest.raises(ValueError, match="unique decoder"):
        make_split_map(set(), 32, 2)


def test_generation_failure_has_traceback(source):
    class FailingBackend:
        def generate(self, messages):
            raise RuntimeError("Expected all tensors to be on the same device")

    pred, detail = answer_question(
        FailingBackend(),
        source,
        {
            "id": "q",
            "business_id": "biz_1",
            "question": "Tổng tiền?",
            "as_of_date": "2026-01-01",
        },
    )
    assert pred["type"] == "abstain"
    assert detail["status"] == "error"
    assert "RuntimeError" in detail["error_traceback"]
    assert "generate" in detail["error_traceback"]


def test_text_only_split_map_has_no_parent_fallback():
    from zaic.model import make_split_map

    names = {
        "model.layers",
        "model.embed_tokens",
        "model.rotary_emb",
        "model.norm",
        "lm_head",
        *(f"model.layers.{i}" for i in range(4)),
    }
    mapping = make_split_map(names, 4, 2)
    assert "" not in mapping
    assert mapping["model.embed_tokens"] == mapping["lm_head"] == 0
    assert mapping["model.layers.2"] == mapping["model.norm"] == 1


def test_cli_stops_on_runtime_error_without_time_extrapolation(
    source, tmp_path, monkeypatch
):
    import importlib.metadata
    import sys

    import inference
    import zaic.model

    questions = [
        {
            "id": f"q{i}",
            "business_id": "biz_1",
            "question": "Tổng tiền?",
            "as_of_date": "2026-01-01",
        }
        for i in range(2)
    ]
    (source / "questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in questions)
    )

    class BrokenBackend:
        def __init__(self, *args):
            self.device_map = {"model.layers.0": "0"}

        def reset_peak(self):
            pass

        def memory(self):
            return {}

        def generate(self, messages):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(zaic.model, "QwenBackend", BrokenBackend)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test")
    output = tmp_path / "failure" / "predictions.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--data-dir",
            str(source),
            "--limit",
            "2",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        inference.main()
    assert exc.value.code == 1
    assert len(read_jsonl(output)) == 1
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["stopped_on_error"] is True
    assert summary["selected_but_not_completed"] == 1
    assert summary["estimated_3000_questions_hours"] is None
    metrics = read_jsonl(output.with_suffix(".metrics.jsonl"))
    assert "generate" in metrics[0]["error_traceback"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            {
                "action": "load_table",
                "path": "sales.csv",
                "name": "DanhSachHoaDon",
                "header_row": 0,
            },
            {
                "action": "tool",
                "name": "load_table",
                "arguments": {
                    "path": "sales.csv",
                    "name": "DanhSachHoaDon",
                    "header_row": 0,
                },
            },
        ),
        (
            {"action": "list_files", "path": "DanhSachHoaDon.xlsx"},
            {"action": "tool", "name": "list_files", "arguments": {}},
        ),
        (
            {"action": "run_sql", "arguments": {"sql": "select 1"}},
            {"action": "tool", "name": "run_sql", "arguments": {"sql": "select 1"}},
        ),
    ],
)
def test_normalize_observed_qwen_calls(raw, expected):
    from zaic.formats import normalize_action

    assert normalize_action(raw) == expected
    assert normalize_action(expected) == expected


@pytest.mark.parametrize(
    "action",
    [
        {"action": "exec", "code": "print(1)"},
        {"action": "tool", "name": "run_python", "arguments": {}},
        {
            "action": "load_table",
            "arguments": {"path": "sales.csv"},
            "name": "conflicting",
        },
        {"action": "run_sql", "arguments": "SELECT 1"},
    ],
)
def test_normalization_rejects_unknown_or_ambiguous_actions(action):
    from zaic.formats import normalize_action

    with pytest.raises((ValueError, TypeError)):
        normalize_action(action)


def test_flat_calls_execute_tools_and_return_answer(source):
    backend = ScriptedBackend(
        [
            '{"action":"load_table","path":"sales.csv","name":"DanhSachHoaDon","header_row":0}',
            '{"action":"list_files","path":"DanhSachHoaDon.xlsx"}',
            '{"action":"run_sql","sql":"SELECT SUM(CAST(amount AS INTEGER)) FROM DanhSachHoaDon WHERE status=\'done\'"}',
            '{"action":"final","type":"number","answer":300}',
        ]
    )
    prediction, detail = answer_question(
        backend,
        source,
        {
            "id": "q",
            "business_id": "biz_1",
            "question": "Tổng tiền?",
            "as_of_date": "2026-01-01",
        },
    )
    assert prediction["answer"] == 300
    assert detail["status"] == "ok"
    assert all("tool_error" not in step for step in detail["trace"])
    assert (
        detail["trace"][0]["normalized_action"]["arguments"]["name"] == "DanhSachHoaDon"
    )
    assert detail["trace"][1]["normalized_action"]["arguments"] == {}
    assert "SECRET LABEL" not in "".join(backend.prompts)


def test_agent_stops_repeated_listing_instead_of_using_all_steps(source):
    backend = ScriptedBackend(['{"action":"list_files","path":"nonexistent.xlsx"}'] * 3)
    prediction, detail = answer_question(
        backend, source, {"id": "q", "business_id": "biz_1"}
    )
    assert prediction["type"] == "abstain"
    assert detail["status"] == "stalled"
    assert detail["steps"] == 3
    assert "three times" in detail["error"]


def test_agent_stops_identical_protocol_errors(source):
    backend = ScriptedBackend(["not json"] * 3)
    _, detail = answer_question(backend, source, {"id": "q", "business_id": "biz_1"})
    assert detail["status"] == "stalled"
    assert detail["steps"] == 3
