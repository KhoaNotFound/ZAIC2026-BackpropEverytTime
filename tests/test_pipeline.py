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
        assert tools.inspect_table("sales.xlsx")["sheets"][0]["preview"][0][0] == "Báo cáo"
        result = tools.load_table("sales.xlsx", "sales", sheet="Tháng 1", header_row=1)
        assert result["columns"] == ["Mã", "Tiền", "_row", "_sheet"]
        assert result["preview"] == [["A", "125", 2, "Tháng 1"]]
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
    # Rulebook §7: reason is optional, but must be a string when present.
    assert valid_answer({"type": "unanswerable"})
    assert not valid_answer({"type": "unanswerable", "reason": 1})
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
    assert [d["points"] for d in report["details"]] == [1, -1, 0]
    assert report["details"][1]["prediction"] == predictions[1]
    assert report["details"][1]["expected"] == answers[1]
    assert report["details"][2]["prediction"] is None
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


@pytest.mark.parametrize("evaluate_after", [False, True])
def test_cli_checkpoint_resume_and_evaluation(
    source, tmp_path, monkeypatch, capsys, evaluate_after
):
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

    monkeypatch.setattr(zaic.model, "HFBackend", FakeBackend)
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
    if evaluate_after:
        argv.append("--evaluate")
    monkeypatch.setattr(sys, "argv", argv)
    inference.main()
    assert read_jsonl(output)[0]["answer"] == 300
    assert '"answer": 300' in capsys.readouterr().out
    assert output.with_suffix(".evaluation.json").exists() == evaluate_after
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--evaluate"])
    inference.main()
    assert len(read_jsonl(output)) == 1
    report = json.loads(output.with_suffix(".evaluation.json").read_text())
    detail = report["details"][0]
    assert detail["prediction"] == detail["expected"] == read_jsonl(output)[0]
    assert detail["question"] == q["question"]
    assert detail["outcome"] == "correct" and detail["points"] == 1
    assert detail["harness_status"] == "ok"
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
    report = json.loads(output.with_suffix(".evaluation.json").read_text())
    assert report["overall"]["accuracy"] == 1
    bench = report["runtime"]
    assert bench["model"] == "Qwen/Qwen3.5-9B" and bench["questions_run"] == 1
    assert bench["status"] == {"ok": 1}
    assert bench["outcome_by_status"] == {"ok": {"correct": 1}}
    assert bench["steps_per_question"]["max"] == 2


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
    (source / "answers.jsonl").write_text(
        "".join(
            json.dumps({"id": q["id"], "type": "number", "answer": 300}) + "\n"
            for q in questions
        )
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

    monkeypatch.setattr(zaic.model, "HFBackend", BrokenBackend)
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
            "--evaluate",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        inference.main()
    assert exc.value.code == 1
    assert len(read_jsonl(output)) == 1
    report = json.loads(output.with_suffix(".evaluation.json").read_text())
    assert report["overall"]["abstained"] == 2
    assert {d["harness_status"] for d in report["details"]} == {"error", "not_run"}
    assert sum(d["prediction"] is None for d in report["details"]) == 1
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
            "arguments": {"path": "sales.csv", "name": "first"},
            "name": "conflicting",
        },
        {"action": "run_sql", "arguments": "SELECT 1"},
        {"action": "tool", "name": "load_table", "arguments": {"path": "sales.csv"}},
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


def test_cli_continues_after_context_limit(source, tmp_path, monkeypatch):
    import importlib.metadata
    import sys

    import inference
    import zaic.model
    from zaic.model import ContextLimitError

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
    replies = iter(
        [
            ContextLimitError("Context limit: 15513 input + 2048 output > 16384"),
            '{"action":"tool","name":"query_sqlite","arguments":{"path":"pos.sqlite","sql":"select sum(amount) from sales"}}',
            '{"action":"final","type":"number","answer":300}',
        ]
    )

    class LongPromptBackend:
        def __init__(self, *args):
            self.device_map = {"": "cpu"}

        def reset_peak(self):
            pass

        def memory(self):
            return {}

        def generate(self, messages):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply, {"input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(zaic.model, "HFBackend", LongPromptBackend)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test")
    output = tmp_path / "ctx" / "predictions.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--data-dir",
            str(source),
            "--limit",
            "2",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
    )
    inference.main()  # no SystemExit: the run is not stopped
    predictions = read_jsonl(output)
    assert len(predictions) == 2
    statuses = [m["status"] for m in read_jsonl(output.with_suffix(".metrics.jsonl"))]
    assert statuses == ["context_limit", "ok"]
    assert predictions[0]["type"] == "abstain" and predictions[1]["answer"] == 300
    assert (
        json.loads(output.with_suffix(".summary.json").read_text())["stopped_on_error"]
        is False
    )


def test_parse_tolerates_prose_and_rejects_truncated_json():
    from zaic.formats import ProtocolError

    assert (
        parse_action('I will list files.\n{"action":"final","type":"abstain"} done')[
            "type"
        ]
        == "abstain"
    )
    assert parse_action('{"bad": } {"action":"final"}')["action"] == "final"
    with pytest.raises(ProtocolError):
        parse_action(
            '{"action":"tool","name":"run_sql","arguments":{"sql":"SELECT 1 OR'
        )


def test_normalize_merges_extras_and_drops_unknown_keys():
    from zaic.formats import normalize_action

    assert normalize_action(
        {"action": "tool", "name": "run_sql", "sql": "select 1", "thought": "x"}
    ) == {"action": "tool", "name": "run_sql", "arguments": {"sql": "select 1"}}
    assert normalize_action(
        {"action": "inspect_table", "name": "inspect_table", "path": "a.csv"}
    ) == {"action": "tool", "name": "inspect_table", "arguments": {"path": "a.csv"}}
    assert (
        normalize_action(
            {"action": "load_table", "path": "a.csv", "name": "t", "header_row": "1"}
        )["arguments"]["header_row"]
        == 1
    )
    final = {"action": "final", "type": "number", "answer": 1, "reasoning": "r"}
    assert normalize_action(final) == {"action": "final", "type": "number", "answer": 1}


def test_tools_call_reports_signature(source):
    tools = Tools(source, "biz_1")
    try:
        with pytest.raises(TypeError, match="required"):
            tools.call("inspect_table", {"file": "sales.csv"})
    finally:
        tools.close()


def test_json_stop_detects_complete_action():
    from zaic.model import reply_complete

    assert reply_complete('{"action":"final","type":"abstain"}', thinking=False)
    assert not reply_complete(
        '{"action":"tool","arguments":{"sql":"x"}', thinking=False
    )
    assert not reply_complete('<think>{"a":1}', thinking=True)
    assert reply_complete('<think>{"a":1}</think>{"action":"final"}', thinking=True)


def test_question_seed_is_stable_and_order_independent():
    from inference import question_seed

    assert question_seed(42, "q1") == question_seed(42, "q1")
    assert question_seed(42, "q1") != question_seed(42, "q2")
    assert question_seed(42, "q1") != question_seed(43, "q1")


def ask(backend, source):
    return answer_question(
        backend,
        source,
        {
            "id": "q",
            "business_id": "biz_1",
            "question": "Tổng?",
            "as_of_date": "2026-01-01",
        },
    )


def test_sql_error_feedback_lists_loaded_columns(source):
    backend = ScriptedBackend(
        [
            '{"action":"load_table","path":"sales.csv","name":"sales"}',
            '{"action":"run_sql","sql":"SELECT SUM(REPLACE(\',\', \'\', amount)) FROM sale"}',
            '{"action":"run_sql","sql":"SELECT SUM(CAST(amount AS INTEGER)) FROM sales WHERE status=\'done\'"}',
            '{"action":"final","type":"number","answer":300}',
        ]
    )
    prediction, detail = ask(backend, source)
    assert prediction["answer"] == 300 and detail["status"] == "ok"
    feedback = detail["trace"][1]["feedback"]
    assert "Tool error from run_sql" in feedback
    assert '"sales": ["id", "status", "amount", "_row"]' in feedback
    assert "list_files" not in feedback


def test_repeated_action_is_warned_not_rerun(source):
    call = '{"action":"tool","name":"query_sqlite","arguments":{"path":"pos.sqlite","sql":"select sum(amount) from sales"}}'
    backend = ScriptedBackend(
        [call, call, '{"action":"final","type":"number","answer":300}']
    )
    prediction, detail = ask(backend, source)
    assert prediction["answer"] == 300 and detail["status"] == "ok"
    assert "tool_result" not in detail["trace"][1]
    assert "repeated your previous action" in detail["trace"][1]["feedback"]
    assert detail["tool_calls"] == 1


def test_repeat_counter_resets_on_different_reply(source):
    listing = '{"action":"list_files"}'
    backend = ScriptedBackend([listing, listing, "not json", listing, listing, listing])
    _, detail = ask(backend, source)
    assert detail["status"] == "stalled" and detail["steps"] == 6


def test_final_needs_data_evidence_not_only_listing(source):
    backend = ScriptedBackend(
        [
            '{"action":"list_files"}',
            '{"action":"final","type":"number","answer":5}',
            '{"action":"final","type":"number","answer":"5"}',
            '{"action":"inspect_table","path":"sales.csv"}',
            '{"action":"final","type":"number","answer":5}',
        ]
    )
    prediction, detail = ask(backend, source)
    assert prediction["answer"] == 5
    assert "No data tool has succeeded" in detail["trace"][1]["feedback"]
    assert "a JSON number" in detail["trace"][2]["feedback"]


def test_invalid_long_reply_is_shortened_in_history(source):
    junk = "{" + " OR x" * 2000
    backend = ScriptedBackend([junk, '{"action":"final","type":"abstain"}'])
    _, detail = ask(backend, source)
    assert detail["trace"][0]["response"] == junk
    assert junk not in backend.prompts[1]
    assert "more characters omitted" in backend.prompts[1]
    assert "SECRET LABEL" not in "".join(backend.prompts)


def test_multi_sheet_load_keeps_order_blanks_and_allows_fill_down(source):
    import datetime as dt

    import openpyxl

    wb = openpyxl.Workbook()
    first = wb.active
    first.title = "T01"
    first.append(["Mã đơn", "Ngày", "Tiền", "Ghi chú"])
    first.append(["D1", dt.datetime(2025, 1, 2), 1000.0, "  "])
    first.append([None, None, 2500.5, None])
    first.append([])  # fully blank row: skipped, numbering kept
    first.append(["D2", dt.datetime(2025, 1, 3, 8, 30), 300, "x"])
    second = wb.create_sheet("T02")
    second.append(["Mã đơn", "Ngày", "Tiền", "Kênh"])
    second.append(["D3", "05/02/2025", 45689, "web"])
    second.append([None, None, 10, None])
    wb.save(source / "data/biz_1/orders.xlsx")
    tools = Tools(source, "biz_1")
    try:
        inspected = tools.inspect_table("orders.xlsx")
        assert [s["sheet"] for s in inspected["sheets"]] == ["T01", "T02"]
        assert inspected["sheets"][0]["rows"] == 5
        # Previews show the same text that load_table stores.
        assert inspected["sheets"][0]["preview"][1] == ["D1", "2025-01-02", "1000"]
        assert inspected["sheets"][0]["preview"][2] == [None, None, "2500.5"]
        assert tools.inspect_table("orders.xlsx", sheet="T02")["sheets"][1]["preview"][1][2] == "45689"
        with pytest.raises(ValueError, match="T01"):
            tools.load_table("orders.xlsx", "t", sheet="T03")
        result = tools.load_table("orders.xlsx", "t", sheet="*")
        assert result["rows"] == 5
        assert result["rows_per_sheet"] == {"T01": 3, "T02": 2}
        assert result["columns"] == ["Mã đơn", "Ngày", "Tiền", "Ghi chú", "Kênh", "_row", "_sheet"]
        assert "different headers" in result["warning"]
        rows = tools.run_sql(
            'SELECT _sheet, _row, "Mã đơn", "Ngày", "Ghi chú" FROM t ORDER BY _sheet, _row'
        )["rows"]
        assert rows == [
            ("T01", 1, "D1", "2025-01-02", None),
            ("T01", 2, None, None, None),
            ("T01", 4, "D2", "2025-01-03 08:30:00", "x"),
            ("T02", 1, "D3", "05/02/2025", None),
            ("T02", 2, None, None, None),
        ]
        # The fill-down pattern documented in TOOL_HELP.
        filled = tools.run_sql(
            'WITH g AS (SELECT *, COUNT("Mã đơn") OVER (PARTITION BY _sheet ORDER BY _row) AS grp FROM t) '
            'SELECT FIRST_VALUE("Mã đơn") OVER (PARTITION BY _sheet, grp ORDER BY _row) AS m, '
            'CAST("Tiền" AS DECIMAL(18,2)) AS tien FROM g ORDER BY _sheet, _row'
        )
        assert [r[0] for r in filled["rows"]] == ["D1", "D1", "D2", "D3", "D3"]
        assert tools.load_table("orders.xlsx", "two", sheet=["T02", 0])["rows"] == 5
    finally:
        tools.close()


def test_decimal_sql_results_encode_as_json_numbers(source):
    from zaic.tools import encode

    tools = Tools(source, "biz_1")
    try:
        tools.load_table("sales.csv", "sales")
        result = tools.run_sql(
            "SELECT SUM(CAST(amount AS DECIMAL(18,2))) AS s, "
            "CAST('12.35' AS DECIMAL(18,2)) AS p FROM sales"
        )
        assert encode(result["rows"]) == "[[1200, 12.35]]"
    finally:
        tools.close()


def test_large_inspect_keeps_every_sheet_name(source):
    import openpyxl

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for month in range(1, 21):
        sheet = wb.create_sheet(f"T{month:02d}")
        sheet.append(["c" * 40 + str(i) for i in range(30)])
        for _ in range(8):
            sheet.append(["v" * 40] * 30)
    wb.save(source / "data/biz_1/big.xlsx")
    tools = Tools(source, "biz_1")
    try:
        from zaic.tools import encode

        result = tools.inspect_table("big.xlsx")
        assert [s["sheet"] for s in result["sheets"]] == [f"T{m:02d}" for m in range(1, 21)]
        assert all(s["rows"] == 9 for s in result["sheets"])
        assert result["sheets"][-1].get("preview_omitted")
        assert len(encode(result)) < 10000  # fits the agent's tool output limit
    finally:
        tools.close()


def test_context_overflow_compacts_old_tool_results(source):
    from zaic.model import ContextLimitError

    replies = iter(
        [
            '{"action":"inspect_table","path":"sales.csv"}',
            '{"action":"load_table","path":"sales.csv","name":"sales"}',
            '{"action":"run_sql","sql":"SELECT SUM(CAST(amount AS INTEGER)) FROM sales WHERE status=\'done\'"}',
            '{"action":"final","type":"number","answer":300}',
        ]
    )

    class SmallContextBackend:
        def __init__(self):
            self.prompts = []

        def generate(self, messages):
            full = sum("Tool result:" in m["content"] for m in messages)
            if full > 1:
                raise ContextLimitError("too long")
            self.prompts.append(json.dumps(messages))
            return next(replies), {"input_tokens": 10, "output_tokens": 5}

    backend = SmallContextBackend()
    prediction, detail = ask(backend, source)
    assert prediction["answer"] == 300 and detail["status"] == "ok"
    assert detail["compactions"] == 2
    assert "omitted to fit the context window" in backend.prompts[-1]


def test_context_overflow_without_anything_to_drop_abstains(source):
    from zaic.model import ContextLimitError

    class TooLong:
        def generate(self, messages):
            raise ContextLimitError("too long")

    prediction, detail = ask(TooLong(), source)
    assert prediction["type"] == "abstain" and detail["status"] == "context_limit"


def test_prompt_states_step_budget_and_shared_files(source):
    backend = ScriptedBackend(
        ['{"action":"list_files"}', '{"action":"final","type":"abstain"}']
    )
    answer_question(
        backend,
        source,
        {"id": "q", "business_id": "biz_1", "question": "?", "as_of_date": "2026-01-01"},
        max_steps=2,
    )
    assert "at most 2 actions" in backend.prompts[0]
    assert "calendar.csv" in backend.prompts[0]
    assert "your next reply must be a final answer" in backend.prompts[1]


def test_scoring_phone_names_and_unreadable_reason():
    gold = {"type": "string", "answer": "Diệp Mai Nam", "alias": ["KH0001"]}
    assert correct({"type": "string", "answer": "Diệp Mai Nam 0937356589"}, gold)
    assert correct({"type": "string", "answer": "0937356589 - Diệp Mai Nam"}, gold)
    assert not correct({"type": "string", "answer": "Diệp Mai Nam 12"}, gold)
    assert not correct({"type": "string", "answer": "Diệp Mai Nam Anh"}, gold)
    answers = [{"id": "a", "type": "unanswerable", "answer": None, "alias": []}] * 1
    answers = answers + [{"id": "b", "type": "unanswerable", "answer": None, "alias": []}]
    report = score(
        [{"id": "a", "type": "unanswerable"}, {"id": "b", "type": "unanswerable", "reason": 5}],
        answers,
    )
    assert [d["outcome"] for d in report["details"]] == ["correct", "abstained"]


def test_chat_template_without_system_role_falls_back():
    from zaic.model import chat_text

    class NoSystemTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if any(m["role"] == "system" for m in messages):
                raise ValueError("System role not supported")
            return "|".join(m["content"] for m in messages)

    text = chat_text(
        NoSystemTokenizer(),
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        thinking=False,
    )
    assert text == "S\n\nU"
