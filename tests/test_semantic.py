"""Semantic layer: value parsers, §2 periods, README roles, Shop build, intents.

Uses a synthetic shop written to tmp_path; no model and no real data.
"""

import datetime as dt
from decimal import Decimal

import pytest

from zaic.intent import entity_index, parse, run
from zaic.metrics import Metrics
from zaic.periods import find_periods, normalize_question, previous_period, tet
from zaic.readme import column_role, header_role, parse_readme
from zaic.semantic import Shop, normalize_status
from zaic.values import parse_datetime, parse_number, parse_qty, separator_style


@pytest.mark.parametrize(
    ("text", "decimal", "expected"),
    [
        ("560,000", None, "560000"),
        ("29.000đ", None, "29000"),
        ("45,000.0000", None, "45000"),
        ("129 000", None, "129000"),
        ("41k", None, "41000"),
        ("1tr090", None, "1090000"),
        ("409tr663", None, "409663000"),
        ("1,45tr", None, "1450000"),
        ("332,5972tr", None, "332597200"),
        ("-49.000đ", None, "-49000"),
        ("(1,000)", None, "-1000"),
        ("1.5", ".", "1.5"),
        ("abc", None, None),
    ],
)
def test_parse_number(text, decimal, expected):
    got = parse_number(text, decimal)
    assert (got is None and expected is None) or got == Decimal(expected)


def test_separator_style_reads_the_whole_column():
    assert separator_style(["1.5", "2.25", "10"]) == "."
    assert separator_style(["1,5", "2,25"]) == ","
    assert separator_style(["1.000.000", "25.000"]) == ","


@pytest.mark.parametrize(
    ("text", "year", "expected"),
    [
        ("01/02/2025", None, dt.datetime(2025, 2, 1)),
        ("2025-02-01 06:00:25", None, dt.datetime(2025, 2, 1, 6, 0, 25)),
        ("45689", None, dt.datetime(2025, 2, 1)),
        ("1 thg 6 2025", None, dt.datetime(2025, 6, 1)),
        ("3/4", 2026, dt.datetime(2026, 4, 3)),
        ("3/4", None, None),
        ("Tổng tháng", None, None),
    ],
)
def test_parse_datetime(text, year, expected):
    assert parse_datetime(text, year) == expected


def test_parse_qty_keeps_the_unit():
    assert parse_qty("2 hộp 30") == (Decimal(2), "hộp 30")
    assert parse_qty("-1 bình") == (Decimal(-1), "bình")
    assert parse_qty("3") == (Decimal(3), None)


def test_normalize_question_repairs_telex_but_not_english():
    q = normalize_question("dthu thangs 5/2026 tawng/giamr mấy % so vs order clear")
    assert "doanh thu thang 5/2026" in q
    assert "tang/giam" in q
    assert "order" in q and "clear" in q


def test_periods_follow_rulebook_section_2(tmp_path):
    today = dt.date(2026, 3, 18)  # a Wednesday
    def spans(text):
        return [(m.period.start, m.period.end) for m in find_periods(normalize_question(text), today)]

    assert spans("doanh thu tuần trước") == [(dt.date(2026, 3, 9), dt.date(2026, 3, 15))]
    assert spans("tuần này") == [(dt.date(2026, 3, 16), today)]
    assert spans("7 ngày gần nhất") == [(dt.date(2026, 3, 12), today)]
    assert spans("tháng 5") == [(dt.date(2025, 5, 1), dt.date(2025, 5, 31))]  # after today -> last year
    assert spans("quý 1") == [(dt.date(2026, 1, 1), dt.date(2026, 3, 31))]
    assert spans("tu dau thang nay den het hnay") == [(dt.date(2026, 3, 1), today)]
    assert spans("đến hết ngày 10/03/2026") == [(None, dt.date(2026, 3, 10))]
    this_month = find_periods("thang nay", dt.date(2026, 3, 31))[0].period
    # Same number of days in the previous month, capped at its end.
    assert previous_period(this_month).end == dt.date(2026, 2, 28)


def test_tet_uses_the_lunar_calendar(tmp_path):
    cal = tmp_path / "calendar.csv"
    cal.write_text(
        "date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n"
        "2026-02-10,2,23,12,2025,false,\n"
        "2026-02-26,4,10,1,2026,false,\n",
        encoding="utf-8",
    )
    p = tet(2026, cal)
    assert (p.start, p.end) == (dt.date(2026, 2, 10), dt.date(2026, 2, 26))


def test_roles_from_descriptions_and_headers():
    assert column_role("thành tiền sau giảm") == "line_amount"
    assert column_role("trạng thái đơn") == "status"
    assert column_role("số lượng đã đổi ra đơn vị cơ bản") == "base_qty"
    assert header_role("Mã hóa đơn") == "order_id"
    assert header_role("NgayBan") == "date"
    assert header_role("SL (ĐV cơ bản)") == "base_qty"
    assert header_role("Thành tiền (đ)") == "line_amount"


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ("Đã hoàn thành", "done"),
        ("đã lấy", "done"),
        ("Hoàn trả", "returned"),
        ("hàng trả về", "returned"),
        ("Đã hủy đơn", "cancelled"),
        ("Không giao được", "cancelled"),
        ("Đang vận chuyển", "shipping"),
    ],
)
def test_status_mapping(raw, status):
    assert normalize_status(raw) == status


README = """# Dữ liệu biz_t

Sổ chi tiết bán hàng: mỗi dòng một dòng hàng.

## File
- `ban_hang.csv` (csv)

## Cột trong `ban_hang.csv`
- **Ngày**: ngày bán
- **Mã HĐ**: mã đơn
- **Hàng**: tên hàng
- **SL**: số lượng
- **Thành tiền**: thành tiền sau giảm
- **TT**: trạng thái đơn
- **Khách**: tên khách (Khách lẻ nếu không lưu)

## Quy ước
- Dữ liệu có đến hết 31/03/2026.
"""

ROWS = [
    "Ngày,Mã HĐ,Hàng,SL,Thành tiền,TT,Khách",
    "01/03/2026,HD1,Cà phê,2,\"40,000\",Xong,An",
    ",,Bánh mì,1,\"20,000\",,",  # grouped row: order fields blank (§8)
    "01/03/2026,HD2,Cà phê,1,\"20,000\",Đã hủy,Bình",
    "02/03/2026,HD3,Trà,1,\"15,000\",Xong,Khách lẻ",
    "02/03/2026,HD3,Trà,1,\"15,000\",Xong,Khách lẻ",  # identical row of one order (§8)
    "03/03/2026,HD4,Cà phê,1,\"20,000\",Trả hàng,An",
    ",,Tổng tháng 3/2026,,\"130,000\",,",  # total row
]


@pytest.fixture
def shop_dir(tmp_path):
    biz = tmp_path / "data" / "biz_t"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(README, encoding="utf-8")
    (biz / "ban_hang.csv").write_text("\n".join(ROWS) + "\n", encoding="utf-8")
    (tmp_path / "rulebook.md").write_text("rules", encoding="utf-8")
    return tmp_path


def test_shop_applies_fill_down_dedup_and_statuses(shop_dir):
    shop = Shop(shop_dir, "biz_t")
    orders = {o["order_id"]: o for o in shop.orders}
    assert set(orders) == {"HD1", "HD2", "HD3", "HD4"}
    assert orders["HD1"]["revenue"] == Decimal(60000)  # filled-down line counted
    assert orders["HD3"]["revenue"] == Decimal(15000)  # duplicate line counted once
    assert orders["HD2"]["status"] == "cancelled"
    assert orders["HD4"]["status"] == "returned"
    assert not any("Tổng" in (r.get("item_name") or "") for r in shop.lines)
    assert parse_readme(README).data_end == dt.date(2026, 3, 31)


def ask(shop_dir, text, today="2026-03-31"):
    shop = Shop(shop_dir, "biz_t")
    q = {"id": "q", "business_id": "biz_t", "question": text, "as_of_date": today}
    intent = parse(q, shop, None, index=entity_index(shop))
    return run(intent, Metrics(shop))


def test_intents_compute_rulebook_answers(shop_dir):
    assert ask(shop_dir, "doanh thu tháng 3/2026") == {"type": "number", "answer": 75000}
    assert ask(shop_dir, "Tháng 3/2026 có bao nhiêu đơn?") == {"type": "number", "answer": 2}
    assert ask(shop_dir, "tỉ lệ huỷ đơn tháng 3/2026") == {"type": "percent", "answer": 25.0}
    assert ask(shop_dir, "tháng 3/2026 ngày nào bán được nhiều tiền nhất") == {
        "type": "date",
        "answer": "2026-03-01",
    }
    # A period entirely outside the data is unanswerable (§7), not zero.
    assert ask(shop_dir, "doanh thu tháng 1/2024")["type"] == "unanswerable"
