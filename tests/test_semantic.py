"""Semantic layer: value parsers, §2 periods, README roles, Shop build, intents.

Uses a synthetic shop written to tmp_path; no model and no real data.
"""

import datetime as dt
from decimal import Decimal

import pytest

from zaic.intent import entity_index, parse, run
from zaic.metrics import Metrics, Unsupported
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
    q = normalize_question("dthu thangs 5/2026 tawng/giamr mấy % so vs online clear")
    assert "doanh thu thang 5/2026" in q
    assert "tang/giam" in q
    assert "online" in q and "clear" in q


def test_normalize_question_reads_chat_spellings():
    q = normalize_question("wan mik bán đc nhiêu bill, sp nào bán nhìu nhứt")
    assert q == "quan minh ban duoc bao nhieu don san pham nao ban nhieu nhat"
    # "thế" is a particle, "thẻ" (card) is a condition: only the particle goes.
    assert normalize_question("doanh thu thế") == "doanh thu"
    assert normalize_question("mấy đơn thẻ") == "may don the"


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
    assert spans("11-2025 bán được bao nhiêu") == [(dt.date(2025, 11, 1), dt.date(2025, 11, 30))]
    assert spans("3/26 doanh thu") == [(dt.date(2026, 3, 1), dt.date(2026, 3, 31))]
    assert spans("doanh thu th1 th2/2026") == [
        (dt.date(2026, 1, 1), dt.date(2026, 1, 31)),
        (dt.date(2026, 2, 1), dt.date(2026, 2, 28)),
    ]
    assert spans("tính từ ngày 1/1 đến hết hôm nay") == [(dt.date(2026, 1, 1), today)]
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


def ask(shop_dir, text, today="2026-03-31", biz="biz_t"):
    shop = Shop(shop_dir, biz)
    q = {"id": "q", "business_id": biz, "question": text, "as_of_date": today}
    intent = parse(q, shop, None, index=entity_index(shop))
    return run(intent, Metrics(shop)) if intent else None


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


RICH_README = """# Dữ liệu biz_r

Quán nhỏ có sổ nhập hàng, sổ thu tiền và tồn kho.

## File
- `don_hang.csv` (csv)
- `chi_tiet.csv` (csv)
- `thanh_toan.csv` (csv)
- `nhap_hang.csv` (csv)
- `ton_kho.csv` (csv)

## Cột trong `don_hang.csv`
- **Mã HĐ**: mã đơn
- **Ngày**: ngày đặt
- **Khách**: tên khách
- **PTTT**: hình thức thanh toán
- **Tổng**: tổng tiền đơn
- **TT**: trạng thái đơn

## Cột trong `chi_tiet.csv`
- **Mã HĐ**: mã đơn
- **Hàng**: tên hàng
- **ĐVT**: đơn vị tính
- **SL**: số lượng
- **Thành tiền**: thành tiền sau giảm
- **Giá vốn**: giá vốn một đơn vị bán

## Cột trong `thanh_toan.csv`
- **Mã phiếu thu**: mã phiếu thu
- **Mã HĐ**: mã đơn
- **Ngày thu**: ngày thu
- **Số tiền**: số tiền trả lần đó

## Cột trong `nhap_hang.csv`
- **Số phiếu**: mã phiếu nhập
- **Ngày nhập**: ngày nhập
- **NCC**: nhà cung cấp
- **Hàng**: tên hàng
- **ĐVT**: đơn vị nhập
- **SL**: số lượng nhập
- **Giá nhập**: giá nhập một đơn vị
- **TT**: trạng thái phiếu nhập

## Cột trong `ton_kho.csv`
- **Hàng**: tên hàng
- **Tồn cuối**: tồn cuối ngày cuối có dữ liệu

## Quy ước
- Dữ liệu có đến hết 31/03/2026.
"""

RICH_FILES = {
    "don_hang.csv": [
        "Mã HĐ,Ngày,Khách,PTTT,Tổng,TT",
        "HD1,01/03/2026,Trần An,Tiền mặt,60000,Xong",
        "HD2,02/03/2026,Lê Bình,Ghi nợ,100000,Xong",
        "HD3,10/03/2026,Lê Bình,Ghi nợ,50000,Xong",
        "HD4,12/03/2026,Phạm Chi,Ghi nợ,30000,Đã hủy",
    ],
    "chi_tiet.csv": [
        "Mã HĐ,Hàng,ĐVT,SL,Thành tiền,Giá vốn",
        "HD1,Cà phê,ly,2,40000,5000",
        "HD1,Mực khô,con,1,20000,8000",
        "HD2,Cà phê,ly,5,100000,5000",
        "HD3,Trà,ly,2,50000,4000",
        "HD4,Trà,ly,1,30000,4000",
    ],
    "thanh_toan.csv": [
        "Mã phiếu thu,Mã HĐ,Ngày thu,Số tiền",
        "PT1,HD1,01/03/2026,60000",
        "PT2,HD2,02/03/2026,20000",  # paid on the order day: not a repayment (§4)
        "PT3,HD2,05/03/2026,30000",
    ],
    "nhap_hang.csv": [
        "Số phiếu,Ngày nhập,NCC,Hàng,ĐVT,SL,Giá nhập,TT",
        "PN1,01/02/2026,Cty A,Cà phê,ly,10,4000,Đã nhập",
        "PN2,05/03/2026,Cty A,Cà phê,ly,30,6000,Đã nhập",
        "PN3,06/03/2026,Cty B,Trà,ly,10,3000,Đã hủy",
        "PN4,20/03/2026,Cty B,Mực khô,con,5,7000,Đã nhập",
        "PN5,10/03/2026,Cty A,Cà phê,ly,10,8000,Đã nhập",
    ],
    "ton_kho.csv": ["Hàng,Tồn cuối", "Cà phê,33", "Trà,8", "Mực khô,4"],
}


@pytest.fixture
def rich_dir(tmp_path):
    biz = tmp_path / "data" / "biz_r"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text(RICH_README, encoding="utf-8")
    for name, rows in RICH_FILES.items():
        (biz / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "rulebook.md").write_text("rules", encoding="utf-8")
    return tmp_path


def rich(rich_dir, text, today="2026-03-31"):
    return ask(rich_dir, text, today, biz="biz_r")


def test_item_names_are_masked_before_intent_matching(rich_dir):
    # "khô" inside a product name is not the stock word "kho".
    assert rich(rich_dir, "doanh thu Mực khô tháng 3/2026") == {"type": "number", "answer": 20000}
    assert rich(rich_dir, "tháng 3/2026 có bao nhiêu bill") == {"type": "number", "answer": 3}
    assert rich(rich_dir, "tháng 3/2026 bán được bao nhiêu ly") == {"type": "number", "answer": 9}


def test_a_question_with_an_unknown_condition_is_declined(rich_dir):
    assert rich(rich_dir, "doanh thu tháng 3/2026") == {"type": "number", "answer": 210000}
    assert rich(rich_dir, "doanh thu tháng 3/2026 tính riêng hàng khuyến mãi") is None
    assert rich(rich_dir, "doanh thu nhóm đồ khô tháng 3/2026") is None  # group not in the data
    # A filter on something the shop never records is unanswerable (§7).
    assert rich(rich_dir, "tháng 3/2026 khách nữ mua bao nhiêu tiền")["type"] == "unanswerable"
    assert rich(rich_dir, "doanh thu các đơn bán qua tiktok tháng 3/2026")["type"] == "unanswerable"


def test_purchase_metrics_follow_section_5(rich_dir):
    assert rich(rich_dir, "tháng 3/2026 nhập hàng hết bao nhiêu tiền") == {"type": "number", "answer": 295000}
    assert rich(rich_dir, "giá nhập bình quân một ly Cà phê trong tháng 3/2026 là bao nhiêu") == {
        "type": "number",
        "answer": 6500,
    }
    assert rich(rich_dir, "tháng 3/2026 nhập hàng nhiều tiền nhất từ nhà cung cấp nào") == {
        "type": "string",
        "answer": "Cty A",
    }
    assert rich(rich_dir, "lấy số nhập trừ số bán của Cà phê trong tháng 3/2026 thì được bao nhiêu ly") == {
        "type": "number",
        "answer": 33,
    }


def test_debt_metrics_follow_section_4(rich_dir):
    assert rich(rich_dir, "đến hết ngày 04/03/2026 khách còn nợ tổng cộng bao nhiêu tiền") == {
        "type": "number",
        "answer": 80000,
    }
    assert rich(rich_dir, "tính đến hết ngày 31/03/2026, số khách còn công nợ là bao nhiêu") == {
        "type": "number",
        "answer": 1,
    }
    assert rich(rich_dir, "top 3 khách nợ nhiều nhất tính đến hết ngày 31/03/2026, kèm số tiền còn nợ") == {
        "type": "table",
        "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Lê Bình", 100000]]},
    }
    assert rich(rich_dir, "tháng 3/2026 thu hồi được bao nhiêu tiền công nợ từ khách mua chịu") == {
        "type": "number",
        "answer": 30000,
    }
    assert rich(rich_dir, "khách Lê Bình tháng 3/2026 nợ thêm bao nhiêu, trả được bao nhiêu") == {
        "type": "list",
        "answer": [130000, 30000],
    }
    aging = "đến hết ngày 31/03/2026, khách còn nợ bao nhiêu ở nhóm tuổi nợ 0–7 ngày, 8–15 ngày và trên 15 ngày"
    assert rich(rich_dir, aging) == {"type": "list", "answer": [0, 0, 100000]}


def test_stock_unsold_and_profit(rich_dir):
    assert rich(rich_dir, "tồn kho Cà phê cuối ngày hôm nay là bao nhiêu ly") == {"type": "number", "answer": 33}
    # The stock report is for the last day of data only: another day is not guessed.
    with pytest.raises(Unsupported):
        rich(rich_dir, "tồn kho Cà phê cuối ngày hôm nay là bao nhiêu ly", today="2026-03-20")
    unsold = "giai đoạn từ 05/03/2026 đến hết 31/03/2026 có mấy mặt hàng ế, không bán được cái nào"
    assert rich(rich_dir, unsold) == {"type": "number", "answer": 2}
    assert rich(rich_dir, "lãi gộp tháng 3/2026 của cửa hàng là bao nhiêu") == {"type": "number", "answer": 159000}
    assert rich(rich_dir, "tháng 3/2026 hàng nào lời nhất") == {"type": "string", "answer": "Cà phê"}
