"""Parse a business README into files, column descriptions and column roles.

READMEs follow one layout: an intro paragraph, `## File`, one
`## Cột trong `<file>`` section per file with `- **<header>**: <description>`
lines, and `## Quy ước` notes. Roles are assigned from the description text
(never from business ids), so unseen shops with the same vocabulary map the
same way; columns whose description matches no rule keep role None.
"""

import datetime as dt
import re
from dataclasses import dataclass, field

from .values import fold

# (role, regex on the folded description). First match wins, so specific rules
# come before general ones.
ROLE_RULES = [
    ("line_no", r"^so thu tu dong"),
    ("order_seq", r"^so thu tu don"),
    ("payment_id", r"^ma phieu thu|^ma giao dich thu"),
    ("purchase_id", r"^ma phieu nhap|^so phieu nhap"),
    ("order_id", r"^ma (?:don|hoa don)\b|^so hoa don|^ma hd\b"),
    ("batch_id", r"^so lo\b|^ma lo\b|^lo hang"),
    ("return_qty", r"^so luong khach tra lai"),
    ("batch_qty", r"^so luong ban lay tu lo"),
    ("base_qty", r"^so luong da doi ra don vi co ban|^so luong quy doi"),
    ("purchase_base_qty", r"^so luong nhap \(don vi co ban\)"),
    ("purchase_qty", r"^so luong nhap"),
    ("stock_qty", r"^ton kho|^ton cuoi|^so luong ton"),
    ("qty", r"^so luong(?! khach tra| ban lay| nhap| ton)"),
    ("datetime", r"^thoi diem dat don|^ngay gio dat"),
    ("time", r"^gio dat don|^gio ban|^gio tao"),
    ("pay_time", r"^thoi diem thu|^ngay thu|^ngay tra no|^ngay tra tien"),
    ("date", r"^ngay ban|^ngay dat|^ngay mua|^ngay lap|^ngay giao dich|^ngay hoa don"),
    ("purchase_date", r"^ngay nhap"),
    ("price_date", r"^ngay (?:bat dau )?ap dung"),
    ("expiry", r"^han su dung|^han dung"),
    ("due_date", r"^han thanh toan|^ngay phai tra"),
    ("due_days", r"^so ngay duoc no"),
    ("delivery_date", r"^ngay giao\b"),
    ("delivery_no", r"^dot giao"),
    ("status", r"^trang thai don|^tinh trang don"),
    ("purchase_status", r"^trang thai phieu nhap"),
    ("note", r"^ghi chu"),
    ("item_code", r"^ma hang|^ma san pham|^ma sp\b|^ma thuoc"),
    ("item_name", r"^ten hang|^hang ban|^ten san pham|^ten mon|^ten thuoc|^mat hang"),
    ("customer_id", r"^ma khach"),
    ("customer_name", r"^ten khach"),
    ("customer_group", r"^nhom khach"),
    ("phone", r"^so dien thoai"),
    ("address", r"^dia chi"),
    ("supplier", r"^ma nha cung cap|^nha cung cap|^ma ncc"),
    ("staff", r"nhan vien"),
    ("channel", r"^kenh ban"),
    ("payment_method", r"^hinh thuc thanh toan"),
    ("unit_base", r"^don vi co ban|^don vi goc"),
    ("unit_big", r"^don vi lon"),
    ("unit_purchase", r"^don vi nhap"),
    ("unit", r"^don vi tinh|^dvt\b"),
    ("factor", r"^he so"),
    ("group", r"^nhom hang|^nhom san pham|^loai hang"),
    ("line_amount", r"^thanh tien sau giam|^thanh tien cua dong|^thanh tien\b(?!.*nhap)|^tien ban cua dong"),
    ("purchase_line_amount", r"^tien nhap cua ca dong|^thanh tien nhap"),
    ("purchase_total", r"^tong tien phieu nhap|^tong tien cua ca phieu nhap"),
    ("purchase_price", r"^gia nhap"),
    ("line_discount", r"^tien giam cua ca dong|^giam gia dong|^giam gia cua dong"),
    ("goods_total", r"^tong tien hang|^tien hang cua don"),
    ("order_discount", r"^giam gia cua ca don|^tien bot cho ca don|^giam gia don"),
    ("shipping", r"^phi giao"),
    ("order_total", r"^tong tien don|^so khach phai tra cho ca don|^khach can tra cho ca don|^tong tien ca don"),
    ("discount", r"^giam gia|^chiet khau"),
    ("unit_cost", r"^gia von"),
    ("unit_price", r"^don gia|^gia ban(?! hien tai| ap dung)|^gia 1"),
    ("current_price", r"^gia ban hien tai|^gia hien tai"),
    ("new_price", r"^gia ban ap dung|^gia moi"),
    ("old_price", r"^gia truoc lan doi|^gia cu"),
    ("paid", r"^da thanh toan|^khach da tra|^so tien khach tra"),
    ("debt", r"^con no|^khach con no"),
    ("pay_amount", r"^so tien\b"),
    ("rx", r"^thuoc ke don"),
    ("ingredient", r"^hoat chat"),
    ("name", r"^ten\b"),
    ("blank", r"^cot trong"),
]
ROLE_PATTERNS = [(role, re.compile(rx)) for role, rx in ROLE_RULES]


def column_role(description):
    text = fold(description)
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(text):
            return role
    return None


@dataclass
class FileSpec:
    path: str
    sheet: str | None = None  # README section "`X.xlsx`, sheet `Y`" covers one sheet
    columns: dict = field(default_factory=dict)  # header -> description
    roles: dict = field(default_factory=dict)  # header -> role

    def by_role(self, role):
        return [h for h, r in self.roles.items() if r == role]

    def has(self, *roles):
        return all(any(r == role for r in self.roles.values()) for role in roles)


@dataclass
class Readme:
    intro: str
    files: dict  # path -> FileSpec
    notes: list  # convention lines (Quy ước)
    data_end: dt.date | None
    conversions: list  # (item_name, big_unit, factor, base_unit)
    text: str


SECTION = re.compile(r"^## Cột trong `(.+?)`(?:,? sheet `(.+?)`)?\s*$", re.MULTILINE)
COLUMN = re.compile(r"^- \*\*(.+?)\*\*: (.+)$", re.MULTILINE)
CONVERSION = re.compile(r"^\s+- (.+?): 1 (.+?) = ([\d.,]+) (.+?)\s*$", re.MULTILINE)
DATA_END = re.compile(r"Dữ liệu có đến hết (\d{1,2})/(\d{1,2})/(\d{4})")


def parse_readme(text):
    parts = text.split("\n\n")
    intro = parts[1].strip() if len(parts) > 1 else ""
    files = {}
    for m in re.finditer(r"^- `(.+?)` \(", text, re.MULTILINE):
        files.setdefault(m.group(1), FileSpec(m.group(1)))
    sections = list(SECTION.finditer(text))
    for i, sec in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        nxt = text.find("\n## ", sec.end())
        if nxt != -1:
            end = min(end, nxt)
        body = text[sec.end() : end]
        name = sec.group(1) if not sec.group(2) else f"{sec.group(1)}#{sec.group(2)}"
        spec = files.setdefault(name, FileSpec(sec.group(1), sec.group(2)))
        whole = files.get(sec.group(1))
        if sec.group(2) and whole is not None and not whole.columns:
            del files[sec.group(1)]  # the workbook is described sheet by sheet
        for c in COLUMN.finditer(body):
            header, desc = c.group(1).strip(), c.group(2).strip()
            spec.columns[header] = desc
            spec.roles[header] = column_role(desc)
    notes = []
    m = re.search(r"^## Quy ước\s*$(.*)", text, re.MULTILINE | re.DOTALL)
    if m:
        notes = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    d = DATA_END.search(text)
    data_end = dt.date(int(d.group(3)), int(d.group(2)), int(d.group(1))) if d else None
    conversions = []
    for c in CONVERSION.finditer(text):
        try:
            factor = float(c.group(3).replace(",", "."))
        except ValueError:
            continue
        conversions.append((c.group(1).strip(), c.group(2).strip(), factor, c.group(4).strip()))
    return Readme(intro, files, notes, data_end, conversions, text)


# Header names seen in POS exports and hand-kept books, folded. Used when the
# README has no column description for a file (or for SQLite tables). Generic
# Vietnamese vocabulary, never shop-specific.
HEADER_ROLES = {
    "order_id": "don so|so don|ma hoa don|ma don hang|ma don|order id|ma bill|so hd|ma hd|ma chung tu|so chung tu|madon|madonhang|mahoadon|ma giao dich|so phieu ban|so hoa don",
    "line_no": "tt dong|no|no.|so thu tu dong|stt dong|dong",
    "order_seq": "stt|so thu tu|sothutu",
    "item_code": "ma h.hoa|ma hh|barcode|ma vach|ma hang|ma hh|ma sp|ma san pham|ma mon|id sp|masp|mahang|ma vat tu|sku|ma hang hoa|masanpham",
    "item_name": "item|ten hang|ten san pham|ten mon|mon|san pham|product|chi tiet hang|hang hoa|mat hang|ten hang hoa|ten sp|tensp|tenhang|tensanpham|hang",
    "customer_id": "ma khach mua|ma kh|ma khach|ma khach hang|customer id|ma kh mua|makh|makhachhang|so kh",
    "customer_name": "khach hang mua|khach hang (ten)|khach hang|ten kh|ten khach|ten khach hang|ten_kh|khach|doi tac|tenkhachhang",
    "phone": "dien thoai|so dien thoai|sdt|sodienthoai",
    "qty": "so luong|slg|sl|s.l|sl ban|soluong|so luong hang",
    "base_qty": "sl (dv co ban)|sl (dvt co ban)|so luong (dvt co ban)|so luong (don vi co ban)|so luong quy doi|sl quy doi|sl co ban|sl quy ra le|quy ra dv le|so luong le|slquydoi|soluongquydoi",
    "unit": "don vi tinh|dvt|d.vi|don vi|dv|dv tinh|donvitinh|dvt co ban|quy cach",
    "unit_price": "don gia|gia niem yet|gia ban le|gia 1 dv|dongia|giaban|giabanle|gia don vi|giadonvi",
    "line_amount": "thanh tien|tri gia|tien ban|thanhtien|trigia|amount",
    "discount": "giam gia|giam|tien giam|khuyen mai|chiet khau|ck|giamgia|bot",
    "unit_cost": "gia von|gia von luc ban|giavonlucban|giavon|gia von hien tai",
    "date": "ngay hd|ngay hoa don|ngay ban|ngay|thoi gian|ngay giao dich|ngaychungtu|ngayban|ngay lap|ngay dat",
    "datetime": "thoi gian tao|gio tao don|thoi diem|ngay gio|thoigiantao|ngaygioban|thoi gian lap|thoigianlap|ngay gio lap",
    "status": "trang thai|trang thai giao|tinh trang|xu ly|ket qua|trangthai",
    "goods_total": "tong tien hang|tien hang truoc giam|tong tien hang truoc giam|tongtienhang",
    "order_total": "tong sau giam gia|tong tien|tong bill|tong thanh toan|phai thu|can tra|tongthanhtoan|tong don",
    "paid": "khach da tra|da thu|da tra|da tra|khach tra|da thanh toan|khachdatra",
    "debt": "con no|cong no|chua tra|du no|con lai|conno",
    "payment_method": "phuong thuc thanh toan|phuong thuc tt|pttt|hinh thuc thanh toan|httt|phuongthucthanhtoan|phuong thuc|phuongthuc",
    "channel": "kenh ban|nguon don|nguon|kenh|ban qua|kenhban",
    "staff": "nguoi ban|ma nv ban|nhan vien|ma nv|manhanvien|ma nhan vien",
    "note": "ghi chu|ghichu|mo ta",
    "shipping": "phi ship|phi giao|phi giao hang|tien ship|phi van chuyen|phivanchuyen",
    "group": "nhom hang|nhom hang(3 cap)|nhom hang 3 cap|phan loai|nhom|nhom san pham|nhomhang|danh muc|danhmuc|nganh hang",
    "current_price": "gia ban|gia|gia le|gia hien tai|gia ban theo don vi quy doi",
    "stock_qty": "ton cuoi ky|ton kho|ton cuoi|ton thuc|toncuoi|ton",
    "factor": "quy doi|bang|he so|ty le quy doi|ti le quy doi|hesoquydoi|he so quy doi",
    "unit_base": "don vi co so|donvicoso|don vi co ban|dv co ban",
    "purchase_base_qty": "sl nhap quy doi|slnhapquydoi|sl nhap (dvt co ban)|sl nhap (dv co ban)|so luong nhap quy doi",
    "return_qty": "sl tra quy doi|sltraquydoi|sl tra (dvt co ban)|sl tra (dv co ban)|sl tra",
    "purchase_qty": "sl nhap|so luong nhap|slnhap",
    "payment_id": "phieu thu|ma phieu thu|maphieuthu",
    "pay_amount": "gia tri|so tien thu|tien thu|sotienthu",
    "pay_time": "thoi gian thu|thoigianthu|ngay thu|ngay gio thu|ngaygiothu|ngay tra",
    "purchase_id": "ma nhap hang|ma phieu nhap|maphieunhap|so phieu nhap",
    "purchase_total": "can tra ncc|tong tien phieu|tonggiatrinhap",
    "supplier": "ma ncc|nha cung cap|mancc",
    "batch_id": "lo|so lo|ma lo|malo",
    "expiry": "hsd|han su dung|handung|hansudung",
    "rx": "ke don|ban theo don|bantheodon",
    "ingredient": "hoat chat|hoatchat|thanh phan chinh|thanhphanchinh|thanh phan",
    "customer_group": "nhom khach|hang thanh vien|loai khach hang|kieu khach|doi tuong kh|hang khach",
}
HEADER_LOOKUP = {}
for _role, _names in HEADER_ROLES.items():
    for _name in _names.split("|"):
        HEADER_LOOKUP.setdefault(_name, _role)


def header_role(header):
    """Role from a header name alone: exact vocabulary hit, else the description
    rules applied to the header text (camelCase split)."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(header)).replace("_", " ")
    text = re.sub(r"\s+", " ", fold(words)).strip()
    compact = text.replace(" ", "")
    for candidate in (text, compact, re.sub(r"\s*\((?:d|vnd|dong)\)\s*", " ", text).strip(), re.sub(r"\s*\(.*?\)\s*", " ", text).strip()):
        if candidate in HEADER_LOOKUP:
            return HEADER_LOOKUP[candidate]
    return column_role(text)


NOTE_COLUMN = re.compile(r"\*\*(.+?)\*\*\s+là\s+([^;.]+)")


def note_descriptions(notes):
    """Column descriptions written in the README conventions:
    '**Tong tien** là số khách phải trả cho cả đơn' -> {'Tong tien': '...'}."""
    out = {}
    for line in notes:
        for m in NOTE_COLUMN.finditer(line):
            out.setdefault(m.group(1).strip(), m.group(2).strip())
    return out


# Roles whose meaning depends on the kind of file they sit in.
CONTEXT_ROLES = {
    "purchases": {"status": "purchase_status", "date": "purchase_date", "datetime": "purchase_date", "qty": "purchase_qty",
                  "unit_price": "purchase_price", "line_amount": "purchase_line_amount", "order_total": "purchase_total",
                  "unit": "unit_purchase"},
    "payments": {"date": "pay_time", "datetime": "pay_time", "line_amount": "pay_amount", "order_total": "pay_amount"},
    "orders": {"discount": "order_discount"},
    "lines": {"discount": "line_discount"},
    "products": {"unit": "unit_base", "unit_price": "current_price"},
    "stock": {"unit": "unit_base"},
    "conversions": {"unit": "unit_big", "current_price": None},
    # "Hạng" (tier) folds to the same text as "Hàng" (item).
    "customers": {"item_name": "customer_group", "group": "customer_group", "status": None},
}
