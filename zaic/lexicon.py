"""Question vocabulary the rule parser accounts for.

After entity names and time phrases are masked, every remaining word of a
question must be in this list (or in the extra list of the matched family);
otherwise the semantic layer declines and the question goes to the next path.
The words are generic Vietnamese question, sales and unit vocabulary in folded
form (no accents), never shop- or question-specific. Words that carry a
condition the rules do not implement ("nữ", "trễ", a street name) are left
out on purpose: answering without them is how a filter gets dropped.
"""

# Fillers, pronouns, particles, polite forms, chat spellings.
FILLER = """
a ah ad ai anh ba bac ban be ben bi biet bo bon can cac cai check cho chi co coi con cua cung da dang dem
do dum duoc e ee em ghi gi gium giup ha hai hay ho hoi hok hong j k ke khoong khong la lai len liet luon ma minh
mot moi nao nay nha nhe nhi nhung noi o oi r ra rieng roi sau shop ta tat thi theo thuoc tiem toi tooi
si trong tu tui tung tuoi va vao vay ve voi vui xem xet xin z
alo chu di dza gap giua hen het muon ne nhanh nho nhoa oi thanks ua xiu za zi 1
"""
# Measures, comparisons and the words of the implemented question families.
BUSINESS = """
am ban bang bao binh cao chay chenh chiem chieu chuyen ck cod cong cu dat den dien dip doanh doi don dong
gia giam gian gioi gop hang hien hon hom hot huy it kem ket khac khach khoan ky lan le lich lien loai luong
luot mang mat may mua ngay nguoi nhan nhat nhau nhieu nhom pham phai phan qua quan sang san size so tang
tet thai thang thanh thay thoi thu thuc ti tien tiep tieu tinh tm tong top tra tram tren tri tro trung truoc
truong tru tuc ty uong vi xuat thap xuong
"""
# Units of measure; the parser only uses a unit the shop actually sells in.
UNITS = """
bao binh bich cai can cay chai chiec con cuon d dia dong g goi gram hop hu kg ky l lang lit lo loc lon ly
m m2 m3 ml mon ong cuc phan qua tam thung to trai tuyp vien vi xe
"""
# Words introducing a filter; the value itself must be matched as an entity.
FILTER_WORDS = "ca chat hoat kenh ke nhom toa trang th"

KNOWN_WORDS = frozenset((FILLER + BUSINESS + UNITS + FILTER_WORDS).split())

# Words that belong to one question family only. "nhập" is expected in a
# purchase question and is an unhandled condition anywhere else.
_PURCHASE = "nhap phieu vao bo het ton chi"
_DEBT = "no con thieu chua tra het chiu thu hoi kem tuoi ghi them bot khoan cach chia loi lai"
_STOCK = "ton kho con lai cuoi luc het nguyen chan nam von"
_UNSOLD = "e han muc cham ai giai doan van dang hang"
_PROFIT = "lai loi gop nhuan loi"
FAMILY_WORDS = {
    "stock_qty": frozenset(_STOCK.split()),
    "stock_value": frozenset(_STOCK.split()),
    "unsold_count": frozenset(_UNSOLD.split()),
    "gross_profit": frozenset(_PROFIT.split()),
    "most_profitable_item": frozenset(_PROFIT.split()),
    **{name: frozenset(_DEBT.split()) for name in ("debt_total", "debt_customers", "debt_top", "debt_aging", "debt_collected", "debt_change")},
    "purchase_total": frozenset(_PURCHASE.split()),
    "purchase_avg": frozenset(_PURCHASE.split()),
    "top_supplier": frozenset((_PURCHASE + " nha cung cap ncc").split()),
    "purchase_minus_sales": frozenset((_PURCHASE + " lay tru lech").split()),
}
