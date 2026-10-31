"""Profile DataMate with Python standard library; source data stays read-only."""
import csv
import hashlib
import json
import sqlite3
import statistics
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def normalize(text):
    return ' '.join(''.join(c for c in unicodedata.normalize('NFD', text.lower().replace('đ', 'd')) if unicodedata.category(c) != 'Mn').split())

def workbook(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets = []
        for s in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            target = rels[s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows, dimension = [], None
            with z.open(target) as f:
                for event, elem in ET.iterparse(f, events=('end',)):
                    if elem.tag.endswith('}dimension'):
                        dimension = elem.attrib.get('ref')
                    if elem.tag == '{%s}row' % NS['s']:
                        values = []
                        for c in elem.findall('s:c', NS):
                            v = c.find('s:v', NS)
                            val = v.text if v is not None else ''
                            if c.attrib.get('t') == 's' and val:
                                val = shared[int(val)]
                            elif c.attrib.get('t') == 'inlineStr':
                                inline = c.find('s:is', NS)
                                val = ''.join(inline.itertext()) if inline is not None else ''
                            values.append({'cell': c.attrib.get('r'), 'value': val})
                        rows.append(values)
                        elem.clear()
                        if len(rows) == 6:
                            break
            sheets.append({'name': s.attrib['name'], 'declared_dimension': dimension, 'first_rows': rows})
        return sheets

def main():
    OUT.mkdir(exist_ok=True)
    result = {'splits': {}, 'inventory': [], 'errors': []}
    questions = {}
    for split in ['train', 'public_test']:
        base = ROOT / 'data' / split
        qs = questions[split] = load(base / 'questions.jsonl')
        businesses = sorted(p.name for p in (base / 'data').iterdir() if p.is_dir())
        ids = [q['id'] for q in qs]
        contexts = Counter((q['business_id'], q['as_of_date'], normalize(q['question'])) for q in qs)
        lengths = [len(q['question']) for q in qs]
        files = [p for p in (base / 'data').rglob('*') if p.is_file()]
        info = {
            'questions': len(qs), 'businesses': len(businesses),
            'questions_per_business': dict(sorted(Counter(q['business_id'] for q in qs).items())),
            'as_of_date_range': [min(q['as_of_date'] for q in qs), max(q['as_of_date'] for q in qs)],
            'question_chars': {'min': min(lengths), 'median': statistics.median(lengths), 'max': max(lengths)},
            'duplicate_ids': len(ids) - len(set(ids)),
            'normalized_duplicate_contexts': sum(n-1 for n in contexts.values()),
            'missing_businesses': sorted({q['business_id'] for q in qs} - set(businesses)),
            'missing_fields': sum(any(not q.get(k) for k in ['id','business_id','question','as_of_date']) for q in qs),
            'source_files': len(files), 'source_bytes': sum(p.stat().st_size for p in files),
            'file_types': dict(Counter(p.suffix for p in files)),
        }
        if (base / 'answers.jsonl').exists():
            ans = load(base / 'answers.jsonl')
            aids = [a['id'] for a in ans]
            info.update(answer_types=dict(Counter(a['type'] for a in ans)), answers=len(ans), duplicate_answer_ids=len(aids)-len(set(aids)),
                        questions_without_answer=len(set(ids)-set(aids)), answers_without_question=len(set(aids)-set(ids)),
                        answers_with_alias=sum(bool(a.get('alias')) for a in ans))
            validators = {'number': lambda x: type(x) in (int,float), 'percent': lambda x: type(x) in (int,float),
                          'list': lambda x: isinstance(x,list), 'table': lambda x: isinstance(x,dict) and 'columns' in x and 'rows' in x,
                          'string': lambda x: isinstance(x,str), 'date': lambda x: isinstance(x,str), 'boolean': lambda x: type(x) is bool}
            info['answer_shape_errors'] = [a['id'] for a in ans if a['type'] in validators and not validators[a['type']](a.get('answer'))]
            info['unanswerable_examples'] = [dict(question=next(q['question'] for q in qs if q['id']==a['id']), label=a) for a in ans if a['type']=='unanswerable'][:8]
        result['splits'][split] = info
        for p in sorted(files):
            entry = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size}
            try:
                if p.suffix == '.xlsx':
                    entry['sheets'] = workbook(p)
                elif p.suffix == '.csv':
                    with p.open(encoding='utf-8-sig', newline='') as f:
                        reader = csv.reader(f)
                        entry['header'] = next(reader)
                        count, widths = 0, Counter()
                        for row in reader:
                            count += 1
                            widths[len(row)] += 1
                        entry.update(rows=count, row_widths=dict(widths))
                elif p.suffix == '.sqlite':
                    with sqlite3.connect(p.as_uri() + '?mode=ro', uri=True) as conn:
                        entry['tables'] = []
                        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                            safe = '"' + name.replace('"','""') + '"'
                            entry['tables'].append({'name': name, 'rows': conn.execute('SELECT count(*) FROM '+safe).fetchone()[0], 'columns': [r[1:3] for r in conn.execute('PRAGMA table_info('+safe+')')]})
            except Exception as e:
                entry['error'] = str(e)
                result['errors'].append({'path': entry['path'], 'error': str(e)})
            result['inventory'].append(entry)
    a,b = questions['train'],questions['public_test']
    result['overlap'] = {'question_ids': len({q['id'] for q in a}&{q['id'] for q in b}),
                         'business_ids': len({q['business_id'] for q in a}&{q['business_id'] for q in b}),
                         'normalized_question_texts': len({normalize(q['question']) for q in a}&{normalize(q['question']) for q in b}),
                         'calendar_identical': hashlib.sha256((ROOT/'data/train/calendar.csv').read_bytes()).digest()==hashlib.sha256((ROOT/'data/public_test/calendar.csv').read_bytes()).digest()}
    (OUT/'data_profile.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},ensure_ascii=False,indent=2))
    print('Inventory files:',len(result['inventory']))

if __name__ == '__main__':
    main()
