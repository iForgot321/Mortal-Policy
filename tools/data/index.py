"""D3 + D4 (04a §3–4): index the converted games and apply reason-coded hygiene.

Inputs: mjai/YYYY/MM/*.json.gz (convert.py) and the source DBs in
db/tenhou-original-xml/YYYY.db. Outputs, per year then combined:

  meta/games.parquet       one row per game
  meta/seats.parquet       one row per game x seat
  meta/quarantine.parquet  (log_id, seat, kyoku, code, detail); seat/kyoku -1 = all
  reports/quarantine.md, reports/db_schema.txt, reports/index.log

Final ranks and scores come from libriichi's own code path (Grp), never ad hoc,
so the GRP, the reward and the player scores agree. Q5 replays every game with
GameplayLoader.validate_gz_log_files (the training loader's code path, minus
featurization, plus a label-legality check).

Quarantine codes (04a §4): Q1 decode/parse, Q2 placeholder names, Q3 structure,
Q4 unexplained score-continuity break, Q4s explained break (see below), Q5
libriichi replay failure, Q6 disconnect window (kyoku-level mask, all seats), Q7
duplicate, Q8 known unauthorized-bot account (seat-level). Whether a code drops
the game is decided by params.toml, not here.

Q4s: in a four-riichi abortive draw the converter emits the fourth `reach` and
its discard but no `reach_accepted`, while Tenhou still takes the deposit, so the
next kyoku starts 1000 short for that seat. Classified by hand on the 2009 shard
(all 8 breaks); labels, GRP inputs and final ranks are unaffected.

Usage: python tools/data/index.py [--mj ~/mj-data] [--workers 10] [--years 2009 2010 ...]
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from urllib.parse import unquote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'mortal'))

UTILITY_CENTI = np.array([90, 50, -10, -130])  # u x 100 (02 §2, D2)
PLACEHOLDER = re.compile(r'^EXAMPLE\d*$')
CHUNK = 2000


def nfc(s):
    return unicodedata.normalize('NFC', s)


def log_id_of(path):
    return os.path.basename(path)[:-len('.json.gz')]


# ---------------------------------------------------------------- mjai pass

def parse_mjai(path):
    """Per-game facts from the mjai stream. Never raises: failures become Q1."""
    log_id = log_id_of(path)
    row = {'log_id': log_id, 'q': []}
    try:
        with open(path, 'rb') as f:
            raw = gzip.decompress(f.read())
        row['content_sha256'] = hashlib.sha256(raw).hexdigest()
        events = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001 - every decode failure is a Q1
        row['q'].append((-1, -1, 'Q1', f'{type(e).__name__}: {e}'[:200]))
        return row

    row['n_events'] = len(events)
    first, last = events[0] if events else {}, events[-1] if events else {}
    names = first.get('names') if first.get('type') == 'start_game' else None
    row['names'] = [nfc(n) for n in names] if names and len(names) == 4 else None
    row['aka_flag'] = bool(first.get('aka_flag', False))

    structural = []
    if first.get('type') != 'start_game' or row['names'] is None:
        structural.append('missing start_game/names')
    if last.get('type') != 'end_game':
        structural.append('missing end_game')
    if '-00a9-' not in log_id:
        structural.append('type_code != 00a9')
    if not row['aka_flag']:
        structural.append('aka_flag false')
    if structural:
        row['q'].append((-1, -1, 'Q3', '; '.join(structural)))

    if row['names'] is not None:
        bad = [n for n in row['names'] if not n.strip() or PLACEHOLDER.match(n)]
        if bad or len(set(row['names'])) < 4:
            row['q'].append((-1, -1, 'Q2', f'names {row["names"]}'))

    # per-seat counts for the style-panel bands, and score continuity (Q4)
    riichi = [0] * 4
    agari = [0] * 4
    houjuu = [0] * 4
    calls = [0] * 4
    n_kyoku = 0
    expect = None
    deltas = np.zeros(4, dtype=np.int64)
    pending = np.zeros(4, dtype=np.int64)  # reach without reach_accepted
    reaches = 0
    last_end = None
    breaks, explained = [], []
    for ev in events:
        t = ev.get('type')
        if t == 'start_kyoku':
            scores = np.asarray(ev['scores'], dtype=np.int64)
            if expect is not None and not (expect + deltas == scores).all():
                diff = scores - expect - deltas
                msg = f'k{n_kyoku}: expected {(expect + deltas).tolist()} got {scores.tolist()}'
                if last_end == 'ryukyoku' and reaches == 4 and (diff == -1000 * pending).all():
                    explained.append(msg)
                else:
                    breaks.append(msg)
            expect = scores
            deltas[:] = 0
            pending[:] = 0
            reaches = 0
            n_kyoku += 1
        elif t == 'reach':
            pending[ev['actor']] = 1
            reaches += 1
        elif t == 'reach_accepted':
            riichi[ev['actor']] += 1
            deltas[ev['actor']] -= 1000
            pending[ev['actor']] = 0
        elif t in ('hora', 'ryukyoku'):
            last_end = t
            if ev.get('deltas') is not None:
                deltas += np.asarray(ev['deltas'], dtype=np.int64)
            if t == 'hora':
                agari[ev['actor']] += 1
                if ev['target'] != ev['actor']:
                    houjuu[ev['target']] += 1
        elif t in ('chi', 'pon', 'daiminkan'):
            calls[ev['actor']] += 1
    if breaks:
        row['q'].append((-1, -1, 'Q4', ' | '.join(breaks)[:500]))
    if explained:
        row['q'].append((-1, -1, 'Q4s', 'four-riichi abortive draw, no 4th reach_accepted: '
                         + ' | '.join(explained)[:450]))
    row.update(n_kyoku=n_kyoku, riichi=riichi, agari=agari, houjuu=houjuu, calls=calls)
    return row


def parse_mjai_chunk(paths):
    return [parse_mjai(p) for p in paths]


# ---------------------------------------------------------------- DB pass

def parse_mjlog(xml_bytes):
    """Names/dan/rate at game start and disconnect windows from the mjlog.

    Disconnect state per seat: BYE sets it; a later UN carrying only that
    seat's name clears it. A kyoku is affected if any seat is disconnected at
    any point during it (at its INIT, or by a BYE inside it). A BYE before the
    first INIT affects kyoku 0 and is flagged explicitly.
    """
    root = ET.fromstring(xml_bytes)
    out = {'names_xml': None, 'dan': None, 'rate': None, 'go_type': None,
           'bye_seats': [], 'affected_kyoku': [], 'bye_before_start': False,
           'n_kyoku_xml': 0, 'anomalies': []}
    kyoku = -1
    disconnected = [False] * 4
    affected = set()
    bye_seats = set()
    for el in root:
        tag, a = el.tag, el.attrib
        if tag == 'GO':
            out['go_type'] = int(a.get('type', -1))
        elif tag == 'UN':
            if 'dan' in a and out['names_xml'] is None:
                out['names_xml'] = [nfc(unquote(a.get(f'n{i}', ''))) for i in range(4)]
                out['dan'] = [int(x) for x in a['dan'].split(',')]
                out['rate'] = [float(x) for x in a['rate'].split(',')]
            else:
                for i in range(4):
                    if f'n{i}' in a:
                        disconnected[i] = False
        elif tag == 'BYE':
            who = int(a['who'])
            disconnected[who] = True
            bye_seats.add(who)
            if kyoku >= 0:
                affected.add(kyoku)
            else:
                out['bye_before_start'] = True
        elif tag == 'INIT':
            kyoku += 1
            if any(disconnected):
                affected.add(kyoku)
    out['n_kyoku_xml'] = kyoku + 1
    out['bye_seats'] = sorted(bye_seats)
    out['affected_kyoku'] = sorted(affected)
    if out['names_xml'] is None:
        out['anomalies'].append('no initial UN')
    return out


def join_db_chunk(args):
    db_path, log_ids = args
    res = {}
    if not os.path.exists(db_path):
        return res
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        for lid in log_ids:
            hit = con.execute('select log from logs where id = ?', (lid,)).fetchone()
            if hit is None or hit[0] is None:
                continue
            try:
                res[lid] = parse_mjlog(gzip.decompress(hit[0]))
            except Exception as e:  # noqa: BLE001
                res[lid] = {'error': f'{type(e).__name__}: {e}'[:200]}
    finally:
        con.close()
    return res


# ---------------------------------------------------------------- libriichi pass

def grp_and_validate(paths, log):
    """Final ranks/scores via Grp and Q5 via the validator, per file."""
    from libriichi.dataset import GameplayLoader, Grp

    loader = GameplayLoader(version=4, oracle=False)
    out = {}
    for i in range(0, len(paths), 10_000):
        chunk = paths[i:i + 10_000]
        try:
            grps = Grp.load_gz_log_files(chunk)
            grp_res = [(g.take_rank_by_player(), g.take_final_scores(), None) for g in grps]
        except Exception:  # one bad file fails the batch: retry file by file
            grp_res = []
            for p in chunk:
                try:
                    g = Grp.load_gz_log_files([p])[0]
                    grp_res.append((g.take_rank_by_player(), g.take_final_scores(), None))
                except Exception as e:  # noqa: BLE001
                    grp_res.append((None, None, str(e).replace('\n', ' | ')[:300]))
        val = loader.validate_gz_log_files(chunk)
        for p, (ranks, finals, gerr), (verr, n_dec) in zip(chunk, grp_res, val):
            out[log_id_of(p)] = (ranks, finals, gerr, verr, n_dec)
        log(f'    libriichi {i + len(chunk)}/{len(paths)}')
    return out


# ---------------------------------------------------------------- assembly

def index_year(year, mj, pool, workers, bots, log):
    paths = sorted(glob.glob(os.path.join(mj, 'mjai', str(year), '*', '*.json.gz')))
    if not paths:
        return None
    t0 = time.time()
    chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
    games = [r for rows in pool.imap(parse_mjai_chunk, chunks) for r in rows]
    log(f'  {year}: mjai parsed {len(games)} in {time.time() - t0:.0f}s')

    db_path = os.path.join(mj, 'db', 'tenhou-original-xml', f'{year}.db')
    ids = [g['log_id'] for g in games]
    xml = {}
    for part in pool.imap(join_db_chunk, [(db_path, ids[i:i + CHUNK]) for i in range(0, len(ids), CHUNK)]):
        xml.update(part)
    log(f'  {year}: db joined {len(xml)}/{len(ids)} in {time.time() - t0:.0f}s')

    lib = grp_and_validate(paths, log)
    log(f'  {year}: libriichi pass done in {time.time() - t0:.0f}s')

    g_cols = {k: [] for k in (
        'log_id', 'ts', 'year', 'month', 'type_code', 'aka_flag', 'names', 'n_kyoku', 'n_events',
        'final_scores', 'final_ranks', 'content_sha256', 'xml_found', 'go_type', 'dan', 'rate',
        'bye_seats', 'bye_before_start', 'affected_kyoku', 'n_kyoku_xml', 'n_decisions')}
    s_cols = {k: [] for k in (
        'log_id', 'seat', 'name', 'final_score', 'rank', 'u', 'dan_at_start', 'rate_at_start',
        'disconnected', 'riichi', 'agari', 'houjuu', 'calls')}
    quarantine = []

    for g in games:
        lid = g['log_id']
        q = list(g['q'])
        x = xml.get(lid)
        ranks, finals, gerr, verr, n_dec = lib.get(lid, (None, None, 'missing', 'missing', 0))
        if gerr or verr:
            if not any(code == 'Q1' for _, _, code, _ in q):
                q.append((-1, -1, 'Q5', (verr or gerr)[:300]))
        if x is not None and 'error' not in x:
            if g.get('names') and x['names_xml'] and x['names_xml'] != g['names']:
                x['anomalies'].append(f'names differ: xml {x["names_xml"]}')
            if x['affected_kyoku']:
                if x['n_kyoku_xml'] != g.get('n_kyoku'):
                    q.append((-1, -1, 'Q6', f'disconnect with kyoku count mismatch '
                                            f'(xml {x["n_kyoku_xml"]}, mjai {g.get("n_kyoku")}): whole game'))
                else:
                    for k in x['affected_kyoku']:
                        q.append((-1, k, 'Q6', f'bye seats {x["bye_seats"]}'))
            for a in x['anomalies']:
                q.append((-1, -1, 'XML', a))
        elif x is not None:
            q.append((-1, -1, 'XML', x['error']))
        names = g.get('names') or [''] * 4
        for seat, name in enumerate(names):
            if name in bots:
                q.append((seat, -1, 'Q8', name))
        quarantine.extend((lid, s, k, c, d) for s, k, c, d in q)

        xml_ok = x is not None and 'error' not in x
        g_cols['log_id'].append(lid)
        g_cols['ts'].append(lid[:10])
        g_cols['year'].append(int(lid[:4]))
        g_cols['month'].append(int(lid[4:6]))
        g_cols['type_code'].append(lid.split('-')[1] if '-' in lid else '')
        g_cols['aka_flag'].append(g.get('aka_flag'))
        g_cols['names'].append(g.get('names'))
        g_cols['n_kyoku'].append(g.get('n_kyoku'))
        g_cols['n_events'].append(g.get('n_events'))
        g_cols['final_scores'].append(list(finals) if finals is not None else None)
        g_cols['final_ranks'].append(list(ranks) if ranks is not None else None)
        g_cols['content_sha256'].append(g.get('content_sha256'))
        g_cols['xml_found'].append(xml_ok)
        g_cols['go_type'].append(x['go_type'] if xml_ok else None)
        g_cols['dan'].append(x['dan'] if xml_ok else None)
        g_cols['rate'].append(x['rate'] if xml_ok else None)
        g_cols['bye_seats'].append(x['bye_seats'] if xml_ok else None)
        g_cols['bye_before_start'].append(x['bye_before_start'] if xml_ok else None)
        g_cols['affected_kyoku'].append(x['affected_kyoku'] if xml_ok else None)
        g_cols['n_kyoku_xml'].append(x['n_kyoku_xml'] if xml_ok else None)
        g_cols['n_decisions'].append(n_dec)

        if g.get('names') is None or ranks is None:
            continue
        for seat in range(4):
            s_cols['log_id'].append(lid)
            s_cols['seat'].append(seat)
            s_cols['name'].append(names[seat])
            s_cols['final_score'].append(int(finals[seat]))
            s_cols['rank'].append(int(ranks[seat]))
            s_cols['u'].append(int(UTILITY_CENTI[ranks[seat]]) / 100)
            s_cols['dan_at_start'].append(x['dan'][seat] if xml_ok and x['dan'] else None)
            s_cols['rate_at_start'].append(x['rate'][seat] if xml_ok and x['rate'] else None)
            s_cols['disconnected'].append(bool(xml_ok and seat in x['bye_seats']))
            s_cols['riichi'].append(g['riichi'][seat])
            s_cols['agari'].append(g['agari'][seat])
            s_cols['houjuu'].append(g['houjuu'][seat])
            s_cols['calls'].append(g['calls'][seat])

    q_cols = {'log_id': [], 'seat': [], 'kyoku': [], 'code': [], 'detail': []}
    for lid, s, k, c, d in sorted(quarantine):
        q_cols['log_id'].append(lid)
        q_cols['seat'].append(s)
        q_cols['kyoku'].append(k)
        q_cols['code'].append(c)
        q_cols['detail'].append(d)
    log(f'  {year}: assembled in {time.time() - t0:.0f}s')
    return pa.table(g_cols), pa.table(s_cols), pa.table(q_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mj', default=os.path.expanduser('~/mj-data'))
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--years', type=int, nargs='*', default=None)
    ap.add_argument('--combine-only', action='store_true',
                    help='skip per-year indexing; rebuild the combined tables from meta/by_year')
    args = ap.parse_args()

    mj = args.mj
    meta = os.path.join(mj, 'meta')
    reports = os.path.join(mj, 'reports')
    for d in ('games', 'seats', 'quarantine'):
        os.makedirs(os.path.join(meta, 'by_year', d), exist_ok=True)
    os.makedirs(reports, exist_ok=True)
    logf = open(os.path.join(reports, 'index.log'), 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    bots_file = os.path.join(mj, 'names', 'excludes_bots.txt')
    bots = set()
    if os.path.exists(bots_file):
        with open(bots_file, encoding='utf-8') as f:
            bots = {nfc(line.split('\t')[0].strip()) for line in f
                    if line.strip() and not line.startswith('#')}
    log(f'index start; {len(bots)} bot accounts in excludes')

    years = [] if args.combine_only else \
        args.years or sorted(int(y) for y in os.listdir(os.path.join(mj, 'mjai')) if y.isdigit())
    with Pool(args.workers) as pool:
        for year in years:
            res = index_year(year, mj, pool, args.workers, bots, log)
            if res is None:
                continue
            for name, table in zip(('games', 'seats', 'quarantine'), res):
                pq.write_table(table, os.path.join(meta, 'by_year', name, f'{year}.parquet'))

    # combine; Q7 duplicates need the whole corpus
    tables = {}
    for name in ('games', 'seats', 'quarantine'):
        files = sorted(glob.glob(os.path.join(meta, 'by_year', name, '*.parquet')))
        # years without source-DB rows (2026) have all-null DB columns, which
        # Arrow types as `null`; promote them to the other years' types
        tables[name] = pa.concat_tables([pq.read_table(f) for f in files], promote_options='default')
    games = tables['games']
    dup_rows = []
    seen_id, seen_sha = set(), set()
    for lid, sha in zip(games['log_id'].to_pylist(), games['content_sha256'].to_pylist()):
        if lid in seen_id or (sha is not None and sha in seen_sha):
            dup_rows.append((lid, -1, -1, 'Q7', 'duplicate log_id or content'))
        seen_id.add(lid)
        if sha is not None:
            seen_sha.add(sha)
    q = tables['quarantine']
    if dup_rows:
        q = pa.concat_tables([q, pa.table({k: [r[i] for r in dup_rows] for i, k in enumerate(q.column_names)},
                                          schema=q.schema)])
    pq.write_table(games, os.path.join(meta, 'games.parquet'))
    pq.write_table(tables['seats'], os.path.join(meta, 'seats.parquet'))
    pq.write_table(q, os.path.join(meta, 'quarantine.parquet'))
    write_quarantine_report(games, q, os.path.join(reports, 'quarantine.md'))
    write_db_schema(mj, os.path.join(reports, 'db_schema.txt'))
    log(f'index done: {games.num_rows} games, {tables["seats"].num_rows} seats, {q.num_rows} quarantine rows')


def write_quarantine_report(games, q, path):
    years = games['year'].to_numpy()
    lid_year = dict(zip(games['log_id'].to_pylist(), years.tolist()))
    n_games = {int(y): int(c) for y, c in zip(*np.unique(years, return_counts=True))}
    kyoku = games['n_kyoku'].to_numpy(zero_copy_only=False)
    kyoku = np.where(np.equal(kyoku, None), 0, kyoku).astype(np.int64)
    n_kyoku = {y: int(kyoku[years == y].sum()) for y in n_games}
    counts = {}
    for lid, kyo, code in zip(q['log_id'].to_pylist(), q['kyoku'].to_pylist(), q['code'].to_pylist()):
        y = lid_year.get(lid, int(lid[:4]))
        unit = 'kyoku' if (code == 'Q6' and kyo >= 0) else 'seat' if code == 'Q8' else 'game'
        key = (y, code, unit)
        counts[key] = counts.get(key, 0) + 1
    codes = sorted({(c, u) for _, c, u in counts})
    with open(path, 'w') as f:
        f.write('# Quarantine counts per year (04a §4)\n\n'
                'Game-level codes count games; Q6 kyoku rows count masked kyoku (all seats); '
                'Q8 counts seats. Q4 and XML rows are flags for inspection, not drops.\n\n')
        f.write('| Year | Games | Kyoku | ' + ' | '.join(f'{c} ({u})' for c, u in codes) + ' |\n')
        f.write('|---|---|---|' + '---|' * len(codes) + '\n')
        for y in sorted(n_games):
            cells = []
            for c, u in codes:
                n = counts.get((y, c, u), 0)
                denom = n_kyoku[y] if u == 'kyoku' else n_games[y] * (4 if u == 'seat' else 1)
                cells.append(f'{n} ({100 * n / max(denom, 1):.2f}%)')
            f.write(f'| {y} | {n_games[y]} | {n_kyoku[y]} | ' + ' | '.join(cells) + ' |\n')


def write_db_schema(mj, path):
    with open(path, 'w') as f:
        for db in sorted(glob.glob(os.path.join(mj, 'db', 'tenhou-original-xml', '*.db'))):
            con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
            f.write(f'== {os.path.basename(db)}\n')
            for (sql,) in con.execute("select sql from sqlite_master where sql is not null"):
                f.write(sql + ';\n')
            con.close()


if __name__ == '__main__':
    main()
