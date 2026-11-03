"""Read-only independent evidence check; run with .venv/bin/python notes/validate_saved_results.py."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

root = Path(__file__).resolve().parents[1]
base = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'd764b45'], cwd=root, text=True).splitlines()
protected = [p for p in base if p.startswith('tests/')] + ['src/dis_tt/problems/toy.py', 'AGENTS.md', 'docs/theory.md']
for p in protected:
    assert (root / p).read_bytes() == subprocess.check_output(['git', 'show', f'd764b45:{p}'], cwd=root), p
print(f'PASS: {len(protected)} original test/fixture/protected files byte-identical to d764b45')
records = {}
for i in range(1, 7):
    paths = list((root / 'notes/T9_results').glob(f'exp{i}_*.json'))
    assert len(paths) == 1
    p = paths[0]
    c = p.with_suffix('.csv')
    for f in (p, c):
        assert f.read_bytes() == (root / 'results' / f.name).read_bytes(), f
    j = json.loads(p.read_text())
    rows = list(csv.DictReader(c.open()))
    records[i] = (j, rows)
    if 'rows' in j:
        assert len(rows) == len(j['rows'])
        for row, raw in zip(rows, j['rows']):
            for k, v in raw.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    np.testing.assert_allclose(float(row[k]), v, rtol=5.1e-6, atol=0)
    print(f'PASS: {p.stem}: {len(rows)} CSV rows; both saved copies identical; JSON SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}')
j, rows = records[1]
for name, info in j.items():
    actual = max(float(r['sin_theta_max']) for r in rows if r['problem'] == name)
    np.testing.assert_allclose(actual, info['max_sin_r2_8'], rtol=5.1e-6)
j, _ = records[2]
assert j['n_test_le_upper'] == sum(r['test_error'] <= r['upper_unprojected'] for r in j['rows']) == 10
assert j['n_test_le_upper_plus_sampling'] == 12
assert all(r['lower_unprojected'] <= r['train_error'] <= r['upper_unprojected'] for r in j['rows'])
assert j['max_identity_rel_gap'] < 4.5e-11
j, _ = records[3]
for rank in (8, 16):
    for basis in ('kl_pca', 'dis_dense'):
        row = next(r for r in j['rows'] if r['side'] == 'input' and r['basis'] == basis and r['r'] == rank)
        assert j[f'input_r{rank}'][basis] == row['H1_test_rel']
j, rows = records[4]
for kind in ('random', 'kl', 'exact'):
    values = [float(r['H1_test_rel']) for r in rows if r['kind'] == kind]
    assert len(values) == 20
    for key, value in [('mean', np.mean(values)), ('std', np.std(values, ddof=1)), ('min', min(values)), ('max', max(values))]:
        np.testing.assert_allclose(value, j[kind][key], rtol=2e-5)
assert all(int(r['budget']) == int(r['N']) * int(r['dirs']) == 1280 for r in rows)
j, rows = records[5]
for basis in ('dis', 'pca'):
    for eps in (0.1, 0.03, 0.01):
        assert j['max_rank'][f'{basis} eps={eps}'] == [int(r['max_rank']) for r in rows if r['basis'] == basis and float(r['eps']) == eps]
j, _ = records[6]
assert (j['n_train'], j['n_test'], j['n_basis'], j['epochs']) == (640, 500, 160, 3000)
assert [r['basis_params'] for r in j['rows']] == [5120, 18480, 10240, 20920]
print('PASS: saved summary statistics and CSV/JSON fields agree; no experiments rerun')
