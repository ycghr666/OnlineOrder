import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'ampl_deps'))
import itertools
import json
import numpy as np
from scipy.optimize import linprog
from amplpy import AMPL

out = Path(__file__).resolve().parents[1] / 'output' / 'q1'
cases = [
    ('A', 1, 1, 4, 1, 1, 'solved', 7),
    ('B', 0, 1, 4, 1, 1, 'solved', 7),
    ('C', -1, -1, 4, -1, 1, 'unbounded', None),
    ('D', -1, 1, 4, 1, 1, 'solved', 9),
    ('E', 1, 1, -1, 1, 1, 'infeasible', None),
    ('F', -1, 1, 0, 1, 1, 'solved', 7),
    ('G', 0, 1, 0, 1, 1, 'solved', 7),
    ('H', -4, -4, 4, 7, 1, 'unbounded', None),
    ('I', -4, -4, -1, 7, 1, 'unbounded', None),
    ('J', -4, -4, -2, 7, 1, 'infeasible', None),
]
ampl = AMPL()
ampl.cd(str(out))
ampl.read('q1.run')
results = []
for name, a, b, c, d, e, status, objective in cases:
    for key, value in zip('abcde', (a, b, c, d, e)):
        ampl.param[key] = value
    ampl.solve()
    actual = ampl.get_value('solve_result')
    assert actual == status, (name, actual, status)
    value = ampl.get_value('z') if actual == 'solved' else None
    if objective is not None:
        assert abs(value - objective) < 1e-8
    results.append(dict(case=name, status=actual, objective=value))
for key, value in zip('abcde', (0, 1, 4, 1, 1)):
    ampl.param[key] = value
ampl.eval('fix x[5] := 1; solve;')
assert abs(ampl.get_value('z') - 7) < 1e-8
assert abs(ampl.get_value('x[5]') - 1) < 1e-8
ampl.eval('unfix x[5];')
ampl.close()

grid = list(itertools.product([-4, -1, 0, 2], [-4, -1, 0, 2], [-2, -8/7, -1, 0, 2], [-1, 0, 7], [-2, 0, 2, 3]))
rng = np.random.default_rng(127)
grid += [tuple(row) for row in rng.uniform(-10, 10, (1500, 5))]
for a, b, c, d, e in grid:
    mat = [[4, b], [-7, d], [e, -2]]
    rhs = [c, 2, 4]
    result = linprog([3, a], A_ub=mat, b_ub=rhs, bounds=(0, None), method='highs', options={'presolve': False})
    L = max(0, d/7)
    s = b + 4*L
    predicted = a + 3*L < -1e-9 and s <= 1e-9 and e*L <= 2 + 1e-9 and (s < -1e-9 or (d > 0 and c >= -8/7 - 1e-9) or (d <= 0 and c >= 0))
    assert (result.status == 3) == predicted, ((a,b,c,d,e), result.status, predicted)
    if c >= 0:
        unique = a > 0 or (c == 0 and b > 0)
        multiple = a == 0 and (c > 0 or b <= 0)
        nonoptimal = a < 0 and (c > 0 or b <= 0)
        assert sum([unique, multiple, nonoptimal]) == 1
        optimal = result.status == 0 and abs(result.fun) < 1e-7
        assert optimal == (unique or multiple), ((a,b,c,d,e), result)
        if optimal:
            alternative = linprog([-1,-1], A_ub=mat, b_ub=rhs, A_eq=[[3,a]], b_eq=[0], bounds=(0,None), method='highs', options={'presolve':False})
            has_other = alternative.status == 3 or (alternative.status == 0 and alternative.fun < -1e-7)
            assert has_other == multiple

report = {'ampl_cases': results, 'alternate_optimum_checked': True, 'parameter_combinations_checked': len(grid), 'result': 'all passed'}
(out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
