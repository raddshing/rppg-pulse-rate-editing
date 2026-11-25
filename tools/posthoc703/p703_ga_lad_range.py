"""Brief 02 7.03 가 (supplement): range of the LAD-optimal slopes (is the LAD fit unique?).

Same fit data as p703_ga.py fits. For each backbone and seed: the LAD optimum (linprog HiGHS), then two further linear programs
over the optimal set (sum |reference - (a predicted + c)| <= optimum x (1 + 1e-9) + 1e-9): minimum and maximum slope a, with the
intercepts at those ends. A width near 0 (of the order of the tolerance) means one optimal slope; a wide range means many optimal slopes.
Writes outputs/posthoc703/ga/ga_lad_range.csv.
"""
import time

from scipy import sparse
from scipy.optimize import linprog

from p703_common import DEV, SA, SRC_DEV, load, main_population, np, pd, primary_har, write
from p703_ga import FIT_NAMES, lad


def slope_range(p, r, opt):
    n = len(p)
    A = sparse.hstack([sparse.csr_matrix(np.c_[p, np.ones(n)]), sparse.identity(n), -sparse.identity(n)]).tocsr()
    ub = sparse.csr_matrix(np.r_[0., 0., np.ones(2 * n)][None, :])
    bounds = [(None, None)] * 2 + [(0, None)] * (2 * n)
    out = {}
    for name, sign in (('min', 1.), ('max', -1.)):
        res = linprog(np.r_[sign, 0., np.zeros(2 * n)], A_ub=ub, b_ub=[opt * (1 + 1e-9) + 1e-9], A_eq=A, b_eq=r, bounds=bounds,
                      method='highs')
        assert res.status == 0, res.message
        out[f'LAD_a_{name}'], out[f'LAD_c_at_a_{name}'] = float(res.x[0]), float(res.x[1])
    return out


def main():
    t0 = time.time()
    rows = []
    for ds in ('MMPD', 'MCD'):
        d, bad = load(DEV, ds)
        z = primary_har(d)
        z = z[z.population.eq(main_population(ds)) & z.arm.eq('(0)')]
        for (bb, sd), g in z.groupby(['backbone', 'seed']):
            g = SA.restrict(g)
            r, p = g.reference_HR.to_numpy(float), g.predicted_HR.to_numpy(float)
            a, c, obj, _ = lad(p, r)
            rows.append(dict(dataset_fit=FIT_NAMES[ds][0], backbone=bb, seed=int(sd), fit_windows=len(g), LAD_a=a, LAD_c=c,
                             LAD_objective=obj, distinct_predicted_values=int(len(np.unique(p))), **slope_range(p, r, obj), source=SRC_DEV))
            print(ds, bb, sd, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df['slope_range_width'] = df.LAD_a_max - df.LAD_a_min  # width of the optimal set within the tolerance
    write(df, 'ga_lad_range.csv', 'ga')
    print(df.drop(columns='source').to_string())


if __name__ == '__main__':
    main()
