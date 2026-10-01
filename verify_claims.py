"""Verify the portfolio notebook's narrative claims against recomputed results."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

fires = pd.read_csv('data/Fires.csv')
salaries = pd.read_csv('data/Salaries.csv')

print('=== Study 1: Fires ===')
print('n =', len(fires), '| cols =', list(fires.columns))
print('mean DISTANCE = %.2f, mean DAMAGE = %.2f' % (fires['DISTANCE'].mean(), fires['DAMAGE'].mean()))
print('std DISTANCE = %.2f, std DAMAGE = %.2f' % (fires['DISTANCE'].std(), fires['DAMAGE'].std()))
print('corr = %.3f' % fires['DISTANCE'].corr(fires['DAMAGE']))
X = sm.add_constant(fires['DISTANCE'])
m1 = sm.OLS(fires['DAMAGE'], X).fit()
print('intercept = %.2f, slope = %.2f' % (m1.params['const'], m1.params['DISTANCE']))
print('R^2 = %.3f' % m1.rsquared)
print('slope t = %.2f, p = %.2e' % (m1.tvalues['DISTANCE'], m1.pvalues['DISTANCE']))
print('slope 95%% CI = [%.2f, %.2f]' % tuple(m1.conf_int().loc['DISTANCE']))
infl = m1.get_influence()
print('max |std resid| = %.2f, max Cooks D = %.3f' % (
    np.max(np.abs(infl.resid_studentized_internal)), np.max(infl.cooks_distance[0])))
# prediction at 3 miles
new_X = pd.DataFrame({'const': 1.0, 'DISTANCE': [3.0]})
pred = m1.get_prediction(new_X).summary_frame(alpha=0.05)
print('pred at 3mi: mean=%.2f, CI=[%.2f, %.2f]' % (pred['mean'][0], pred['obs_ci_lower'][0], pred['obs_ci_upper'][0]))

print()
print('=== Study 2: Salaries ===')
sal = salaries.drop(columns=['ID'])
print('n =', len(sal))
full = smf.ols('Y ~ X1+X2+X3+X4+X5+X6+X7+X8+X9+X10', data=sal).fit()
print('full R^2 = %.3f, adj R^2 = %.3f, F p = %.2e' % (full.rsquared, full.rsquared_adj, full.f_pvalue))
Xv = sm.add_constant(sal.drop(columns=['Y']))
vifs = {c: variance_inflation_factor(Xv.values, i) for i, c in enumerate(Xv.columns) if c != 'const'}
print('VIFs:', {k: round(v, 2) for k, v in sorted(vifs.items(), key=lambda x: -x[1])[:4]})
print('corr X1,X7 = %.2f' % sal['X1'].corr(sal['X7']))
final = smf.ols('Y ~ X1+X2+X3+X4+X5', data=sal).fit()
print('final adj R^2 = %.3f' % final.rsquared_adj)
print('final coefs:')
for v in ['X1', 'X2', 'X3', 'X4', 'X5']:
    print('  %s: %.4f (p=%.4f)' % (v, final.params[v], final.pvalues[v]))
# example prediction
nd = pd.DataFrame({'X1': [10], 'X2': [16], 'X3': [1], 'X4': [250], 'X5': [180]})
pl = final.predict(nd)[0]
print('example: log-sal = %.3f -> $%.0f' % (pl, np.exp(pl)))
infl3 = final.get_influence()
sr = infl3.resid_studentized_internal
print('final max |std resid| = %.2f, max Cooks D = %.3f' % (np.max(np.abs(sr)), np.max(infl3.cooks_distance[0])))
