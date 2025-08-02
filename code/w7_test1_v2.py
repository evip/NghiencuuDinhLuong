import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
new_process = np.random.binomial(1, 0.18, 5550)
old_process = np.random.binomial(1,0.16,5300)

data = pd.DataFrame({
    'process': ['new' ]*5550 + ['old']*5300,
    'converted': np.concatenate([new_process, old_process])
})

conversion_summary = data.groupby('process')['converted'].agg(['count', 'sum', 'mean'])
conversion_summary['conversion_rate'] = conversion_summary['mean']
print("Conversion Summary:")
print(conversion_summary)

from statsmodels.stats.proportion import proportions_ztest
# Prepare data for test
successes = conversion_summary['sum'].values
nobs = conversion_summary['count'].values
# Two-sample proportion test
print("---------------------")
print("len: ", len(new_process))
print("count: ", np.sum(new_process))
print("rate: ", np.sum(new_process)/len(new_process))
#print(new_process)

z_stat, p_value = proportions_ztest(successes, nobs)
print(f"\nTest Results:")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significance level: 0.05")
print(f"Reject H0: {p_value < 0.05}")
