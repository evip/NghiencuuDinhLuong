import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress as MoHinhDuDoan

dl = pd.read_csv('gdp.csv')
dk_loc = dl["Country Code"]=="VNM"
dl_vn = dl[dk_loc]
X = dl_vn["Year"]
Y = dl_vn["Value"]

kq = MoHinhDuDoan(X,Y)
print("Ket qua mo hinh:")
print(kq)
a = kq[0]
b = kq[1]

# Y = a * X + b

x_test = np.array([2024, 2025, 2026, 2027])

y_predict = a * x_test + b
print("Ket qua du doan: ")
print(y_predict)

plt.plot(X, Y)
plt.plot(x_test, y_predict, color="red", marker='o')
plt.show()

