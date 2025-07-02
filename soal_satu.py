import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset_aktivitas_fisik_kalori_berat_badan.csv")

# Definisikan variabel bebas dan terikat
X = df[["Aktivitas_Fisik_Jam", "Kalori_Harian_Kkal"]]
y = df["Berat_Badan_Kg"]
X_with_const = sm.add_constant(X)

# Fit model regresi
model = sm.OLS(y, X_with_const).fit()

# VIF (Multikolinearitas)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Uji heteroskedastisitas
_, pval_bpg, _, _ = het_breuschpagan(model.resid, model.model.exog)

# Uji normalitas residual
_, pval_normal = normal_ad(model.resid)

# Output hasil
print(model.summary())
print("\n===== Uji Asumsi Klasik =====")
print(vif_data)
print(f"P-value Breusch-Pagan: {pval_bpg:.4f}")
print(f"P-value Normalitas (Anderson-Darling): {pval_normal:.4f}")
print("\n===== Koefisien Determinasi =====")
print(f"R-squared: {model.rsquared:.3f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")

# Visualisasi residual
plt.figure(figsize=(8, 5))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={"color": "red"})
plt.xlabel("Nilai Prediksi")
plt.ylabel("Residual")
plt.title("Plot Residual vs Nilai Prediksi")
plt.grid(True)
plt.tight_layout()
plt.show()
