import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Membuat tabel kontingensi 2x2 dari data:
#                  Setelah
#                P      N
#       P [[70,   15],   # Sebelum
#       N  [5,    25]]

table = np.array([[70, 15],
                  [5,  25]])

# Melakukan uji McNemar (menggunakan exact=False untuk uji chi-squared, atau exact=True untuk uji binomial)
result = mcnemar(table, exact=True)

# Menampilkan hasil
print("Statistik:", result.statistic)
print("p-value  :", result.pvalue)

# Interpretasi sederhana
alpha = 0.05
if result.pvalue < alpha:
    print("Terdapat perubahan signifikan dalam diagnosis setelah penerapan teknologi baru (tolak H0).")
else:
    print("Tidak terdapat perubahan signifikan dalam diagnosis setelah penerapan teknologi baru (gagal tolak H0).")
