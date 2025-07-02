import pandas as pd
from scipy.stats import kruskal

# Baca dataset dari CSV
df = pd.read_csv("dataset_soal_dua.csv")

# Ambil kolom waktu eksekusi tiap algoritma
a = df["Algoritma_A"]
b = df["Algoritma_B"]
c = df["Algoritma_C"]

# Uji Kruskal-Wallis
stat, p = kruskal(a, b, c)

print("Hasil Uji Kruskal-Wallis:")
print(f"Statistik H = {stat:.4f}")
print(f"p-value     = {p:.4f}")

if p < 0.05:
    print("Terdapat perbedaan signifikan antara algoritma.")
else:
    print("Tidak terdapat perbedaan signifikan antara algoritma.")
