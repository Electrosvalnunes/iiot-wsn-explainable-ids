import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Carregar resultado do teste de robustez
# ==============================
df = pd.read_csv("rf_robustez_ruido.csv")

# Converter ruído para porcentagem
df["Noise_Percentage"] = df["Noise_Level"] * 100

# ==============================
# 2. Gerar gráfico
# ==============================
plt.figure(figsize=(8, 5))
plt.plot(df["Noise_Percentage"], df["Accuracy"], marker="o")
plt.xlabel("Noise level (%)")
plt.ylabel("Accuracy")
plt.title("Random Forest robustness under noisy input features")
plt.ylim(0.90, 1.01)
plt.grid(True)
plt.tight_layout()

plt.savefig("rf_robustez_ruido.png", dpi=300)
plt.show()

print("Gráfico de robustez salvo com sucesso!")