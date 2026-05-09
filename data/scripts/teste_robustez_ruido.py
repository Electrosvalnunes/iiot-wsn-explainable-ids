import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ==============================
# 1. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 2. Definir features e alvo
# ==============================
features = [
    "Topologia",
    "Avg_RSSI",
    "DIO_Count",
    "DIS_Count",
    "Rank_Changes",
    "PDR",
    "Avg_Delay",
    "Throughput_kbps",
    "Energy_Consumed"
]

target = "Attack_Class"

X = df[features].copy()
y = df[target].copy()

# ==============================
# 3. Codificar variáveis categóricas
# ==============================
le_topologia = LabelEncoder()
X["Topologia"] = le_topologia.fit_transform(X["Topologia"])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ==============================
# 4. Dividir treino e teste
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==============================
# 5. Treinar modelo base
# ==============================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# ==============================
# 6. Função para adicionar ruído
# ==============================
def add_noise(X_data, noise_level):
    X_noisy = X_data.copy()

    numeric_cols = [
        "Avg_RSSI",
        "DIO_Count",
        "DIS_Count",
        "Rank_Changes",
        "PDR",
        "Avg_Delay",
        "Throughput_kbps",
        "Energy_Consumed"
    ]

    for col in numeric_cols:
        std = X_noisy[col].std()
        noise = np.random.normal(0, noise_level * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise

    return X_noisy

# ==============================
# 7. Testar diferentes níveis de ruído
# ==============================
noise_levels = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

results = []

for noise in noise_levels:
    X_test_noisy = add_noise(X_test, noise)

    y_pred = rf.predict(X_test_noisy)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Noise_Level": noise,
        "Accuracy": acc
    })

    print("\n" + "="*60)
    print(f"Nível de ruído: {noise}")
    print(f"Accuracy: {acc:.4f}")

# ==============================
# 8. Salvar resultados
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("rf_robustez_ruido.csv", index=False)

print("\nResumo do teste de robustez:")
print(results_df)

print("\nTeste de robustez concluído com sucesso!")