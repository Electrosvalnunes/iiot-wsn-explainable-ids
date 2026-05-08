import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

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

topologies = sorted(df["Topologia"].unique())

results = []

# ==============================
# 3. Loop: deixar uma topologia fora
# ==============================
for test_topology in topologies:
    print("\n" + "="*70)
    print(f"Treinando sem {test_topology} e testando em {test_topology}")
    print("="*70)

    train_df = df[df["Topologia"] != test_topology].copy()
    test_df = df[df["Topologia"] == test_topology].copy()

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    y_train = train_df[target].copy()
    y_test = test_df[target].copy()

    # ==============================
    # 4. Codificar Topologia
    # ==============================
    le_topologia = LabelEncoder()

    # Ajustar usando todas as topologias para evitar erro de categoria desconhecida
    le_topologia.fit(df["Topologia"])

    X_train["Topologia"] = le_topologia.transform(X_train["Topologia"])
    X_test["Topologia"] = le_topologia.transform(X_test["Topologia"])

    # ==============================
    # 5. Codificar classe alvo
    # ==============================
    le_target = LabelEncoder()
    le_target.fit(df[target])

    y_train_encoded = le_target.transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    # ==============================
    # 6. Treinar modelo
    # ==============================
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train_encoded)

    # ==============================
    # 7. Predição
    # ==============================
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)

    print(f"\nAccuracy em {test_topology}: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test_encoded,
        y_pred,
        target_names=le_target.classes_
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_encoded, y_pred)
    print(cm)

    results.append({
        "Test_Topology": test_topology,
        "Accuracy": acc
    })

# ==============================
# 8. Salvar resultados finais
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("rf_leave_one_topology_results.csv", index=False)

print("\nResumo final:")
print(results_df)

print("\nMédia de accuracy:", results_df["Accuracy"].mean())
print("Desvio padrão:", results_df["Accuracy"].std())

print("\nValidação por topologia concluída com sucesso!")