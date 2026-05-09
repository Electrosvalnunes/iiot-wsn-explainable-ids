import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# ==============================
# 1. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 2. Definir features sem Topologia
# ==============================
features = [
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
# 3. Leave-one-topology-out
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

    # Codificar classes
    le_target = LabelEncoder()
    le_target.fit(df[target])

    y_train_encoded = le_target.transform(y_train)
    y_test_encoded = le_target.transform(y_test)

    # ==============================
    # 4. XGBoost mais regularizado
    # ==============================
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="multi:softmax",
        num_class=len(le_target.classes_),
        eval_metric="mlogloss",
        random_state=42
    )

    xgb.fit(X_train, y_train_encoded)

    y_pred = xgb.predict(X_test)

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
# 5. Salvar resumo
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv("xgb_leave_one_topology_sem_topologia_results.csv", index=False)

print("\nResumo final:")
print(results_df)

print("\nMédia de accuracy:", results_df["Accuracy"].mean())
print("Desvio padrão:", results_df["Accuracy"].std())

print("\nValidação XGBoost sem Topologia concluída com sucesso!")