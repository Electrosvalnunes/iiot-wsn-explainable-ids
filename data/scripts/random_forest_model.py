import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 1. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 2. Selecionar features e alvo
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
# Topologia
le_topologia = LabelEncoder()
X["Topologia"] = le_topologia.fit_transform(X["Topologia"])

# Classe alvo
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ==============================
# 4. Dividir treino e teste
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Tamanho treino:", X_train.shape)
print("Tamanho teste:", X_test.shape)

# ==============================
# 5. Treinar Random Forest
# ==============================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# ==============================
# 6. Fazer previsões
# ==============================
y_pred = rf.predict(X_test)

# ==============================
# 7. Avaliação
# ==============================
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ==============================
# 8. Salvar matriz de confusão em CSV
# ==============================
cm_df = pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_)
cm_df.to_csv("rf_confusion_matrix.csv")

# ==============================
# 9. Importância das features
# ==============================
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nImportância das Features:")
print(feature_importance_df)

feature_importance_df.to_csv("rf_feature_importance.csv", index=False)

# ==============================
# 10. Gráfico da importância das features
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=300)
plt.show()

print("\nModelo Random Forest treinado e avaliado com sucesso!")