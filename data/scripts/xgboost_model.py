import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ==============================
# 1. Tentar importar XGBoost
# ==============================
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost não está instalado.")
    print("Instale com o comando:")
    print("pip install xgboost")
    raise

# ==============================
# 2. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 3. Definir features e alvo
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
# 4. Codificar variáveis categóricas
# ==============================
le_topologia = LabelEncoder()
X["Topologia"] = le_topologia.fit_transform(X["Topologia"])

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ==============================
# 5. Dividir treino e teste
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Tamanho treino:", X_train.shape)
print("Tamanho teste:", X_test.shape)

# ==============================
# 6. Treinar XGBoost
# ==============================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=len(le_target.classes_),
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)

# ==============================
# 7. Predição
# ==============================
y_pred = xgb.predict(X_test)

# ==============================
# 8. Avaliação
# ==============================
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(acc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# 9. Salvar matriz de confusão
# ==============================
cm_df = pd.DataFrame(
    cm,
    index=le_target.classes_,
    columns=le_target.classes_
)

cm_df.to_csv("xgb_confusion_matrix.csv")

# ==============================
# 10. Importância das features
# ==============================
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nImportância das Features:")
print(feature_importance_df)

feature_importance_df.to_csv("xgb_feature_importance.csv", index=False)

# ==============================
# 11. Gráfico da importância das features
# ==============================
plt.figure(figsize=(8, 5))
plt.bar(feature_importance_df["Feature"], feature_importance_df["Importance"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=300)
plt.show()

print("\nModelo XGBoost treinado e avaliado com sucesso!")