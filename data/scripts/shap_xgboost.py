import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ==============================
# 1. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 2. Features sem Topologia
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

X = df[features].copy()
y = df[target].copy()

# ==============================
# 3. Codificar classe alvo
# ==============================
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("Classes codificadas:")
for i, cls in enumerate(le_target.classes_):
    print(i, "=", cls)

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
# 5. Treinar XGBoost ajustado
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
    objective="multi:softprob",
    num_class=len(le_target.classes_),
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)

# ==============================
# 6. Amostra para SHAP
# ==============================
# Usamos uma amostra menor para evitar lentidão
X_sample = X_test.sample(n=1000, random_state=42)

# ==============================
# 7. Calcular SHAP
# ==============================
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

# ==============================
# 8. SHAP summary bar plot
# ==============================
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    class_names=le_target.classes_,
    plot_type="bar",
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary_bar_xgb.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================
# 9. SHAP beeswarm/summary plot
# ==============================
shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=features,
    class_names=le_target.classes_,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary_beeswarm_xgb.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGráficos SHAP gerados com sucesso!")
print("Arquivos salvos:")
print("shap_summary_bar_xgb.png")
print("shap_summary_beeswarm_xgb.png")