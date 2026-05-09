import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================================
# 1. Carregar dataset limpo
# ==========================================================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==========================================================
# 2. Selecionar features da Rede Bayesiana
# ==========================================================
features = [
    "PDR",
    "Avg_Delay",
    "Throughput_kbps",
    "Energy_Consumed",
    "DIO_Count",
    "DIS_Count",
    "Rank_Changes"
]

target = "Attack_Class"

X = df[features].copy()
y = df[target].copy()

# ==========================================================
# 3. Dividir treino e teste
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 4. Discretizar métricas em Low / Medium / High
# ==========================================================
disc = KBinsDiscretizer(
    n_bins=3,
    encode="ordinal",
    strategy="quantile"
)

X_train_disc = disc.fit_transform(X_train)
X_test_disc = disc.transform(X_test)

X_train_disc = pd.DataFrame(
    X_train_disc,
    columns=features,
    index=X_train.index
)

X_test_disc = pd.DataFrame(
    X_test_disc,
    columns=features,
    index=X_test.index
)

bin_map = {
    0.0: "Low",
    1.0: "Medium",
    2.0: "High",
    0: "Low",
    1: "Medium",
    2: "High"
}

for col in features:
    X_train_disc[col] = X_train_disc[col].map(bin_map).astype(str)
    X_test_disc[col] = X_test_disc[col].map(bin_map).astype(str)

# ==========================================================
# 5. Criar base discreta para a Rede Bayesiana
# ==========================================================
train_bn = X_train_disc.copy()
train_bn[target] = y_train.values

test_bn = X_test_disc.copy()
test_bn[target] = y_test.values

# ==========================================================
# 6. Aprender probabilidades da Rede Bayesiana
# Estrutura: Attack_Class -> cada métrica
# Usamos suavização de Laplace para evitar probabilidade zero
# ==========================================================
classes = sorted(train_bn[target].unique())
states = ["Low", "Medium", "High"]
alpha = 1.0  # Laplace smoothing

# Probabilidade a priori P(Attack_Class)
class_counts = train_bn[target].value_counts().reindex(classes, fill_value=0)
priors = (class_counts + alpha) / (len(train_bn) + alpha * len(classes))

print("\nProbabilidades a priori P(Attack_Class):")
print(priors.round(4))

# Probabilidades condicionais P(feature_state | Attack_Class)
conditional_probs = {}

for feat in features:
    table = pd.DataFrame(index=classes, columns=states, dtype=float)

    for cls in classes:
        subset = train_bn[train_bn[target] == cls]
        counts = subset[feat].value_counts().reindex(states, fill_value=0)

        probs = (counts + alpha) / (len(subset) + alpha * len(states))
        table.loc[cls] = probs.values

    conditional_probs[feat] = table

    # Salvar tabela condicional de cada métrica
    table.to_csv(f"bn_cpt_{feat}.csv")

# ==========================================================
# 7. Função de inferência
# Calcula: P(classe | evidências) proporcional a:
# P(classe) * produto P(feature | classe)
# ==========================================================
def predict_bn(row):
    scores = {}

    for cls in classes:
        log_prob = np.log(priors.loc[cls])

        for feat in features:
            state = row[feat]
            prob = conditional_probs[feat].loc[cls, state]
            log_prob += np.log(prob)

        scores[cls] = log_prob

    # Converter log-probabilidades para probabilidades normalizadas
    max_log = max(scores.values())
    exp_scores = {cls: np.exp(score - max_log) for cls, score in scores.items()}
    total = sum(exp_scores.values())

    probs = {cls: exp_scores[cls] / total for cls in classes}

    pred_class = max(probs, key=probs.get)

    return pred_class, probs

# ==========================================================
# 8. Fazer predições no conjunto de teste
# ==========================================================
y_pred = []
probability_rows = []

for idx, row in X_test_disc.iterrows():
    pred_class, probs = predict_bn(row)

    y_pred.append(pred_class)

    prob_row = {
        "Index": idx,
        "True_Class": y_test.loc[idx],
        "Predicted_Class": pred_class
    }

    for cls in classes:
        prob_row[f"P_{cls}"] = probs[cls]

    probability_rows.append(prob_row)

# ==========================================================
# 9. Avaliação
# ==========================================================
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy da Rede Bayesiana:", round(acc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

print("\nConfusion Matrix:")
print(cm_df)

# ==========================================================
# 10. Salvar resultados
# ==========================================================
cm_df.to_csv("bn_confusion_matrix.csv")

probabilities_df = pd.DataFrame(probability_rows)
probabilities_df.to_csv("bn_prediction_probabilities.csv", index=False)

edges = [(target, feat) for feat in features]
edges_df = pd.DataFrame(edges, columns=["Parent", "Child"])
edges_df.to_csv("bn_structure_edges.csv", index=False)

priors.to_csv("bn_priors.csv", header=["Probability"])

# Salvar os limites dos bins usados na discretização
bin_edges_data = []

for feat, edges_array in zip(features, disc.bin_edges_):
    bin_edges_data.append({
        "Feature": feat,
        "Low_to_Medium": edges_array[1],
        "Medium_to_High": edges_array[2]
    })

bin_edges_df = pd.DataFrame(bin_edges_data)
bin_edges_df.to_csv("bn_discretization_thresholds.csv", index=False)

print("\nArquivos salvos com sucesso:")
print("bn_confusion_matrix.csv")
print("bn_prediction_probabilities.csv")
print("bn_structure_edges.csv")
print("bn_priors.csv")
print("bn_discretization_thresholds.csv")

for feat in features:
    print(f"bn_cpt_{feat}.csv")

print("\nRede Bayesiana interpretável executada com sucesso!")