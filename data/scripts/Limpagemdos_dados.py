import pandas as pd

# ==============================
# 1. Carregar dataset OMNeT++
# ==============================
df_omnet = pd.read_csv("dataset_omnetpp_20k.csv")

# ==============================
# 2. Criar coluna de classe única
# ==============================
class_columns = [
    "Normal",
    "Flooding",
    "Blackhole",
    "Wormhole",
    "Backoff_Manipulado"
]

df_omnet["Attack_Class"] = df_omnet[class_columns].idxmax(axis=1)

# ==============================
# 3. Selecionar features principais
# ==============================
features_omnet = [
    "Topologia",
    "Avg_RSSI",
    "DIO_Count",
    "DIS_Count",
    "Rank_Changes",
    "PDR",
    "Avg_Delay",
    "Throughput_kbps",
    "Energy_Consumed",
    "Attack_Class"
]

df_omnet_clean = df_omnet[features_omnet]

# ==============================
# 4. Verificar resultado
# ==============================
print(df_omnet_clean.head())
print(df_omnet_clean["Attack_Class"].value_counts())
print(df_omnet_clean["Topologia"].value_counts())

# ==============================
# 5. Salvar dataset limpo
# ==============================
df_omnet_clean.to_csv("omnetpp_dataset_clean.csv", index=False)

print("Dataset OMNeT++ limpo salvo com sucesso!")