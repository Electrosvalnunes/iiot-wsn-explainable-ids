import pandas as pd

# ==============================
# 1. Carregar dataset limpo
# ==============================
df = pd.read_csv("omnetpp_dataset_clean.csv")

# ==============================
# 2. Conferir estrutura básica
# ==============================
print("\nDimensão do dataset:")
print(df.shape)

print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações gerais:")
print(df.info())

print("\nValores ausentes:")
print(df.isnull().sum())

# ==============================
# 3. Distribuição das classes
# ==============================
print("\nDistribuição por classe:")
print(df["Attack_Class"].value_counts())

# ==============================
# 4. Distribuição das topologias
# ==============================
print("\nDistribuição por topologia:")
print(df["Topologia"].value_counts())

# ==============================
# 5. Estatísticas gerais das métricas
# ==============================
metrics = [
    "PDR",
    "Avg_Delay",
    "Throughput_kbps",
    "Energy_Consumed",
    "DIO_Count",
    "DIS_Count",
    "Rank_Changes",
    "Avg_RSSI"
]

print("\nEstatísticas gerais:")
print(df[metrics].describe())

# ==============================
# 6. Média das métricas por ataque
# ==============================
mean_by_attack = df.groupby("Attack_Class")[metrics].mean().round(3)

print("\nMédia das métricas por ataque:")
print(mean_by_attack)

# ==============================
# 7. Média das métricas por topologia
# ==============================
mean_by_topology = df.groupby("Topologia")[metrics].mean().round(3)

print("\nMédia das métricas por topologia:")
print(mean_by_topology)

# ==============================
# 8. Média por ataque e topologia
# ==============================
mean_by_attack_topology = df.groupby(["Topologia", "Attack_Class"])[metrics].mean().round(3)

print("\nMédia das métricas por topologia e ataque:")
print(mean_by_attack_topology)

# ==============================
# 9. Salvar tabelas para uso no artigo
# ==============================
mean_by_attack.to_csv("tabela_media_por_ataque.csv")
mean_by_topology.to_csv("tabela_media_por_topologia.csv")
mean_by_attack_topology.to_csv("tabela_media_por_ataque_topologia.csv")

print("\nTabelas salvas com sucesso!")