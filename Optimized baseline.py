"""
Optimisation réseau – coût minimum
Hypothèses validées :
  • chaque client peut être servi par n’importe quelle plateforme
  • chaque plateforme peut se fournir auprès de F1 et/ou F2
  • coûts transport linéaires €/u (feuille « Transport cost »)
  • coefficients matières premières fixes (Process)
  • capacités Max capacity sont des contraintes dures
  • objectif = min( Transport + Production + Purchase )
  • produits indépendants, une seule période
"""

import pandas as pd
import pulp

# ───────────────────────── 1. CHARGEMENT DES DONNÉES ──────────────────────────
xl = "Data2025.xlsx"
cust_df      = pd.read_excel(xl, "Customers Demand")        
base_df      = pd.read_excel(xl, "Base line")
process_df   = pd.read_excel(xl, "Process")                 
plat_df      = pd.read_excel(xl, "Plateform")               
sup_df       = pd.read_excel(xl, "Suppliers")               
transport_df = pd.read_excel(xl, "Transport cost")          

# Nettoyage automatique des colonnes (supprime les espaces cachés)
process_df.columns    = [c.strip() for c in process_df.columns]
sup_df.columns        = [c.strip() for c in sup_df.columns]
transport_df.columns  = [c.strip() for c in transport_df.columns]

# Affichage pour vérification (sécurité)
print("Colonnes process_df :", process_df.columns.tolist())
print("Colonnes transport_df :", transport_df.columns.tolist())

# ─────────────────────── 2. ENSEMBLES DE BASE ────────────────────────────────
products   = cust_df["Product" ].unique().tolist()
clients    = cust_df["Customer"].unique().tolist()
platforms  = plat_df["Plateform"].unique().tolist()
suppliers  = sup_df["Supplier" ].unique().tolist()
materials  = ["A", "B"]

# ─────────────────────── 3. PARAMÈTRES NUMÉRIQUES ────────────────────────────
# 3.1 Demande client-produit
demand = {(row.Customer, row.Product): row.Demande for _, row in cust_df.iterrows()}

# 3.2 Coût de production unitaire par plateforme
prod_cost = plat_df.set_index("Plateform")["Production cost (€/u)"].to_dict()

# 3.3 Capacité plateforme
cap_ptf = plat_df.set_index("Plateform")["Max capacity"].to_dict()

# 3.4 Coef matière première par plateforme-produit
alpha = {}
beta  = {}
for _, row in process_df.iterrows():
    alpha[(row["Plateforme"], row["Product"])] = row["Raw material A"]
    beta [(row["Plateforme"], row["Product"])] = row["Raw metrial B"]

# 3.5 Prix d’achat matière par supplier
purch_cost = {(row["Supplier"], row["Product"]): row["Purchasing cost (€/u)"] for _, row in sup_df.iterrows()}

# 3.6 Capacité supplier (somme des deux matières)
cap_sup = sup_df.groupby("Supplier")["Max capacity"].sum().to_dict()

# 3.7 Coût transport plateforme→client
trans_ptf_cli = {}
# 3.8 Coût transport supplier→plateforme
trans_sup_ptf = {}

for _, row in transport_df.iterrows():
    src  = row["GeoPoint#Name"]
    dst  = row["GeoPoint_1#Name"]
    cost = row["€/U"]
    
    if str(dst).startswith("C"):          
        trans_ptf_cli[(src, dst)] = cost
    elif str(dst).startswith("P"):        
        trans_sup_ptf[(src, dst)] = cost

# ─────────────────────── 4. MODÈLE PULP ───────────────────────────────────────
prob = pulp.LpProblem("Network_Optim", pulp.LpMinimize)

# 4.1 Variables x[ptf,c,prd] & y[sup,ptf,mat]
x = pulp.LpVariable.dicts(
        "x", ((p,c,pr) for p in platforms for c in clients for pr in products),
        lowBound=0, cat="Continuous")

y = pulp.LpVariable.dicts(
        "y", ((s,p,r) for s in suppliers for p in platforms for r in materials),
        lowBound=0, cat="Continuous")

# 4.2 Objectif = transport + production + purchase
prob += (
    pulp.lpSum(trans_ptf_cli.get((p,c), 1e9) * x[(p,c,pr)]
               for p in platforms for c in clients for pr in products) +
    pulp.lpSum(trans_sup_ptf.get((s,p), 1e9) * y[(s,p,r)]
               for s in suppliers for p in platforms for r in materials) +
    pulp.lpSum(prod_cost[p] * x[(p,c,pr)]
               for p in platforms for c in clients for pr in products) +
    pulp.lpSum(purch_cost[(s,r)] * y[(s,p,r)]
               for s in suppliers for p in platforms for r in materials)
)

# 4.3 Contraintes
# a) satisfaction de la demande
for c in clients:
    for pr in products:
        prob += pulp.lpSum(x[(p,c,pr)] for p in platforms) == demand.get((c,pr), 0)

# b) capacité plateforme
for p in platforms:
    prob += pulp.lpSum(x[(p,c,pr)] for c in clients for pr in products) <= cap_ptf[p]

# c) bilan matière A & B sur chaque plateforme
for p in platforms:
    # matière A
    prob += pulp.lpSum(y[(s,p,"A")] for s in suppliers) == \
            pulp.lpSum(alpha[(p,pr)] * x[(p,c,pr)]
                        for c in clients for pr in products)
    # matière B
    prob += pulp.lpSum(y[(s,p,"B")] for s in suppliers) == \
            pulp.lpSum(beta[(p,pr)] * x[(p,c,pr)]
                        for c in clients for pr in products)

# d) capacité supplier
for s in suppliers:
    prob += pulp.lpSum(y[(s,p,"A")] + y[(s,p,"B")]  for p in platforms) <= cap_sup[s]

# ─────────────────────── 5. RÉSOLUTION ───────────────────────────────────────
prob.solve(pulp.PULP_CBC_CMD(msg=True))

print("Status :", pulp.LpStatus[prob.status])
print("Coût optimal   :", "{:,.0f}".format(pulp.value(prob.objective)), "€")

# ─────────────────────── 6. REPORT (exemple) ──────────────────────────────────
# Extraction des flux optimisés
print("\nFlux optimisés plateforme → client :")
for (p,c,pr), var in x.items():
    if var.varValue > 1e-5:
        print(f"{p} → {c} [{pr}] : {var.varValue:.1f} u")

print("\nFlux optimisés supplier → plateforme :")
for (s,p,r), var in y.items():
    if var.varValue > 1e-5:
        print(f"{s} → {p} [{r}] : {var.varValue:.1f} u")


# ─────────────────────── 7. MAP BASELINE OPTIMISÉE ──────────────────────────────────
import folium
import webbrowser

# === Lecture des coordonnées des sites ===
site_df = pd.read_excel(xl, sheet_name="Site's location")
coord = {row["Name"]: [row["Latitude"], row["Longitude"]] for _, row in site_df.iterrows()}

# === Listes ===
plateforms = site_df[site_df["Type"] == "Plateform"]["Name"].tolist()
customers  = site_df[site_df["Type"] == "Customers"]["Name"].tolist()
suppliers  = site_df[site_df["Type"] == "Supplier"]["Name"].tolist()

# === Création de la carte ===
m = folium.Map([49.8694, 2.209222], zoom_start=5)

# Ajout des clients, plateformes, fournisseurs
for c in customers:
    folium.Marker(location=coord[c], popup=c, icon=folium.Icon(color="green")).add_to(m)
for p in plateforms:
    folium.Marker(location=coord[p], popup=p, icon=folium.Icon(color="red")).add_to(m)
for s in suppliers:
    folium.Marker(location=coord[s], popup=s, icon=folium.Icon(color="blue")).add_to(m)

# === Flux plateforme → client optimisés ===
def plot_optimized_product(product_filter=None):
    for (p, c, pr), var in x.items():
        volume = var.varValue
        if volume is None or volume <= 1e-5:
            continue
        if product_filter and pr != product_filter:
            continue
        folium.PolyLine(
            locations=[coord[p], coord[c]],
            color="green",
            weight=3,  # poids FIXE
            popup=f"{p} → {c} | {pr} | {volume:.1f} u"
        ).add_to(m)

# Appel :
plot_optimized_product()

# === Flux supplier → plateforme optimisés ===
for (s, p, r), var in y.items():
    qty = var.varValue
    if qty is None or qty <= 1e-5:
        continue
    folium.PolyLine(
        locations=[coord[s], coord[p]],
        color="blue",
        weight=3,  # poids FIXE
        popup=f"{s} → {p} | {r} | {qty:.1f} u"
    ).add_to(m)

# === Sauvegarde de la carte ===
map_filename = "optimized_baseline_map.html"
m.save(map_filename)
webbrowser.open(map_filename)
