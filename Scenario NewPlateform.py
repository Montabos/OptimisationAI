"""
Scénario : on ferme exactement UNE des 4 plateformes actuelles, on ouvre UNE
nouvelle plateforme candidate (200 positions aléatoires France métrop. seed=42).
On re-optimise simultanément :
    – affectation clients ↔ plateformes
    – sourcing matières premières suppliers ↔ plateformes
Objectif : min( Transport + Production + Purchase ).

Règles supplémentaires :
    • transport €/u = 0,3448 × km + 33,591
    • new-platform : capacité & α/β identiques à la plateforme fermée
                     prod-cost = moyenne des 2 plateformes existantes les + proches
"""

import pandas as pd, numpy as np, math, matplotlib.pyplot as plt, folium, webbrowser
import pulp, random, os

# ---------------------- 1. Chargement & données de base -----------------------
xl = "Data2025.xlsx"
cust_df      = pd.read_excel(xl, "Customers Demand")
process_df   = pd.read_excel(xl, "Process")
plat_df      = pd.read_excel(xl, "Plateform")
sup_df       = pd.read_excel(xl, "Suppliers")
site_df      = pd.read_excel(xl, "Site's location")

process_df.columns   = [c.strip() for c in process_df.columns]
sup_df.columns       = [c.strip() for c in sup_df.columns]

products   = cust_df["Product" ].unique().tolist()
clients    = cust_df["Customer"].unique().tolist()
platforms  = plat_df["Plateform"].unique().tolist()
suppliers  = sup_df["Supplier" ].unique().tolist()
materials  = ["A","B"]

# demande client-produit
demand = {(r.Customer, r.Product): r.Demande for _,r in cust_df.iterrows()}

# capacité / coût prod des PTF existantes
cap_ptf   = plat_df.set_index("Plateform")["Max capacity"].to_dict()
prod_cost = plat_df.set_index("Plateform")["Production cost (€/u)"].to_dict()

# α / β
alpha, beta = {}, {}
for _,r in process_df.iterrows():
    alpha[(r["Plateforme"], r["Product"])] = r["Raw material A"]
    beta [(r["Plateforme"], r["Product"])] = r["Raw metrial B"]

# capacité suppliers & prix matières
cap_sup = sup_df.groupby("Supplier")["Max capacity"].sum().to_dict()
purch_cost = {(r.Supplier, r.Product): r["Purchasing cost (€/u)"] for _,r in sup_df.iterrows()}

# coordonnées de tous les sites existants
coord = {r.Name: (r.Latitude, r.Longitude) for _,r in site_df.iterrows()}

# ---------------------- 2. Génération 500 candidats ---------------------------
rnd = random.Random(42)
cand_coords = [(rnd.uniform(42,51), rnd.uniform(-5,8)) for _ in range(200)]
cand_ids    = [f"CAND{k}" for k in range(200)]
for cid, (la,lo) in zip(cand_ids, cand_coords):
    coord[cid] = (la, lo)

# haversine km
def hav_km(a,b):
    lat1,lon1 = a; lat2,lon2 = b
    R=6371
    dφ=math.radians(lat2-lat1); dλ=math.radians(lon2-lon1)
    φ1,φ2=math.radians(lat1),math.radians(lat2)
    a2=math.sin(dφ/2)**2+math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R*2*math.atan2(math.sqrt(a2),math.sqrt(1-a2))

# distances
dist_cli = {(p,c): hav_km(coord[p],coord[c]) for p in platforms+cand_ids for c in clients}
dist_sup = {(s,p): hav_km(coord[s],coord[p]) for s in suppliers   for p in platforms+cand_ids}

# prod-cost candidat = moyenne 2 PTF les plus proches
prod_cost_cand = {}
for cid in cand_ids:
    dists = sorted([(hav_km(coord[cid],coord[p]), p) for p in platforms])
    p1,p2 = dists[0][1], dists[1][1]
    prod_cost_cand[cid] = (prod_cost[p1]+prod_cost[p2])/2

# ---------------------- 3. Modèle PuLP mixte ----------------------------------
prob = pulp.LpProblem("CloseOpenOptim", pulp.LpMinimize)

# binaires : fermer exactement 1 PTF, ouvrir exactement 1 candidat
close = pulp.LpVariable.dicts("close", platforms, 0,1, cat="Binary")
open_ = pulp.LpVariable.dicts("open" , cand_ids,0,1, cat="Binary")
prob += pulp.lpSum(close[p] for p in platforms) == 1
prob += pulp.lpSum(open_[k] for k in cand_ids)  == 1

# flux
x = pulp.LpVariable.dicts("x",
        ((p,c,pr) for p in platforms+cand_ids for c in clients for pr in products),
        lowBound=0, cat="Continuous")
y = pulp.LpVariable.dicts("y",
        ((s,p,r)  for s in suppliers for p in platforms+cand_ids for r in materials),
        lowBound=0, cat="Continuous")

# helper : prod_cost_active(p)
def prod_cost_p(p):
    return prod_cost_cand[p] if p in cand_ids else prod_cost[p]

# ---------------------- 4. Objectif -------------------------------------------
prob += (
    pulp.lpSum((0.3448*dist_cli[(p,c)]+33.591) * x[(p,c,pr)]
               for p in platforms+cand_ids for c in clients for pr in products)
  + pulp.lpSum((0.3448*dist_sup[(s,p)]+33.591) * y[(s,p,r)]
               for s in suppliers for p in platforms+cand_ids for r in materials)
  + pulp.lpSum(prod_cost_p(p) * x[(p,c,pr)]
               for p in platforms+cand_ids for c in clients for pr in products)
  + pulp.lpSum(purch_cost[(s,r)] * y[(s,p,r)]
               for s in suppliers for p in platforms+cand_ids for r in materials)
)

# ---------------------- 5. Contraintes ----------------------------------------
# 5.1 demande
for (c,pr), dem in demand.items():
    prob += pulp.lpSum(x[(p,c,pr)] for p in platforms+cand_ids) == dem

# 5.2 capacité PTF existantes
for p in platforms:
    prob += pulp.lpSum(x[(p,c,pr)] for c in clients for pr in products) <= cap_ptf[p] * (1 - close[p])

# 5.3 capacité candidats
big_M = 1e6
cap_inherited = pulp.lpSum(cap_ptf[p] * close[p] for p in platforms)
for k in cand_ids:
    prob += pulp.lpSum(x[(k,c,pr)] for c in clients for pr in products) <= big_M * open_[k]
    prob += pulp.lpSum(x[(k,c,pr)] for c in clients for pr in products) <= cap_inherited

# 5.4 bilans matière A/B
# pour PTF existantes
for p in platforms:
    for r, coef in zip(["A","B"], [alpha,beta]):
        prob += pulp.lpSum(y[(s,p,r)] for s in suppliers) == \
                pulp.lpSum(coef[(p,pr)] * x[(p,c,pr)] for c in clients for pr in products)

# pour candidats - utiliser les coefficients de la plateforme fermée
for k in cand_ids:
    for pr in products:
        for p in platforms:
            prob += pulp.lpSum(y[(s,k,"A")] for s in suppliers) >= \
                    alpha[(p,pr)] * x[(k,c,pr)] - big_M * (1 - close[p])
            prob += pulp.lpSum(y[(s,k,"A")] for s in suppliers) <= \
                    alpha[(p,pr)] * x[(k,c,pr)] + big_M * (1 - close[p])
            prob += pulp.lpSum(y[(s,k,"B")] for s in suppliers) >= \
                    beta[(p,pr)] * x[(k,c,pr)] - big_M * (1 - close[p])
            prob += pulp.lpSum(y[(s,k,"B")] for s in suppliers) <= \
                    beta[(p,pr)] * x[(k,c,pr)] + big_M * (1 - close[p])

# 5.5 capacité supplier
for s in suppliers:
    prob += pulp.lpSum(y[(s,p,"A")] + y[(s,p,"B")] for p in platforms+cand_ids) <= cap_sup[s]

# ---------------------- 6. Solve ---------------------------------------------
prob.solve(pulp.PULP_CBC_CMD(msg=True))
print("Statut :", pulp.LpStatus[prob.status], " | Coût total :", f"{pulp.value(prob.objective):,.0f} €")

# ---------------------- 7. Résultat clé ---------------------------------------
closed = [p for p in platforms if close[p].varValue>0.5][0]
opened = [k for k in cand_ids  if open_[k].varValue>0.5][0]
print("Plateforme fermée :", closed)
print("Plateforme ouverte :", opened, "coord=", coord[opened])

# ---------------------- 8. Bar chart coûts ------------------------------------
def cost_transport_cli(): return sum((0.3448*dist_cli[(p,c)]+33.591)*v.varValue
                                     for (p,c,pr),v in x.items() if v.varValue>1e-5)
def cost_transport_sup(): return sum((0.3448*dist_sup[(s,p)]+33.591)*v.varValue
                                     for (s,p,r),v in y.items() if v.varValue>1e-5)
def cost_prod():          return sum(prod_cost_p(p)*v.varValue
                                     for (p,c,pr),v in x.items() if v.varValue>1e-5)
def cost_purch():         return sum(purch_cost[(s,r)]*v.varValue
                                     for (s,p,r),v in y.items() if v.varValue>1e-5)

vals=[cost_transport_cli()+cost_transport_sup(), cost_prod(), cost_purch()]
plt.bar(["Transport","Production","Purchase"], vals, color=["skyblue","lightgreen","salmon"])
plt.ylabel("€"); plt.title("Coûts scénario optimal"); plt.tight_layout(); plt.show()

# ---------------------- 9. Carte Folium ---------------------------------------
m=folium.Map([46.5,2],zoom_start=5)
for c in clients:
    folium.Marker(coord[c],icon=folium.Icon(color="green"),popup=c).add_to(m)
for p in platforms:
    color="orange" if p==closed else "red"
    folium.Marker(coord[p],icon=folium.Icon(color=color),popup=p).add_to(m)
for s in suppliers:
    folium.Marker(coord[s],icon=folium.Icon(color="blue"),popup=s).add_to(m)
folium.Marker(coord[opened],icon=folium.Icon(color="purple"),popup=opened).add_to(m)

for (p,c,pr),v in x.items():
    if v.varValue>1e-5:
        folium.PolyLine([coord[p],coord[c]],color="green",weight=3).add_to(m)
for (s,p,r),v in y.items():
    if v.varValue>1e-5:
        folium.PolyLine([coord[s],coord[p]],color="blue",weight=3).add_to(m)

m.save("scenario_global_map.html"); webbrowser.open("scenario_global_map.html")
