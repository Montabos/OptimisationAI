import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chemin vers le fichier
excel_path = "Data2025.xlsx"

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES FEUILLES
# ─────────────────────────────────────────────
cust_df      = pd.read_excel(excel_path, sheet_name="Customers Demand")     
base_df      = pd.read_excel(excel_path, sheet_name="Base line")            
process_df   = pd.read_excel(excel_path, sheet_name="Process")              
plat_df      = pd.read_excel(excel_path, sheet_name="Plateform")            
sup_df       = pd.read_excel(excel_path, sheet_name="Suppliers")            
transport_df = pd.read_excel(excel_path, sheet_name="Transport cost")       

# ─────────────────────────────────────────────
# 2. MAPPINGS (tirés de Base line)
# ─────────────────────────────────────────────
# client → plateforme
mask_cli = base_df["SiteType_1#Name"].str.startswith("C")
cli_ptf = (base_df[mask_cli]
           .set_index("GeoPoint_1#Name")["GeoPoint#Name"])

# plateforme → supplier
mask_ptf = (base_df["SiteType#Name"].eq("Supplier") &
            base_df["SiteType_1#Name"].str.startswith("P"))
ptf_sup = (base_df[mask_ptf]
           .set_index("GeoPoint_1#Name")["GeoPoint#Name"])

# ─────────────────────────────────────────────
# 3. VOLUMES PRODUITS PAR PLATEFORME
# ─────────────────────────────────────────────
demand_df = cust_df.copy()
demand_df["Plateform"] = demand_df["Customer"].map(cli_ptf)

vol_ptf_prod = demand_df.groupby(["Plateform", "Product"])["Demande"].sum()

# ─────────────────────────────────────────────
# 4. PRODUCTION COST TOTAL
# ─────────────────────────────────────────────
prod_cost_u = plat_df.set_index("Plateform")["Production cost (€/u)"]
production_cost_total = (vol_ptf_prod.groupby("Plateform").sum() * prod_cost_u).sum()

# ─────────────────────────────────────────────
# 5. ACHATS DE MATIÈRES PREMIÈRES ET PURCHASE COST
# ─────────────────────────────────────────────
proc = process_df.set_index(["Plateforme", "Product"])
need = vol_ptf_prod.reset_index().merge(proc, left_on=["Plateform","Product"],
                                        right_index=True)

need["Need_A"] = need["Demande"] * need["Raw material A"]
need["Need_B"] = need["Demande"] * need["Raw metrial B"]
need_ptf = need.groupby("Plateform")[["Need_A","Need_B"]].sum()

price_sup = sup_df.set_index(["Supplier","Product"])["Purchasing cost (€/u)"]

purchase_cost_total = 0
for ptf, row in need_ptf.iterrows():
    supplier = ptf_sup[ptf]                     
    cost_A   = row.Need_A * price_sup[(supplier, "A")]
    cost_B   = row.Need_B * price_sup[(supplier, "B")]
    purchase_cost_total += cost_A + cost_B

# ─────────────────────────────────────────────
# 6. TRANSPORT COST TOTAL (supplier→plateform + plateform→client)
# ─────────────────────────────────────────────
def get_cost_per_unit(src, dst):
    """retourne le coût €/u pour un flux src→dst (ordre important)"""
    match = transport_df[(transport_df["GeoPoint#Name"]   == src) &
                         (transport_df["GeoPoint_1#Name"] == dst)]
    if match.empty:
        return np.nan
    return match["€/U"].iloc[0]

# 6a. plateforme → client
trans_cost_ptf_cli = 0
for _, r in demand_df.iterrows():
    ptf, cli, qty = r["Plateform"], r["Customer"], r["Demande"]
    cost_u = get_cost_per_unit(ptf, cli)       
    trans_cost_ptf_cli += qty * cost_u

# 6b. supplier → plateforme (matières premières)
trans_cost_sup_ptf = 0
for ptf, row in need_ptf.iterrows():
    supplier = ptf_sup[ptf]
    qty_rm   = row.Need_A + row.Need_B         
    cost_u   = get_cost_per_unit(supplier, ptf)
    trans_cost_sup_ptf += qty_rm * cost_u

transport_cost_total = trans_cost_ptf_cli + trans_cost_sup_ptf

# ─────────────────────────────────────────────
# 7. BAR CHART DES COÛTS
# ─────────────────────────────────────────────
categories = ["Transport", "Production", "Purchase"]
values     = [transport_cost_total, production_cost_total, purchase_cost_total]

plt.figure(figsize=(7,4))
plt.bar(categories, values, color=["skyblue","lightgreen","salmon"])
plt.ylabel("€")
plt.title("Coûts globaux actuels par catégorie")
plt.tight_layout()
plt.show()

print(
    f"Transport : {transport_cost_total:,.0f} €\n"
    f"Production: {production_cost_total:,.0f} €\n"
    f"Purchase  : {purchase_cost_total:,.0f} €"
)

# Ajout du coût total global
print(f"Total cost: {transport_cost_total + production_cost_total + purchase_cost_total:,.0f} €")

