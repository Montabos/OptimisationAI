import pandas as pd
import folium
import webbrowser

# === Lecture des données ===
site_df = pd.read_excel('Data2025.xlsx', sheet_name="Site's location")
customer_df = pd.read_excel('Data2025.xlsx', sheet_name="Customers Demand")
process_df = pd.read_excel('Data2025.xlsx', sheet_name="Process")
baseline_df = pd.read_excel('Data2025.xlsx', sheet_name="Base line")

# === Listes de repérage ===
coord = {row["Name"]: [row["Latitude"], row["Longitude"]] for _, row in site_df.iterrows()}
plateforms = site_df[site_df["Type"] == "Plateform"]["Name"].tolist()
customers = site_df[site_df["Type"] == "Customers"]["Name"].tolist()
suppliers = site_df[site_df["Type"] == "Supplier"]["Name"].tolist()

# === Quels produits sont fabriqués par chaque plateforme ? ===
produits_par_plateforme = (
    process_df.groupby("Plateforme")["Product"].apply(list).to_dict()
)

# === Mapping plateforme-client depuis "Base line" ===
# On garde uniquement les lignes Plateform → Customers
baseline_ptf_cust = baseline_df[
    (baseline_df["SiteType#Name"] == "Plateform") &
    (baseline_df["SiteType_1#Name"] == "Customers")
]

# === Création de la carte ===
m = folium.Map([49.8694, 2.209222], zoom_start=5)

# Ajout des clients et plateformes sur la carte
for c in customers:
    folium.Marker(
        location=coord[c],
        popup=c,
        icon=folium.Icon(color="green"),
    ).add_to(m)

for p in plateforms:
    folium.Marker(
        location=coord[p],
        popup=p,
        icon=folium.Icon(color="red"),
    ).add_to(m)

for s in suppliers:
    folium.Marker(
        location=coord[s],
        popup=s,
        icon=folium.Icon(color="blue"),
    ).add_to(m)

# === Fonction pour filtrer la visualisation par produit ===
def plot_product(product_filter=None):
    for _, row in baseline_ptf_cust.iterrows():
        ptf = row["GeoPoint#Name"]
        cust = row["GeoPoint_1#Name"]
        # Liste des produits produits par la plateforme
        if ptf not in produits_par_plateforme:
            continue
        produits = produits_par_plateforme[ptf]
        # Pour chaque produit (ou filtré)
        for prod in produits:
            if product_filter and prod != product_filter:
                continue
            # Demande du client pour ce produit
            demande = customer_df[
                (customer_df["Customer"] == cust) &
                (customer_df["Product"] == prod)
            ]
            if not demande.empty:
                volume = int(demande["Demande"].iloc[0])
                if volume > 0:
                    folium.PolyLine(
                        locations=[coord[ptf], coord[cust]],
                        color="green",
                        weight=3,
                        popup=f"{ptf} → {cust} Produit : {prod} Volume : {volume}"
                    ).add_to(m)

# === Appel de la fonction pour afficher TOUS les produits ou un seul (ex: 'P1') ===
# Pour tout afficher :
plot_product()
# Pour filtrer uniquement 'P1' :
# plot_product("P1")

# FLUX FOURNISSEUR → PLATEFORME (lignes bleues)
baseline_f2ptf = baseline_df[
    (baseline_df["SiteType#Name"] == "Supplier") &
    (baseline_df["SiteType_1#Name"] == "Plateform")
]

for _, row in baseline_f2ptf.iterrows():
    supplier = row["GeoPoint#Name"]
    ptf = row["GeoPoint_1#Name"]
    folium.PolyLine(
        locations=[coord[supplier], coord[ptf]],
        color="blue",
        weight=3,
        popup=f"{supplier} → {ptf}"
    ).add_to(m)

# === Affichage de la carte ===
m.save("base_line_map.html")
webbrowser.open("base_line_map.html")
