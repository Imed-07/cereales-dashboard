import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(page_title="🌍 Dashboard Céréales & Fret IA", layout="wide")
st.markdown('<h1 style="text-align:center; color:#2E8B57; font-weight:bold;">🌍 Dashboard Céréales & Fret IA</h1>', unsafe_allow_html=True)

# ==============================
# DONNÉES SIMULÉES (AJOUT DE L'ORGE)
# ==============================
SOURCES = {
    "Eurostat": {
        "Blé tendre": 248, "Blé dur": 288, "Maïs": 218, "Soja": 525, "Orge": 202
    },
    "USDA": {
        "Blé tendre": 270, "Blé dur": 310, "Maïs": 190, "Soja": 480, "Orge": 185
    },
    "FAO": {
        "Blé tendre": 260, "Blé dur": 295, "Maïs": 205, "Soja": 500, "Orge": 195
    },
    "Trading Econ": {
        "Blé tendre": 255, "Blé dur": 300, "Maïs": 200, "Soja": 510, "Orge": 198
    }
}

# Fret maritime (en $/tonne)
FRET_BASE = 28.5

@st.cache_data(ttl=3600)
def generer_historique(actif: str, jours: int = 60):
    np.random.seed(42)
    dates = [datetime.today() - timedelta(days=x) for x in range(jours)][::-1]
    
    if actif == "Fret maritime":
        base = FRET_BASE
        prix = []
        for i in range(jours):
            trend = base * (1 + 0.0002 * i)
            seasonal = 2 * np.sin(2 * np.pi * i / 30)
            noise = np.random.normal(0, 3)
            prix.append(max(trend + seasonal + noise, base * 0.7))
        return pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Prix": np.round(prix, 2),
            "Volume": np.random.randint(5000, 12000, jours)
        })
    else:
        base = np.mean([src[actif] for src in SOURCES.values()])
        prix = []
        for i in range(jours):
            trend = base * (1 + 0.0005 * i)
            seasonal = 5 * np.sin(2 * np.pi * i / 7)
            noise = np.random.normal(0, 8)
            prix.append(max(trend + seasonal + noise, base * 0.8))
        return pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Prix": np.round(prix, 2),
            "Volume": np.random.randint(8000, 18000, jours)
        })

# Liste complète des actifs
ACTIFS_DISPONIBLES = ["Blé tendre", "Blé dur", "Maïs", "Soja", "Orge", "Fret maritime"]

# ==============================
# CONFIGURATION UTILISATEUR
# ==============================
with st.expander("⚙️ Configuration", expanded=True):
    actif = st.selectbox("Actif", ACTIFS_DISPONIBLES, key="actif_selector")
    zone = st.radio("Zone", ["Europe 🇪🇺", "USA 🇺🇸"], horizontal=True, key="zone_selector")

# Déterminer devise et zone
if actif == "Fret maritime":
    devise = "$"
    zone_nom = "Global"
else:
    zone_nom = "Europe" if "Europe" in zone else "USA"
    devise = "€" if zone_nom == "Europe" else "$"

st.markdown(
    f'<div style="text-align:center; padding:12px; background:#f8fff8; border-radius:10px; margin:1rem 0; border:1px solid #e8f5e8;">'
    f'<strong>Données actives :</strong> {actif} • {zone_nom} • {devise}/t</div>',
    unsafe_allow_html=True
)

# ==============================
# CHARGEMENT DES DONNÉES
# ==============================
df_hist = generer_historique(actif)
prix_actuel = df_hist["Prix"].iloc[-1]
volatilite = df_hist["Prix"].std()

# ==============================
# INDICATEURS CLÉS
# ==============================
st.subheader("💰 Indicateurs du marché")
col1, col2, col3 = st.columns(3)
col1.metric("Prix actuel", f"{prix_actuel:.1f} {devise}/t")
col2.metric("Volume moyen", f"{df_hist['Volume'].mean():,.0f} t")
col3.metric("Volatilité", f"{volatilite:.1f}")

# ==============================
# HISTORIQUE RÉCENT
# ==============================
st.subheader(f"📈 Historique : {actif} (30 derniers jours)")
df_recent = df_hist.tail(30).copy()
df_recent["Date"] = pd.to_datetime(df_recent["Date"])
st.line_chart(df_recent.set_index("Date")["Prix"], use_container_width=True)

# ==============================
# COMPARAISON DES SOURCES
# ==============================
if actif != "Fret maritime":
    st.subheader("🌍 Comparaison des sources")
    sources_data = {src: f"{val[actif]:.1f}" for src, val in SOURCES.items()}
    st.dataframe(pd.DataFrame([sources_data]), use_container_width=True)
else:
    st.info("📊 Le fret maritime est un indice global (Baltic Dry Index simulé).")

# ==============================
# PRÉVISION
# ==============================
st.subheader("🔮 Prévision sur 15 jours")

if st.button("✨ Générer la prévision", key="btn_prevision"):
    with st.status("🧠 Entraînement du modèle...", expanded=True) as status:
        status.write("📊 Préparation des données...")
        df_pred = df_hist.copy()
        df_pred['Jour'] = np.arange(len(df_pred))
        
        status.write("📈 Entraînement...")
        model = LinearRegression()
        model.fit(df_pred[['Jour']], df_pred['Prix'])
        
        status.write("🔮 Prévision...")
        futur_X = np.arange(len(df_pred), len(df_pred) + 15).reshape(-1, 1)
        y_pred = model.predict(futur_X)
        dates_futures = pd.date_range(start=df_hist['Date'].iloc[-1], periods=16)[1:]
        
        status.update(label="✅ Prêt !", state="complete")
    
    historique = df_hist[["Date", "Prix"]].copy()
    historique.columns = ["Date", "Valeur"]
    historique["Type"] = "Historique"
    
    prevision = pd.DataFrame({
        "Date": dates_futures.strftime("%Y-%m-%d"),
        "Valeur": np.round(y_pred, 2),
        "Type": "Prévision"
    })
    
    combo = pd.concat([historique, prevision], ignore_index=True)
    combo["Date"] = pd.to_datetime(combo["Date"])
    combo = combo.sort_values("Date")
    
    st.subheader(f"📊 Historique + Prévision : {actif}")
    st.line_chart(combo.set_index("Date")["Valeur"], use_container_width=True)
    
    with st.expander("📋 Données brutes"):
        st.dataframe(combo, use_container_width=True)
    
    st.session_state['prevision'] = y_pred[-1]
    st.session_state['prix_actuel'] = prix_actuel

# ==============================
# RAG : ANALYSE CONTEXTUELLE
# ==============================
st.subheader("🧠 Recommandation IA (RAG)")

def recuperer_actualites(actif: str):
    if actif == "Fret maritime":
        return [
            "Tensions géopolitiques affectent les routes maritimes.",
            "Capacité portuaire mondiale sous pression.",
            "Demande de navires bulk carriers en hausse."
        ]
    elif "Blé" in actif:
        return [
            "Récoltes européennes impactées par la sécheresse.",
            "Exportations russes de blé en hausse.",
            "Stocks mondiaux de céréales stables."
        ]
    elif actif == "Maïs":
        return [
            "Conditions climatiques favorables au Brésil.",
            "Demande chinoise en légère baisse.",
            "Subventions américaines maintenues."
        ]
    elif actif == "Soja":
        return [
            "Récolte record attendue en Amérique du Sud.",
            "Demande chinoise robuste pour l'huile végétale.",
            "Concurrence colza/soja sur les marchés."
        ]
    elif actif == "Orge":
        return [
            "Demande brassicole européenne en hausse.",
            "Récoltes d'orge fourragère excédentaires.",
            "Subventions PAC stables pour les céréaliers."
        ]
    return [
        "Marché stable avec faible volatilité.",
        "Aucun événement majeur rapporté.",
        "Tendances techniques neutres."
    ]

if st.button("🔍 Générer analyse contextuelle", key="btn_rag"):
    with st.status("🌍 Recherche d'actualités + analyse IA...", expanded=True) as rag_status:
        rag_status.write("🌐 Récupération des actualités...")
        actualites = recuperer_actualites(actif)
        rag_status.write(f"✅ {len(actualites)} articles trouvés")
        rag_status.write("🧠 Analyse contextuelle...")
        prix = st.session_state.get('prix_actuel', prix_actuel)
        prev = st.session_state.get('prevision', prix * 1.02)
        tendance = "hausse" if prev > prix else "baisse"
        vol = "faible" if volatilite < (5 if actif == "Fret maritime" else 10) else "élevée"
        rag_status.write("✅ Recommandation prête !")
    
    rec = f"**Analyse IA** :\n- Prix actuel : {prix:.1f} {devise}/t\n- Prévision : {prev:.1f} {devise}/t ({tendance})\n- Volatilité : {vol}"
    if "sécheresse" in " ".join(actualites).lower() or "tensions" in " ".join(actualites).lower():
        rec += "\n- ⚠️ Facteur de risque détecté."
    if vol == "faible" and tendance == "hausse":
        rec += "\n\n✅ **Recommandation** : Opportunité d'achat."
    else:
        rec += "\n\n⚠️ **Recommandation** : Surveillance accrue."
    
    st.success(rec)
    
    st.subheader("📰 Sources consultées")
    for i, act in enumerate(actualites, 1):
        st.caption(f"{i}. {act}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption(f"**Mise à jour** : {datetime.now().strftime('%d/%m/%Y %H:%M')} • Données simulées (Eurostat, USDA, FAO, Baltic Dry Index)")
st.caption("✅ Prévision linéaire • 🌐 Céréales + Fret • 🧠 RAG contextuel")
