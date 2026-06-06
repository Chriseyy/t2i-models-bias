"""
cluster_and_visualize.py
========================
UNIVERSAL-VERSION FÜR CLIP & DINOv3:
1. AUTOMATISCHE STRUKTURIERUNG: Erkennt den Vektor-Typ und speichert alle Plots
   sauber getrennt unter:
   - outputs/plots/cluster/clip_plots/
   - outputs/plots/cluster/dinov3_plots/
2. INTER-MODELL-MAPPING: Erstellt Einzellaufe (4 Attribute), 1-vs-1 Vergleiche 
   (alle 4 Attribute paarweise) und das globale "Alle gegen Alle" (5 Dimensionen).
"""

import pickle
import glob
import re
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# =============================================================
# KONFIGURATION: STEUERE HIER WELCHEN VEKTORRAUM DU AUSWERTEN WILLST
# =============================================================
# Setze hier entweder "clip" oder "dinov3" ein!
FEATURE_TYPE = "dinov3"  # oder "clip"

# =============================================================
# PFADE STRUKTURIEREN (Automatisch angepasst)
# =============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != 'analysis' else SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Zielverzeichnis wird dynamisch je nach FEATURE_TYPE erzeugt
PLOTS_DIR = OUTPUT_DIR / "plots" / "cluster" / f"{FEATURE_TYPE}_plots"

# Makro-Ergebnisdateien als Datenquellen
OLLAMA_CSV = OUTPUT_DIR / "ollama_results.csv"
DEEPFACE_CSV = OUTPUT_DIR / "deepface_results.csv"
SKIN_CSV = OUTPUT_DIR / "skin_metrics_results.csv"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def load_pkl_file(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def clean_strings(df):
    """Bereinigt kategoriale Spalten für einheitliche Legenden"""
    for col in df.columns:
        if any(keyword in col for keyword in ['Race', 'Gender', 'Scale', 'MST']):
            df[col] = df[col].astype(str).str.title().str.strip()
            df[col] = df[col].replace('Nan', np.nan).replace('Unknown', np.nan)
    return df

def build_macro_metadata():
    """Verschmilzt alle Makro-Ergebnisdateien zu einer lückenlosen Datenbasis"""
    print("📋 Baue Makro-Metadaten-Verzeichnis aus CSV-Dateien auf...")
    master_meta = None

    if DEEPFACE_CSV.exists():
        df_df = pd.read_csv(DEEPFACE_CSV).drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
        df_df = clean_strings(df_df)
        master_meta = df_df[['Image_Name', 'T2I_Model', 'DeepFace_Gender', 'DeepFace_Race']]
        print(f"   -> DeepFace geladen: {len(master_meta)} Einträge.")

    if SKIN_CSV.exists():
        df_skin = pd.read_csv(SKIN_CSV).drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
        df_skin = clean_strings(df_skin)
        df_skin_sub = df_skin[['Image_Name', 'T2I_Model', 'ITA_Scale_MST', 'MonkScale_RGB']]
        if master_meta is not None:
            master_meta = pd.merge(master_meta, df_skin_sub, on=['Image_Name', 'T2I_Model'], how='outer')
        else:
            master_meta = df_skin_sub
        print(f"   -> Skin Metrics geladen. Gesamt-Metadaten: {len(master_meta)} Einträge.")

    if OLLAMA_CSV.exists():
        df_ol = pd.read_csv(OLLAMA_CSV).drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
        df_ol = clean_strings(df_ol)
        df_ol_sub = df_ol[['Image_Name', 'T2I_Model', 'VLM_Gender', 'VLM_Race', 'VLM_MST']]
        if master_meta is not None:
            master_meta = pd.merge(master_meta, df_ol_sub, on=['Image_Name', 'T2I_Model'], how='outer')
        else:
            master_meta = df_ol_sub
        print(f"   -> Ollama VLMs geladen. Gesamt-Metadaten: {len(master_meta)} Einträge.")

    return master_meta

# =============================================================
# MAIN PROCESSING
# =============================================================
def main():
    print("=" * 70)
    print(f"🔮 STARTE AUTOMATISCHE VEKTORRAUM-ANALYSE FÜR: [{FEATURE_TYPE.upper()}]")
    print("=" * 70)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    macro_meta = build_macro_metadata()
    if macro_meta is None:
        print("❌ Keine Ergebnis-CSVs im outputs-Ordner gefunden!")
        return

    # Filtert dynamisch nach clip_embeddings_*.pkl oder dinov3_embeddings_*.pkl
    pkl_pattern = str(OUTPUT_DIR / f"{FEATURE_TYPE}_embeddings_*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"❌ Keine Model-Vektoren vom Typ '{FEATURE_TYPE}_embeddings_*.pkl' in {OUTPUT_DIR} gefunden.")
        print(f"Bitte lasse zuerst das entsprechende Extraktions-Skript laufen!")
        return

    models_available = {}
    for f in pkl_files:
        match = re.search(f"{FEATURE_TYPE}_embeddings_(.+)\.pkl", Path(f).name)
        if match:
            models_available[match.group(1)] = Path(f)

    print(f"✅ Verfügbare Vektor-Pakete für das Clustering: {list(models_available.keys())}")

    # =============================================================
    # TEIL 1: EINZEL-ORDNER ANALYSEN
    # =============================================================
    print("\n📊 --- TEIL 1: GENERIERE MULTI-BIAS-PLOTS PRO MODELL ---")
    
    for model_name, pkl_path in models_available.items():
        print(f"\nProcessing Einzel-Modell: [{model_name.upper()}]")
        data_list = load_pkl_file(pkl_path)
        
        embeddings = np.array([item["Embedding"] for item in data_list])
        df_vectors = pd.DataFrame({
            "Image_Name": [item["Image_Name"] for item in data_list],
            "T2I_Model": [item["T2I_Model"] for item in data_list],
            "Prompt_Subject": [item["Prompt_Subject"] for item in data_list]
        })

        df = pd.merge(df_vectors, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

        perplexity_val = min(30, max(5, len(df) // 5))
        tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
        X_2d = tsne.fit_transform(embeddings)
        df["x"], df["y"] = X_2d[:, 0], X_2d[:, 1]

        n_prompts = df["Prompt_Subject"].nunique()
        kmeans = KMeans(n_clusters=n_prompts, random_state=42, n_init="auto")
        df["Visual_Cluster_ID"] = kmeans.fit_transform(X_2d).argmin(axis=1)
        centroids = df.groupby("Visual_Cluster_ID")[["x", "y"]].mean().reset_index()

        prefix = f"{FEATURE_TYPE.upper()}_{model_name}"

        # A) Plot Thema
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x="x", y="y", hue="Prompt_Subject", style="Prompt_Subject", s=80, alpha=0.75, palette="Set1", ax=ax)
        for _, c in centroids.iterrows():
            ax.text(c["x"], c["y"], f"Insel {int(c['Visual_Cluster_ID'])}", color="black", weight="bold", fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
        ax.set_title(f"{FEATURE_TYPE.upper()} Vektorraum (Themen): {model_name.upper()}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{prefix}_CLUSTER_BY_PROMPT.png", dpi=300)
        plt.close(fig)

        # B) Plot Gender
        if "DeepFace_Gender" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(data=df, x="x", y="y", hue="DeepFace_Gender", style="DeepFace_Gender", s=80, alpha=0.75, palette="Dark2", ax=ax)
            ax.set_title(f"{FEATURE_TYPE.upper()} Vektorraum (KI-Gender): {model_name.upper()}")
            ax.legend(title="DeepFace Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{prefix}_CLUSTER_BY_GENDER.png", dpi=300)
            plt.close(fig)

        # C) Plot Race
        if "DeepFace_Race" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(data=df, x="x", y="y", hue="DeepFace_Race", style="DeepFace_Race", s=80, alpha=0.75, palette="tab10", ax=ax)
            ax.set_title(f"{FEATURE_TYPE.upper()} Vektorraum (KI-Ethnie): {model_name.upper()}")
            ax.legend(title="DeepFace Race", bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{prefix}_CLUSTER_BY_RACE.png", dpi=300)
            plt.close(fig)

        # D) Plot Hautton
        if "ITA_Scale_MST" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            mst_order = [str(i) for i in range(1, 11)] + ["Error", "Unclear"]
            present_order = [o for o in mst_order if o in df["ITA_Scale_MST"].unique()]
            sns.scatterplot(data=df, x="x", y="y", hue="ITA_Scale_MST", hue_order=present_order, s=80, alpha=0.75, palette="rocket_r", ax=ax)
            ax.set_title(f"{FEATURE_TYPE.upper()} Vektorraum (Hautton MST): {model_name.upper()}")
            ax.legend(title="ITA Scale", bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{prefix}_CLUSTER_BY_HAUTTON.png", dpi=300)
            plt.close(fig)

        # Matrizen speichern
        pd.crosstab(df["Visual_Cluster_ID"], df["Prompt_Subject"], margins=True).to_csv(PLOTS_DIR / f"{prefix}_MATRIX_PROMPT.csv")
        if "DeepFace_Gender" in df.columns:
            pd.crosstab(df["Visual_Cluster_ID"], df["DeepFace_Gender"], margins=True).to_csv(PLOTS_DIR / f"{prefix}_MATRIX_GENDER.csv")
        if "DeepFace_Race" in df.columns:
            pd.crosstab(df["Visual_Cluster_ID"], df["DeepFace_Race"], margins=True).to_csv(PLOTS_DIR / f"{prefix}_MATRIX_RACE.csv")
        if "ITA_Scale_MST" in df.columns:
            pd.crosstab(df["Visual_Cluster_ID"], df["ITA_Scale_MST"], margins=True).to_csv(PLOTS_DIR / f"{prefix}_MATRIX_HAUTTON.csv")

        print(f"   -> Einzelplots und Matrizen für {model_name} gesichert.")

    # =============================================================
    # TEIL 2: ERWEITERTE 1-VS-1 PLOTS (MULTIDIMENSIONAL PAARWEISE)
    # =============================================================
    if len(models_available) >= 2:
        print("\n⚔️ --- TEIL 2: GENERIERE ERWEITERTE 1-VS-1 COMPARISONS ---")
        model_pairs = list(combinations(models_available.keys(), 2))

        for modA, modB in model_pairs:
            print(f"Berechne gemeinsamen Raum für: [{modA.upper()}] vs [{modB.upper()}]")
            dataA = load_pkl_file(models_available[modA])
            dataB = load_pkl_file(models_available[modB])
            combined_data = dataA + dataB
            
            embeddings_comb = np.array([item["Embedding"] for item in combined_data])
            df_comb_vec = pd.DataFrame({
                "Image_Name": [item["Image_Name"] for item in combined_data],
                "T2I_Model": [item["T2I_Model"] for item in combined_data],
                "Prompt_Subject": [item["Prompt_Subject"] for item in combined_data]
            })
            df_comb = pd.merge(df_comb_vec, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

            tsne_comb = TSNE(n_components=2, perplexity=min(30, max(5, len(df_comb) // 5)), random_state=42)
            X_2d_comb = tsne_comb.fit_transform(embeddings_comb)
            df_comb["x"], df_comb["y"] = X_2d_comb[:, 0], X_2d_comb[:, 1]

            attributes_to_plot = [
                ("Prompt_Subject", "PROMPT", "Set1", "Thema"),
                ("DeepFace_Gender", "GENDER", "Dark2", "KI-Gender"),
                ("DeepFace_Race", "RACE", "tab10", "KI-Ethnie"),
                ("ITA_Scale_MST", "HAUTTON", "rocket_r", "Hautton MST")
            ]

            for col_attr, filename_suffix, pal, title_label in attributes_to_plot:
                if col_attr in df_comb.columns:
                    fig, ax = plt.subplots(figsize=(14, 9))
                    sns.scatterplot(data=df_comb, x="x", y="y", hue=col_attr, style="T2I_Model", 
                                    markers=["o", "X"], s=110, alpha=0.75, palette=pal, ax=ax)
                    ax.set_title(f"1-vs-1 Raum ({title_label} | {FEATURE_TYPE.upper()}): {modA.upper()} vs {modB.upper()}")
                    ax.set_xlabel("Gemeinsames t-SNE X")
                    ax.set_ylabel("Gemeinsames t-SNE Y")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    fig.tight_layout()
                    fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_COMPARISON_1vs1_{modA}_vs_{modB}_{filename_suffix}.png", dpi=300)
                    plt.close(fig)

    # =============================================================
    # TEIL 3: DER ULTIMATIVE GLOBAL-PLOT (ALLE GEGEN ALLE)
    # =============================================================
    print("\n🌌 --- TEIL 3: GENERIERE GLOBALEN 'ALLE-GEGEN-ALLE' VEKTORRAUM ---")
    all_combined_data = []
    for model_name, pkl_path in models_available.items():
        all_combined_data.extend(load_pkl_file(pkl_path))

    print(f"   Insgesamt {len(all_combined_data)} Bilder aus ALLEN Modellen geladen. Berechne globales t-SNE...")
    
    embeddings_global = np.array([item["Embedding"] for item in all_combined_data])
    df_global_vec = pd.DataFrame({
        "Image_Name": [item["Image_Name"] for item in all_combined_data],
        "T2I_Model": [item["T2I_Model"] for item in all_combined_data],
        "Prompt_Subject": [item["Prompt_Subject"] for item in all_combined_data]
    })
    df_global = pd.merge(df_global_vec, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

    tsne_global = TSNE(n_components=2, perplexity=min(50, max(5, len(df_global) // 5)), random_state=42)
    X_2d_global = tsne_global.fit_transform(embeddings_global)
    df_global["x"], df_global["y"] = X_2d_global[:, 0], X_2d_global[:, 1]

    # Global-Plot 0: Die Modell-Galaxien
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.scatterplot(data=df_global, x="x", y="y", hue="T2I_Model", style="T2I_Model", s=80, alpha=0.7, palette="Set2", ax=ax)
    ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Die Modell-Galaxien (Alle KIs im Vergleich)")
    ax.legend(title="T2I Bildgenerator", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_0_ALL_MODELS.png", dpi=300)
    plt.close(fig)

    # Global-Plot 1: Nach Prompt
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.scatterplot(data=df_global, x="x", y="y", hue="Prompt_Subject", style="T2I_Model", s=80, alpha=0.7, palette="Set1", ax=ax)
    ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Konzept-Verteilung über alle Prompts")
    ax.legend(title="Prompts", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_1_BY_PROMPT.png", dpi=300)
    plt.close(fig)

    # Global-Plot 2: Nach KI-Gender
    if "DeepFace_Gender" in df_global.columns:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.scatterplot(data=df_global, x="x", y="y", hue="DeepFace_Gender", style="T2I_Model", s=80, alpha=0.7, palette="Dark2", ax=ax)
        ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Geschlechter-Strukturen über alle KIs")
        ax.legend(title="DeepFace Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_2_BY_GENDER.png", dpi=300)
        plt.close(fig)

    # Global-Plot 3: Nach KI-Race
    if "DeepFace_Race" in df_global.columns:
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.scatterplot(data=df_global, x="x", y="y", hue="DeepFace_Race", style="T2I_Model", s=80, alpha=0.7, palette="tab10", ax=ax)
        ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Ethnische Strukturen über alle KIs")
        ax.legend(title="DeepFace Race", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_3_BY_RACE.png", dpi=300)
        plt.close(fig)

    # Global-Plot 4: Nach Hautton (MST)
    if "ITA_Scale_MST" in df_global.columns:
        fig, ax = plt.subplots(figsize=(14, 10))
        mst_order = [str(i) for i in range(1, 11)] + ["Error", "Unclear"]
        present_order = [o for o in mst_order if o in df_global["ITA_Scale_MST"].unique()]
        sns.scatterplot(data=df_global, x="x", y="y", hue="ITA_Scale_MST", hue_order=present_order, style="T2I_Model", s=80, alpha=0.7, palette="rocket_r", ax=ax)
        ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Hautton-Gradienten über alle KIs")
        ax.legend(title="ITA Scale", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_4_BY_HAUTTON.png", dpi=300)
        plt.close(fig)

    print("\n" + "="*80)
    print(f"🎉 EXCELLENT! Das multidimensionale Mapping für {FEATURE_TYPE.upper()} wurde erfolgreich beendet!")
    print(f"Sämtliche Auswertungen befinden sich in: {PLOTS_DIR.resolve()}")
    print("="*80)

if __name__ == "__main__":
    main()