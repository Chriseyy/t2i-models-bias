"""
cluster_and_visualize_umap.py
=============================
UMAP-VERSION FÜR HIGH-QUALITY EMBEDDING CLUSTERING:
1. GLOBAL STRUCTURE RETENTION: Ersetzt t-SNE durch UMAP, um reale semantische Abstände 
   zwischen den Clustern und Modell-Galaxien physikalisch interpretierbar zu machen.
2. AUTOMATISCHE STRUKTURIERUNG: Legt alle Ergebnisse getrennt in UMAP-spezifischen Ordnern ab:
   - outputs/plots/cluster/clip_plots_umap/
   - outputs/plots/cluster/dinov3_plots_umap/
3. MULTI-VLM PIVOTING & COSSIM REPORT: Behält alle lückenlosen Makro-Metadaten-Analysen bei.
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
import umap  
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================
# KONFIGURATION: HIER VEKTORRAUM WÄHLEN ("dinov3" oder "clip")
# =============================================================
FEATURE_TYPE = "dinov3"  
FEATURE_TYPE = "clip"  

# =============================================================
# PFADE STRUKTURIEREN (Dynamisch auf _umap angepasst)
# =============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != 'analysis' else SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Zielverzeichnis exakt nach User-Wunsch auf f"{FEATURE_TYPE}_plots_umap" gesetzt
PLOTS_DIR = OUTPUT_DIR / "plots" / "cluster" / f"{FEATURE_TYPE}_plots_umap"

OLLAMA_CSV = OUTPUT_DIR / "ollama_results.csv"
DEEPFACE_CSV = OUTPUT_DIR / "deepface_results.csv"
SKIN_CSV = OUTPUT_DIR / "skin_metrics_results.csv"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def load_pkl_file(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def clean_strings(df):
    """Bereinigt kategoriale Spalten robust. Schützt T2I_Model vor Fehl-Kapitalisierung."""
    for col in df.columns:
        if col == "T2I_Model":
            df[col] = df[col].astype(str).str.strip().str.lower()
            continue
            
        if any(keyword in col for keyword in ['Race', 'Gender', 'Scale', 'MST', 'Model']):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: x.split('.')[0] if x.replace('.','',1).isdigit() and '.' in x else x)
            df[col] = df[col].str.title()
            df[col] = df[col].replace({'Nan': np.nan, 'Unknown': np.nan, 'None': np.nan})
    return df

def build_macro_metadata():
    print("Baue Makro-Metadaten-Verzeichnis aus CSV-Dateien auf...")
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
        df_ol = pd.read_csv(OLLAMA_CSV)
        df_ol = clean_strings(df_ol)
        
        df_ol['VLM_Model_Clean'] = df_ol['VLM_Model'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', str(x)))
        df_ol = df_ol.drop_duplicates(subset=['Image_Name', 'T2I_Model', 'VLM_Model_Clean'], keep='last')
        
        print(f"   ⚙️ Gefundene VLMs in Ollama-CSV: {df_ol['VLM_Model_Clean'].unique().tolist()}")
        
        df_ol_pivot = df_ol.pivot(
            index=['Image_Name', 'T2I_Model'], 
            columns='VLM_Model_Clean', 
            values=['VLM_Gender', 'VLM_Race', 'VLM_MST']
        )
        df_ol_pivot.columns = [f"{col[1]}_{col[0].replace('VLM_', '')}" for col in df_ol_pivot.columns]
        df_ol_pivot = df_ol_pivot.reset_index()
        
        if master_meta is not None:
            master_meta = pd.merge(master_meta, df_ol_pivot, on=['Image_Name', 'T2I_Model'], how='outer')
        else:
            master_meta = df_ol_pivot
        print(f"   -> Ollama Multi-VLMs erfolgreich integriert. Gesamtzeilen: {len(master_meta)}")

    return master_meta

def main():
    print("=" * 70)
    print(f"STARTE AUTOMATISCHE COMPACT-MAP-ANALYSE (UMAP) FÜR: [{FEATURE_TYPE.upper()}]")
    print("=" * 70)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    macro_meta = build_macro_metadata()
    if macro_meta is None:
        print("❌ Keine Ergebnis-CSVs im outputs-Ordner gefunden!")
        return

    pkl_pattern = str(OUTPUT_DIR / f"{FEATURE_TYPE}_embeddings_*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"Keine Model-Vektoren vom Typ '{FEATURE_TYPE}_embeddings_*.pkl' in {OUTPUT_DIR} gefunden.")
        return

    models_available = {re.search(f"{FEATURE_TYPE}_embeddings_(.+)\.pkl", Path(f).name).group(1): Path(f) for f in pkl_files if re.search(f"{FEATURE_TYPE}_embeddings_(.+)\.pkl", Path(f).name)}
    print(f"Verfügbare Vektor-Pakete für das Clustering: {list(models_available.keys())}")

    # Basis-Attribute definieren
    attributes_to_plot = [
        ("Prompt_Subject", "PROMPT", "Set1", "Thema"),
        ("DeepFace_Gender", "DF_GENDER", "Dark2", "DeepFace Gender"),
        ("DeepFace_Race", "DF_RACE", "tab10", "DeepFace Ethnie"),
        ("ITA_Scale_MST", "SKIN_MST", "rocket_r", "Skin Metrics MST"),
    ]

    # Dynamisch nach den aufgefalteten VLM-Spalten suchen
    palettes_pool = ["Accent", "Paired", "Set3", "flare", "Pastel1", "muted"]
    pal_idx = 0
    
    vlm_cols = [c for c in macro_meta.columns if any(c.endswith(sfx) for sfx in ['_Gender', '_Race', '_MST'])]
    vlm_prefixes = sorted(list(set([c.split('_')[0] for c in vlm_cols if '_' in c and c.split('_')[0] not in ['DeepFace', 'ITA']])))
    
    for prefix in vlm_prefixes:
        if f"{prefix}_Gender" in macro_meta.columns:
            attributes_to_plot.append((f"{prefix}_Gender", f"{prefix.upper()}_GENDER", palettes_pool[pal_idx % len(palettes_pool)], f"{prefix} VLM Gender"))
            pal_idx += 1
        if f"{prefix}_Race" in macro_meta.columns:
            attributes_to_plot.append((f"{prefix}_Race", f"{prefix.upper()}_RACE", palettes_pool[pal_idx % len(palettes_pool)], f"{prefix} VLM Ethnie"))
            pal_idx += 1
        if f"{prefix}_MST" in macro_meta.columns:
            attributes_to_plot.append((f"{prefix}_MST", f"{prefix.upper()}_MST", palettes_pool[pal_idx % len(palettes_pool)], f"{prefix} VLM MST Hautton"))
            pal_idx += 1

    # =============================================================
    # TEIL 1: EINZEL-ORDNER ANALYSEN
    # =============================================================
    print("\n========= TEIL 1: EINZEL-PLOTS & MATRIZEN PRO MODELL (UMAP) =========")
    for model_name, pkl_path in models_available.items():
        print(f"Processing Einzel-Modell: [{model_name.upper()}]")
        data_list = load_pkl_file(pkl_path)
        
        embeddings = np.array([item["Embedding"] for item in data_list])
        df_vectors = pd.DataFrame({
            "Image_Name": [item["Image_Name"] for item in data_list],
            "T2I_Model": [item["T2I_Model"] for item in data_list],
            "Prompt_Subject": [item["Prompt_Subject"] for item in data_list]
        })
        df_vectors["T2I_Model"] = df_vectors["T2I_Model"].astype(str).str.strip().str.lower()
        
        if "Prompt_Subject" in macro_meta.columns:
            macro_meta = macro_meta.drop(columns=["Prompt_Subject"])
            
        df = pd.merge(df_vectors, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

        # UMAP 2D Reduktion robust konfiguriert
        n_neighbors_val = min(15, max(2, len(df) // 10))
        reducer = umap.UMAP(n_neighbors=n_neighbors_val, min_dist=0.1, n_components=2, random_state=42)
        X_2d = reducer.fit_transform(embeddings)
        df["x"], df["y"] = X_2d[:, 0], X_2d[:, 1]

        # K-Means Inseln bestimmen
        n_prompts = df["Prompt_Subject"].nunique()
        kmeans = KMeans(n_clusters=n_prompts, random_state=42, n_init="auto")
        df["Visual_Cluster_ID"] = kmeans.fit_transform(X_2d).argmin(axis=1)
        centroids = df.groupby("Visual_Cluster_ID")[["x", "y"]].mean().reset_index()

        for col_attr, filename_suffix, pal, title_label in attributes_to_plot:
            if col_attr in df.columns and not df[col_attr].dropna().empty:
                fig, ax = plt.subplots(figsize=(11, 7))
                
                h_order = None
                if "MST" in col_attr or "Scale" in col_attr:
                    h_order = [str(i) for i in range(1, 11)]
                    h_order = [o for o in h_order if o in df[col_attr].unique()]

                sns.scatterplot(data=df, x="x", y="y", hue=col_attr, hue_order=h_order, style=col_attr, s=85, alpha=0.8, palette=pal, ax=ax)
                
                if col_attr == "Prompt_Subject":
                    for _, c in centroids.iterrows():
                        ax.text(c["x"], c["y"], f"Insel {int(c['Visual_Cluster_ID'])}", color="black", weight="bold", fontsize=11,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
                
                ax.set_title(f"{FEATURE_TYPE.upper()} UMAP - {title_label}: {model_name.upper()}")
                ax.set_xlabel("UMAP Dimension X")
                ax.set_ylabel("UMAP Dimension Y")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                fig.tight_layout()
                fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_{model_name}_SINGLE_{filename_suffix}.png", dpi=300)
                plt.close(fig)

                pd.crosstab(df["Visual_Cluster_ID"], df[col_attr], margins=True).to_csv(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_{model_name}_MATRIX_{filename_suffix}.csv")

    # =============================================================
    # TEIL 2: ERWEITERTE 1-VS-1 COMPARISONS (UMAP GEMEINSAM)
    # =============================================================
    if len(models_available) >= 2:
        print("\n========= TEIL 2: PAARWEISE GEMEINSAME 1-VS-1 RÄUME (UMAP) =========")
        for modA, modB in list(combinations(models_available.keys(), 2)):
            print(f"Berechne gemeinsamen UMAP-Raum: [{modA.upper()}] vs [{modB.upper()}]")
            combined_data = load_pkl_file(models_available[modA]) + load_pkl_file(models_available[modB])
            
            embeddings_comb = np.array([item["Embedding"] for item in combined_data])
            df_comb_vec = pd.DataFrame({
                "Image_Name": [item["Image_Name"] for item in combined_data],
                "T2I_Model": [item["T2I_Model"] for item in combined_data],
                "Prompt_Subject": [item["Prompt_Subject"] for item in combined_data]
            })
            df_comb_vec["T2I_Model"] = df_comb_vec["T2I_Model"].astype(str).str.strip().str.lower()
            df_comb = pd.merge(df_comb_vec, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

            n_neighbors_val = min(15, max(2, len(df_comb) // 10))
            reducer_comb = umap.UMAP(n_neighbors=n_neighbors_val, min_dist=0.1, n_components=2, random_state=42)
            X_2d_comb = reducer_comb.fit_transform(embeddings_comb)
            df_comb["x"], df_comb["y"] = X_2d_comb[:, 0], X_2d_comb[:, 1]

            for col_attr, filename_suffix, pal, title_label in attributes_to_plot:
                if col_attr in df_comb.columns and not df_comb[col_attr].dropna().empty:
                    fig, ax = plt.subplots(figsize=(13, 8))
                    sns.scatterplot(data=df_comb, x="x", y="y", hue=col_attr, style="T2I_Model", markers=["o", "X"], s=100, alpha=0.75, palette=pal, ax=ax)
                    ax.set_title(f"1-vs-1 ({title_label} | {FEATURE_TYPE.upper()} UMAP): {modA.upper()} vs {modB.upper()}")
                    ax.set_xlabel("Gemeinsames UMAP X")
                    ax.set_ylabel("Gemeinsames UMAP Y")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    fig.tight_layout()
                    fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_1VS1_{modA}_vs_{modB}_{filename_suffix}.png", dpi=300)
                    plt.close(fig)

    # =============================================================
    # TEIL 3: GLOBAL-PLOTS (UMAP SYSTEMWEIT)
    # =============================================================
    print("\n========= TEIL 3: GLOBALER OVERVIEW (ALLE MODELLE GEMEINSAM IN UMAP) =========")
    all_combined_data = []
    for pkl_path in models_available.values():
        all_combined_data.extend(load_pkl_file(pkl_path))

    embeddings_global = np.array([item["Embedding"] for item in all_combined_data])
    df_global_vec = pd.DataFrame({
        "Image_Name": [item["Image_Name"] for item in all_combined_data],
        "T2I_Model": [item["T2I_Model"] for item in all_combined_data],
        "Prompt_Subject": [item["Prompt_Subject"] for item in all_combined_data]
    })
    df_global_vec["T2I_Model"] = df_global_vec["T2I_Model"].astype(str).str.strip().str.lower()
    df_global = pd.merge(df_global_vec, macro_meta, on=["Image_Name", "T2I_Model"], how="left")

    reducer_global = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    X_2d_global = reducer_global.fit_transform(embeddings_global)
    df_global["x"], df_global["y"] = X_2d_global[:, 0], X_2d_global[:, 1]

    # Global Plot 0: Modell-Galaxien
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.scatterplot(data=df_global, x="x", y="y", hue="T2I_Model", style="T2I_Model", s=75, alpha=0.7, palette="Set2", ax=ax)
    ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: Die Modell-Galaxien (UMAP Stil-Signatur)")
    ax.set_xlabel("Globales UMAP X")
    ax.set_ylabel("Globales UMAP Y")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_0_ALL_MODELS.png", dpi=300)
    plt.close(fig)

    # Global Plots für alle weiteren Attribute
    for col_attr, filename_suffix, pal, title_label in attributes_to_plot:
        if col_attr in df_global.columns and not df_global[col_attr].dropna().empty:
            fig, ax = plt.subplots(figsize=(13, 9))
            sns.scatterplot(data=df_global, x="x", y="y", hue=col_attr, style="T2I_Model", s=75, alpha=0.7, palette=pal, ax=ax)
            ax.set_title(f"Globaler {FEATURE_TYPE.upper()} Raum: {title_label} über alle Architekturen (UMAP)")
            ax.set_xlabel("Globales UMAP X")
            ax.set_ylabel("Globales UMAP Y")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"{FEATURE_TYPE.upper()}_GLOBAL_1_{filename_suffix}.png", dpi=300)
            plt.close(fig)

    # =============================================================
    # TEIL 4: QUANTITATIVE KOSINUS-ÄHNLICHKEIT (Roh-Vektoren bleiben unverändert)
    # =============================================================
    print("\n========= TEIL 4: MAXIMUM QUANTITATIVE KOSINUS-ÄHNLICHKEIT =========")
    cosine_report_rows = []
    unique_prompts = sorted(df_global["Prompt_Subject"].dropna().unique())

    forbidden_cols = ['Image_Name', 'T2I_Model', 'Prompt_Subject', 'Embedding', 'x', 'y', 'Visual_Cluster_ID', 'MonkScale_RGB']
    target_attributes = [c for c in df_global.columns if c not in forbidden_cols]

    for p_sub in unique_prompts:
        p_idx = df_global[df_global["Prompt_Subject"] == p_sub].index
        if len(p_idx) == 0: 
            continue
        avg_prompt_vector = embeddings_global[p_idx].mean(axis=0).reshape(1, -1)

        for attr in target_attributes:
            unique_values = df_global[attr].dropna().unique()
            for val in unique_values:
                val_idx = df_global[df_global[attr] == val].index
                if len(val_idx) > 0:
                    avg_val_vector = embeddings_global[val_idx].mean(axis=0).reshape(1, -1)
                    sim_value = cosine_similarity(avg_prompt_vector, avg_val_vector)[0][0]
                    
                    parts = attr.split('_')
                    metric_source = parts[0]
                    metric_type = '_'.join(parts[1:]) if len(parts) > 1 else "Unknown"
                    
                    cosine_report_rows.append({
                        "Prompt_Subject": p_sub,
                        "Metric_Source": metric_source,
                        "Metric_Type": metric_type,
                        "Category_Value": val,
                        "Cosine_Similarity": round(float(sim_value), 4)
                    })

    df_cosine_report = pd.DataFrame(cosine_report_rows)
    
    if not df_cosine_report.empty:
        df_cosine_report = df_cosine_report.sort_values(by=["Prompt_Subject", "Metric_Type", "Cosine_Similarity"], ascending=[True, True, False])
        print("\n" + "="*90)
        print("HIGH-DIMENSIONAL BIAS REPORT (Lückenloser Multi-VLM-Abgleich):")
        print("="*90)
        print(df_cosine_report.to_string(index=False))
        print("="*90)
        
        report_output_path = PLOTS_DIR / f"{FEATURE_TYPE.upper()}_RAW_EMBEDDING_COSINE_BIAS.csv"
        df_cosine_report.to_csv(report_output_path, index=False)
        print(f"Mathematischer Tidy-Format Report gespeichert: {report_output_path.name}")

    print(f"\nUMAP PIPELINE VOLLSTÄNDIG ABSCHLOSSEN! Ordner: outputs/plots/cluster/{FEATURE_TYPE}_plots_umap/")

if __name__ == "__main__":
    main()