"""
model_correlation_matrix.py
===========================
GLOBALE ARCHITEKTUR-KORRELATION (DINOv3 & CLIP MULTI-RUN)
1. LOOP PROCESSING: Verarbeitet nacheinander vollautomatisch sowohl "dinov3" als auch "clip".
2. CENTROID-BERECHNUNG: Bildet für jedes Modell den globalen Mittelwertsvektor im 
   originalen hochdimensionalen Vektorraum (1024D bzw. 768D).
3. VISUALISIERUNG: Speichert getrennte 5x5 Heatmaps und CSV-Tabellen unter:
   outputs/plots/model_correlation_matrix/
"""

import pickle
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================
# PFADE STRUKTURIEREN
# =============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != 'analysis' else SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_OUT_DIR = OUTPUT_DIR / "plots" / "model_correlation_matrix"

sns.set_theme(style="white", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def load_pkl_file(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def run_correlation_for_feature_type(feature_type):
    print("\n" + "=" * 70)
    print(f"📊 STARTE GLOBALE MODELL-KORRELATIONS-ANALYSE ({feature_type.upper()}-RAUM)")
    print("=" * 70)

    # 1. Suche nach den spezifischen Vektor-Paketen (dinov3 oder clip)
    pkl_pattern = str(OUTPUT_DIR / f"{feature_type}_embeddings_*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        print(f"⚠️ Keine Vektor-Pakete für '{feature_type}_embeddings_*.pkl' in {OUTPUT_DIR} gefunden. Überspringe.")
        return

    # Extrahiere die Modellnamen dynamisch aus den Dateinamen passend zum feature_type
    models_available = {}
    for f in pkl_files:
        match = re.search(rf"{feature_type}_embeddings_(.+)\.pkl", Path(f).name)
        if match:
            models_available[match.group(1).upper()] = Path(f)

    sorted_model_names = sorted(list(models_available.keys()))
    print(f"✅ Gefundene Modelle für {feature_type.upper()}: {sorted_model_names}")

    # 2. Berechne den globalen Schwerpunkt (Centroid) für jedes Modell
    model_centroids = {}
    for model_name in sorted_model_names:
        pkl_path = models_available[model_name]
        data_list = load_pkl_file(pkl_path)
        
        # Extrahiere alle Embeddings dieses Modells
        embeddings = np.array([item["Embedding"] for item in data_list])
        
        # Mittelwertsvektor berechnen
        centroid = embeddings.mean(axis=0)
        model_centroids[model_name] = centroid
        print(f"  -> {model_name}: Centroid aus {len(embeddings)} Vektoren berechnet.")

    # 3. Baue die quadratische Korrelationsmatrix auf
    n_models = len(sorted_model_names)
    matrix_data = np.zeros((n_models, n_models))

    for i, mod_A in enumerate(sorted_model_names):
        for j, mod_B in enumerate(sorted_model_names):
            vec_A = model_centroids[mod_A].reshape(1, -1)
            vec_B = model_centroids[mod_B].reshape(1, -1)
            
            sim_val = cosine_similarity(vec_A, vec_B)[0][0]
            matrix_data[i, j] = sim_val

    df_matrix = pd.DataFrame(matrix_data, index=sorted_model_names, columns=sorted_model_names)

    # 4. Zeichne die professionelle Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        df_matrix, 
        annot=True,          
        fmt=".3f",           
        cmap="Blues",        
        vmin=0.5, vmax=1.0,  
        square=True,         
        linewidths=.5,       
        cbar_kws={"label": f"Kosinus-Ähnlichkeit im {feature_type.upper()}-Raum"},
        ax=ax
    )

    ax.set_title(f"Globale Vektorraum-Korrelation ({feature_type.upper()}): Verwandtschaft der Modelle", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    # Dynamische Pfade je nach Feature-Typ generieren
    plot_path = PLOTS_OUT_DIR / f"{feature_type.upper()}_GLOBAL_MODEL_CORRELATION_MATRIX.png"
    csv_path = PLOTS_OUT_DIR / f"{feature_type.upper()}_GLOBAL_MODEL_CORRELATION_MATRIX.csv"
    
    fig.savefig(plot_path, dpi=300)
    df_matrix.to_csv(csv_path)
    plt.close(fig)

    print(f"💾 Grafik gesichert: {plot_path.name}")
    print(f"💾 CSV-Tabelle gesichert: {csv_path.name}")


def main():
    # Zielordner einmalig erzeugen
    PLOTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Schleife läuft nacheinander durch beide Feature-Spaces
    for feature_space in ["dinov3", "clip"]:
        run_correlation_for_feature_type(feature_space)
        
    print("\n" + "=" * 70)
    print("🎉 MULTI-RUN BEENDET! Alle System-Heatmaps wurden erfolgreich berechnet.")
    print("=" * 70)

if __name__ == "__main__":
    main()