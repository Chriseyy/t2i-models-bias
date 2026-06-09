"""
create_cfg_plots.py
===================
Automatisiertes Auswertungsskript für die CFG-Ablationsreihen.
Erstellt für jeden Prompt eine saubere Vergleichsmatrix (Modelle x CFG-Werte)
und markiert den visuellen Freeze-Punkt (Mode Collapse) automatisch rot.
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch import seed

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR    = PROJECT_ROOT / "outputs" / "images_cfg_test"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "plots" / "cfg"

# Reihenfolge der Zeilen in der Matrix (West nach Ost sortiert)
MODEL_ORDER  = ["sd35", "flux", "flux_klein", "qwen", "zimage"]

# Regex zum krisensicheren Parsen der Dateinamen (z.B. prof_doctor_seed101_cfg7.5.png)
FILENAME_REGEX = re.compile(
    r"^(?P<prompt_id>.+)_seed(?P<seed>\d+)_cfg(?P<cfg>[\d\.]+)\.(png|jpg|jpeg)$", 
    re.IGNORECASE
)

def discover_images():
    """Scannt das Verzeichnis und gruppiert Bilder nach (prompt_id, seed)"""
    data_groups = {}
    
    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Verzeichnis {INPUT_DIR} existiert nicht!")
        return data_groups

    for model_folder in INPUT_DIR.iterdir():
        if model_folder.is_dir():
            model_name = model_folder.name
            for img_path in model_folder.glob("*.*"):
                match = FILENAME_REGEX.match(img_path.name)
                if match:
                    prompt_id = match.group("prompt_id")
                    seed = match.group("seed")
                    cfg_val = float(match.group("cfg"))
                    
                    group_key = (prompt_id, seed)
                    if group_key not in data_groups:
                        data_groups[group_key] = {}
                    if model_name not in data_groups[group_key]:
                        data_groups[group_key][model_name] = {}
                        
                    data_groups[group_key][model_name][cfg_val] = img_path
                    
    return data_groups

def build_matrix_plots():
    """Baut die Vergleichsmatrizen und speichert sie ab"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = discover_images()
    
    if not groups:
        print("⚠️ Keine passenden Bilder für die CFG-Testreihe gefunden.")
        return

    for (prompt_id, seed), models_dict in groups.items():
        print(f"\n📊 Verarbeite Gruppe: {prompt_id} (Seed: {seed})")
        
        # Alle vorkommenden CFG-Werte für diese Gruppe sammeln und sortieren
        all_cfgs = sorted(list({cfg for m in models_dict.values() for cfg in m.keys()}))
        num_cols = len(all_cfgs)
        num_rows = len(MODEL_ORDER)
        
        # Dynamische Grid-Größe festlegen
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.5))
        fig.suptitle(f"CFG Ablation Study: '{prompt_id}' (Seed: {seed})", fontsize=18, fontweight='bold', y=0.96)
        
        for r_idx, model_name in enumerate(MODEL_ORDER):
            for c_idx, cfg_val in enumerate(all_cfgs):
                # Achsen-Objekt holen (falls 1D Array, sicherstellen dass es 2D bleibt)
                ax = axes[r_idx, c_idx] if num_rows > 1 else axes[c_idx]
                
                # Pfad zum Bild ermitteln
                img_path = models_dict.get(model_name, {}).get(cfg_val, None)
                
                if img_path and img_path.exists():
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center', color='red')
                else:
                    # Platzhalter falls ein Modell oder ein CFG-Wert fehlt
                    ax.patch.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, f"N/A\n{model_name}\nCFG {cfg_val}", ha='center', va='center', color='gray', fontsize=10)
                
                # Achsen-Striche ausblenden für cleanen Paper-Look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Spaltenüberschriften (CFG-Werte) nur in der ersten Zeile anzeigen
                if r_idx == 0:
                    ax.set_title(f"CFG {cfg_val}", fontsize=14, fontweight='bold', pad=10)
                
                # Zeilenüberschriften (Modellnamen) nur in der ersten Spalte anzeigen
                if c_idx == 0:
                    display_name = model_name.upper().replace("_", " ")
                    ax.set_ylabel(display_name, fontsize=14, fontweight='bold', labelpad=15)
        
        # Speichern des finalen Plots
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path = OUTPUT_DIR / f"{prompt_id}_seed{seed}_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Matrix erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    print("🚀 Starte CFG Matrix Generator...")
    build_matrix_plots()
    print("\n🎉 Alle Plots erfolgreich generiert!")