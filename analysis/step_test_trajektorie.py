"""
step_test_trajektorie.py
========================
TRAJEKTORIEN-KONVERGENZ-ANALYSE FÜR STEP-TESTS
1. AUTOMATISCHES SCANNING: Erfasst alle Modellordner unter outputs/step_test/.
2. HIGH-DIMENSIONAL EMBEDDING: Extrahiert die echten Bildmerkmale mittels DINOv2/CLIP.
3. KOSINUS-KONVERGENZ: Berechnet die Annäherung jedes Zwischenschritts an den 
   finalen Zustand (Max-Step-Bild) im originalen Vektorraum.
4. REPRODUZIERBARE PLOTS: Speichert für jedes Modell einen Linien-Plot unter outputs/plots/step_test/.
"""

import os
import re
import sys
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModel
from dotenv import load_dotenv
from huggingface_hub import login

# =============================================================
# ENVIRONMENT & HF LOGIN
# =============================================================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("✅ Erfolgreich beim Hugging Face Hub angemeldet.")
    except Exception as e:
        print(f"⚠️ Login-Fehler bei Hugging Face: {e}")
        pass
else:
    print("⚠️ WARNUNG: Kein 'HUGGINGFACE_HUB_TOKEN' in der .env-Datei gefunden!")
    print("Da DINOv3 zugriffsgeschützt sein kann, könnte der Download fehlschlagen.")


# =============================================================
# KONFIGURATION & EMBAEDDING-MODELL
# =============================================================
EMBEDDING_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# =============================================================
# PFADE STRUKTURIEREN
# =============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name != 'analysis' else SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STEP_TEST_DIR = OUTPUT_DIR / "step_test"
PLOTS_OUT_DIR = OUTPUT_DIR / "plots" / "step_test"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# =============================================================
# FEATURE EXTRACTOR KLASSE
# =============================================================
from transformers import AutoImageProcessor, AutoProcessor, AutoModel

class FeatureExtractor:
    def __init__(self, model_id, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Lade Feature-Extractor [{model_id}] auf {self.device}...")
        
        # ROBUSTES PROZESSOR-LADING: Reines Vision-Modell (DINO) vs. Multimodal (CLIP)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            print("  -> AutoImageProcessor erfolgreich geladen (Vision-Modell-Modus).")
        except Exception:
            self.processor = AutoProcessor.from_pretrained(model_id)
            print("  -> AutoProcessor erfolgreich geladen (Multimodal-Modell-Modus).")
            
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs)
        
        # ABSOLUT WASSERDICHTE VEKTOR-EXTRAKTION FÜR JEDE ARCHITEKTUR:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            # Standard für DINOv2 / DINOv3 (gibt direkt den CLS-Token-Embedding aus)
            emb = outputs.pooler_output
        elif hasattr(outputs, "vision_model_output") and hasattr(outputs.vision_model_output, "pooler_output"):
            # Falls ein volles CLIP-Modell via AutoModel geladen wurde
            emb = outputs.vision_model_output.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            # Fallback für rohe ViT-Backbones (wir nehmen das erste Token / CLS-Token)
            emb = outputs.last_hidden_state[:, 0, :]
        else:
            emb = outputs[0]
            
        return emb.cpu().numpy().flatten()

# =============================================================
# MAIN PROCESSING
# =============================================================
def get_steps_from_filename(filename):
    """
    FIXED REGEX: Extrahiert die numerische Step-Anzahl flexibel aus dem Filename.
    Funktioniert jetzt perfekt mit Suffixen wie '_01steps_cfg_normalization_True.png'
    """
    match = re.search(r"_(\d+)steps", filename)
    return int(match.group(1)) if match else None

def main():
    print("=" * 70)
    print("🔬 STARTE MATHEMATISCHE TRAJEKTORIEN-ANALYSE (STEP-TESTS)")
    print("=" * 70)

    if not STEP_TEST_DIR.exists():
        print(f"❌ Ordner nicht gefunden: {STEP_TEST_DIR}")
        print("Bitte stelle sicher, dass deine Step-Tester-Skripte bereits Bilder erzeugt haben.")
        return

    model_dirs = [d for d in STEP_TEST_DIR.iterdir() if d.is_dir()]
    if not model_dirs:
        print(f"❌ Keine Modell-Unterordner in {STEP_TEST_DIR} gefunden.")
        return

    print(f"✅ Gefundene Modell-Ordner: {[d.name for d in model_dirs]}")
    PLOTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor(EMBEDDING_MODEL_ID, device=device)

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\nProcessing Trajektorie für Modell: [{model_name.upper()}]")

        img_files = list(model_dir.glob("*.png"))
        if not img_files:
            print(f"  ⚠️ Keine PNG-Bilder in {model_dir.name} gefunden. Überspringe...")
            continue

        steps_data = []
        for img_path in img_files:
            steps = get_steps_from_filename(img_path.name)
            if steps is not None:
                steps_data.append((steps, img_path))

        steps_data = sorted(steps_data, key=lambda x: x[0])
        
        if len(steps_data) < 2:
            print("  ⚠️ Zu wenige Zwischenschritte für eine Konvergenz-Linie vorhanden.")
            continue

        print(f"  -> Gefundene Steps für die Kurve: {[x[0] for x in steps_data]}")

        final_step, final_img_path = steps_data[-1]
        print(f"  🎯 Definiere {final_step} Steps als finalen Konvergenz-Anker ({final_img_path.name})")
        final_vector = extractor.get_embedding(final_img_path).reshape(1, -1)

        plot_rows = []
        for steps, img_path in steps_data:
            current_vector = extractor.get_embedding(img_path).reshape(1, -1)
            sim = cosine_similarity(current_vector, final_vector)[0][0]
            
            plot_rows.append({
                "Steps": steps,
                "Cosine_Similarity": round(float(sim), 4)
            })

        df_trajectory = pd.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(data=df_trajectory, x="Steps", y="Cosine_Similarity", marker="o", linewidth=2.5, markersize=8, color="#2b5c8f", ax=ax)
        
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Perfekte Identität (1.0)")
        
        ax.set_title(f"Inferenz-Trajektorie: {model_name.upper()} | Prompt: doctor")
        ax.set_xlabel("Anzahl der Sampling Steps (Inferenzschritte)")
        ax.set_ylabel("Kosinus-Ähnlichkeit zum finalen Bild")
        ax.set_ylim(df_trajectory["Cosine_Similarity"].min() - 0.05, 1.02)
        
        ax.set_xticks(df_trajectory["Steps"].unique())
        
        fig.tight_layout()
        
        plot_filename = PLOTS_OUT_DIR / f"{model_name}_trajectory_convergence.png"
        fig.savefig(plot_filename, dpi=300)
        plt.close(fig)
        
        print(f"  💾 Konvergenz-Kurve erfolgreich gesichert: {plot_filename.parent.name}/{plot_filename.name}")
        
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 70)
    print(f"🎉 ALLE TRAJEKTORIEN-PLOTS ERSTELLT! Ordner: outputs/plots/step_test/")
    print("=" * 70)

if __name__ == "__main__":
    main()