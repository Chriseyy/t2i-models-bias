"""
extract_dinov3_embeddings.py
============================
Extraktion von Bild-Embeddings via DINOv3 (ViT-Large).
MÄCHTIGER ORDNER-SCANNER: Ignoriert die limitierte Human-CSV und verarbeitet
ALLE Bilder, die sich physisch in den Modellordnern unter 
outputs/cropped_persons/<T2I_Model>/ befinden für die echte Makro-Analyse.
Erstellt pro Modell eine eigene .pkl-Datei im Ausgabeordner.
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
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
# PFADE STRUKTURIEREN
# =============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR
if SCRIPT_DIR.name in ['analysis', 'src', 'scripts']:
    PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CROPPED_DIR = OUTPUT_DIR / "cropped_persons"
DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# =============================================================
# HILFSFUNKTION FÜR PROMPT-EXTRAKTION
# =============================================================
def extract_prompt_from_filename(filename):
    """Extrahiert das Prompt-Thema (z.B. prof_doctor) aus dem Dateinamen vor '_seed'"""
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

# =============================================================
# MAIN PROCESSING
# =============================================================
def main():
    print("-" * 60)
    print(f"📂 Projekt-Hauptverzeichnis: {PROJECT_ROOT.resolve()}")
    print(f"✂️ Cropped-Makro-Ordner:    {CROPPED_DIR.resolve()}")
    print("-" * 60)

    if not CROPPED_DIR.exists():
        print(f"❌ Ordner '{CROPPED_DIR.name}' existiert nicht! Bitte Pfade prüfen.")
        return

    # Finde alle Modell-Unterordner im cropped_persons Verzeichnis
    model_dirs = [d for d in CROPPED_DIR.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print(f"❌ Keine Modell-Unterordner (wie 'flux', 'sd35') in {CROPPED_DIR} gefunden!")
        return

    print(f"📊 Gefundene Modell-Ordner für den Makro-Scan: {[d.name for d in model_dirs]}")

    # DINOv3 laden (Einmalig für alle Ordner)
    print(f"🤖 Lade DINOv3 Modell: {DINOV3_MODEL_ID}...")
    try:
        processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=HF_TOKEN)
        model = AutoModel.from_pretrained(
            DINOV3_MODEL_ID,
            device_map="auto",
            attn_implementation="sdpa",  # Beschleunigung über Scaled Dot Product Attention
            token=HF_TOKEN
        )
        print(f"✅ DINOv3 erfolgreich geladen auf Gerät: {model.device}\n")
    except Exception as e:
        print(f"❌ Fehler beim Laden von DINOv3: {e}")
        print("Bitte überprüfe deine Berechtigungen auf HF und ob 'transformers' aktuell ist.")
        return

    # ITERATION ÜBER JEDEN MODELL-ORDNER
    for model_dir in model_dirs:
        t2i_model = model_dir.name
        
        print("\n" + "="*60)
        print(f"🎬 SCANNE ORDNER UND EXTRAHIERE VEKTOREN FÜR: {t2i_model.upper()}")
        print("-"*60)
        
        # Finde ALLE Bilder in diesem spezifischen Modellordner (png, jpg, jpeg)
        valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        img_paths = []
        for ext in valid_extensions:
            img_paths.extend(list(model_dir.glob(ext)))
            
        n_images = len(img_paths)
        print(f"🔍 Physisch gefunden: {n_images} Bilder im Ordner '{t2i_model}'. Processing startet...")

        if n_images == 0:
            print(f"⚠️ Ordner '{t2i_model}' ist leer. Überspringe.")
            continue

        extracted_data = []
        
        # Schleife über ALLE physisch gefundenen Bilder dieses Modells
        for img_path in tqdm(img_paths, desc=f"DINOv3 [{t2i_model}]"):
            img_name = img_path.name
            
            # Automatische Prompt-Erkennung aus dem Dateinamen! (z.B. prof_doctor)
            prompt_subject = extract_prompt_from_filename(img_name)

            try:
                # Bild laden und für DINOv3 vorbereiten
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(model.device)
                
                with torch.inference_mode():
                    outputs = model(**inputs)
                
                # Extrahiere den globalen 1024-dimensionalen Vektor (CLS-Token von DINOv3)
                embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
                
                extracted_data.append({
                    "Image_Name": img_name,
                    "T2I_Model": t2i_model,
                    "Prompt_Subject": prompt_subject,
                    "Embedding": embedding
                })
                
            except Exception as e:
                print(f"❌ Fehler bei Bild {img_name}: {e}")

        # Speichern der .pkl DATEI FÜR DIESES MODELL
        if extracted_data:
            # Sichert eindeutige Namen: dinov3_embeddings_{modell}.pkl
            embeddings_output = OUTPUT_DIR / f"dinov3_embeddings_{t2i_model}.pkl"
            
            print(f"💾 Speichere Makro-Vektoren von {len(extracted_data)} Bildern...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            with open(embeddings_output, "wb") as f:
                pickle.dump(extracted_data, f)
                
            print(f"🎉 Makro-Datei erfolgreich erstellt: {embeddings_output.name}")
        else:
            print(f"⚠️ Keine Embeddings generiert für Ordner {t2i_model}.")

    print("\n" + "="*60)
    print("🎉 ABSOLUTER MAKRO-SCAN BEENDET! Alle DINOv3 pkl-Dateien wurden erzeugt.")
    print("="*60)

if __name__ == "__main__":
    main()