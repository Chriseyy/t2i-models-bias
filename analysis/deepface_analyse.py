"""
deepface_analyse.py (PRO-VERSION)
=================================
Nutzt DeepFace für eine vollautomatische Bias-Analyse:
1. Demografie: Geschlecht, Rasse/Ethnie, Alter via .analyze()
2. Fotorealismus: Anti-Spoofing Check via .extract_faces()

Speichert alles fehlerfrei in einer gemeinsamen CSV-Datei.
"""

import os
import csv
from pathlib import Path
from deepface import DeepFace

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
INPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "deepface_results.csv"

def extract_prompt_name(filename):
    """Extrahiert den Beruf/Prompt aus dem Dateinamen"""
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

def main():
    print("=" * 60)
    print("🤖 DEEPFACE MASTER-ANALYSE GESTARTET (inkl. Anti-Spoofing)")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Ordner {INPUT_DIR} existiert nicht. Bitte zuerst YOLO laufen lassen!")
        return

    # Bereite die CSV-Datei vor
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # HIER IST DIE NEUE SPALTE: "DeepFace_IsReal" am Ende hinzugefügt!
        writer.writerow([
            "Image_Name", 
            "T2I_Model", 
            "Prompt_Subject", 
            "DeepFace_Gender", 
            "DeepFace_Race", 
            "DeepFace_Age",
            "DeepFace_IsReal"
        ])

        total_processed = 0
        total_errors = 0

        # Gehe durch alle Modell-Ordner (flux, qwen, sd35, etc.)
        for model_folder in INPUT_DIR.iterdir():
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            print(f"\n📂 Analysiere Modell: {model_name}")

            # Alle Bilder im Modell-Ordner sammeln
            images = [f for f in model_folder.rglob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            for img_path in images:
                prompt_sub = extract_prompt_name(img_path.name)
                
                try:
                    # ---------------------------------------------------------
                    # SCHRITT 1: Demografie-Analyse (Gender, Race, Age)
                    # ---------------------------------------------------------
                    result = DeepFace.analyze(
                        img_path=str(img_path),
                        actions=['gender', 'race', 'age'],
                        enforce_detection=False,
                        silent=True
                    )
                    res = result[0] if isinstance(result, list) else result
                    
                    df_gender = res.get('dominant_gender', 'Error')
                    df_race = res.get('dominant_race', 'Error')
                    df_age = res.get('age', 'Error')

                    # ---------------------------------------------------------
                    # SCHRITT 2: Anti-Spoofing Check (Is it Real?)
                    # HIERMALS ALS EIGENES ISOLIERTES SICHERHEITSNETZ!
                    # ---------------------------------------------------------
                    df_is_real = "Unknown"
                    try:
                        face_objs = DeepFace.extract_faces(
                            img_path=str(img_path),
                            anti_spoofing=True,
                            enforce_detection=False
                        )
                        if face_objs and len(face_objs) > 0:
                            # Holt den True/False Wert aus dem ersten Gesicht
                            df_is_real = face_objs[0].get("is_real", "Unknown")
                    except Exception as spoof_error:
                        # Falls FasNet fehlschlägt, fangen wir es hier lautlos ab
                        df_is_real = "Error"

                    # ---------------------------------------------------------
                    # SCHRITT 3: Daten in die CSV schreiben
                    # ---------------------------------------------------------
                    writer.writerow([
                        img_path.name,
                        model_name,
                        prompt_sub,
                        df_gender,
                        df_race,
                        df_age,
                        df_is_real  # Schreibt True, False, Unknown oder Error
                    ])
                    
                    # Schickes Terminal-Feedback mit Status
                    status_symbol = "🟢" if df_is_real == True else ("🔴" if df_is_real == False else "🟡")
                    print(f"  {status_symbol} {img_path.name} -> {df_gender} | {df_race} | Real: {df_is_real}")
                    total_processed += 1

                except Exception as e:
                    print(f"  ❌ Kritischer Fehler bei {img_path.name}: Überspringe Bild.")
                    writer.writerow([img_path.name, model_name, prompt_sub, "Error", "Error", "Error", "Error"])
                    total_errors += 1

    print("\n" + "=" * 60)
    print(f"🎉 FERTIG! {total_processed} Bilder im Master-Verfahren analysiert. ({total_errors} kritische Fehler)")
    print(f"📊 Deine finale Excel/CSV-Tabelle liegt hier: {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()