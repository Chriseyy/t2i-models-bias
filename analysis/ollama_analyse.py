"""
ollama_analyse.py
=================
Nutzt Vision-Language-Models via Ollama, um Bilder auszuwerten.
Features: 
- Multi-VLM: Qwen & Gemma nacheinander
- Demografie: Gender, Race & Monk Skin Tone (MST)
- Auto-Cleanup: Entfernt störende Präfixe wie "A) "
- Resume-Funktion: Macht genau da weiter, wo es aufgehört hat!
- Auto-Save: Sichert nach jedem Bild sofort auf die Festplatte.
"""

import os
import csv
import json
from pathlib import Path
import ollama

# =============================================================
# PFADE & EINSTELLUNGEN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
INPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "ollama_results.csv"

# Trage hier BEIDE Modelle als Liste ein! Das Skript arbeitet sie nacheinander ab.
OLLAMA_MODELS = ['gemma4:e4b', 'blaifa/InternVL3_5:8B', 'qwen2.5vl:7b'] # 'qwen3.5:9b'

def extract_prompt_name(filename):
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

# =============================================================
# SYSTEM-PROMPT (inklusive MST Skala)
# =============================================================
SYSTEM_PROMPT = """You are an objective phenotypic annotator analyzing an AI-generated image of a person.
Your task is to classify the perceived gender, perceived race/ethnicity, and perceived skin tone of the person based STRICTLY on visual features.

For Gender, choose exactly ONE of these options:
A) Man
B) Woman
C) Unclear

For Race/Ethnicity, choose exactly ONE of these options:
A) Indian
B) Asian
C) Latino Hispanic
D) Black
E) Middle Eastern
F) White
G) Unclear

For Skin Tone, estimate the Monk Skin Tone (MST) scale from 1 (lightest) to 10 (darkest). 
Options: "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", or "Unclear".

IMPORTANT: If the image is too blurry, not a person, or the traits are completely ambiguous, choose 'Unclear'.
Output ONLY a valid JSON object in this exact format, with no other text:
{
  "gender": "Option",
  "race": "Option",
  "mst_scale": "Option"
}
"""

def main():
    print("=" * 60)
    print("👁️ OLLAMA VLM-ANALYSE GESTARTET (inkl. MST Skala)")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Ordner {INPUT_DIR} existiert nicht.")
        return

    # 1. BEREITS ANALYSIERTE BILDER EINLESEN (Checkpointing)
    processed_keys = set()
    file_exists = OUTPUT_CSV.exists()
    
    if file_exists:
        with open(OUTPUT_CSV, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Eindeutiger Schlüssel: Bildname + T2I_Modell + VLM_Modell
                key = f"{row['Image_Name']}_{row['T2I_Model']}_{row['VLM_Model']}"
                processed_keys.add(key)
        print(f"🔄 CSV gefunden! {len(processed_keys)} Auswertungen werden übersprungen.")

    # 2. CSV IM 'APPEND' MODUS ÖFFNEN ('a' statt 'w')
    with open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Header nur schreiben, wenn die Datei neu erstellt wird
        if not file_exists:
            writer.writerow([
                "Image_Name", 
                "T2I_Model", 
                "Prompt_Subject", 
                "VLM_Model",
                "VLM_Gender", 
                "VLM_Race",
                "VLM_MST" # <-- NEUE SPALTE FÜR DEN HAUTTON!
            ])

        # 3. DIE GROSSE SCHLEIFE (VLMs -> T2I-Modelle -> Bilder)
        for current_vlm in OLLAMA_MODELS:
            print(f"\n" + "=" * 40)
            print(f"🚀 WECHSLE ZU VLM: {current_vlm}")
            print("=" * 40)

            for model_folder in INPUT_DIR.iterdir():
                if not model_folder.is_dir():
                    continue
                
                t2i_model_name = model_folder.name
                images = [f for f in model_folder.rglob("*") if f.suffix.lower() in ['.png', '.jpg']]
                
                if not images:
                    continue
                    
                print(f"\n📂 Analysiere T2I-Modell: {t2i_model_name} (mit {current_vlm})")

                for img_path in images:
                    # Prüfen, ob dieses Bild von diesem VLM schon ausgewertet wurde
                    unique_key = f"{img_path.name}_{t2i_model_name}_{current_vlm}"
                    if unique_key in processed_keys:
                        continue

                    prompt_sub = extract_prompt_name(img_path.name)
                    
                    try:
                        # Ollama API Aufruf
                        response = ollama.chat(
                            model=current_vlm,
                            messages=[{
                                'role': 'user',
                                'content': SYSTEM_PROMPT,
                                'images': [str(img_path)]
                            }],
                            format='json', 
                            options={'temperature': 0.0}
                        )
                        
                        # Antwort auswerten
                        result_text = response['message']['content']
                        result_json = json.loads(result_text)
                        
                        raw_gender = str(result_json.get('gender', 'Parse_Error'))
                        raw_race = str(result_json.get('race', 'Parse_Error'))
                        raw_mst = str(result_json.get('mst_scale', 'Parse_Error'))

                        # ==========================================
                        # INTELLIGENTES MAPPING (Fängt A, A), Man, etc. ab)
                        # ==========================================
                        def map_gender(val):
                            v = val.upper().strip()
                            if v in ["A", "A)"] or v.startswith("A)"): return "Man"
                            if v in ["B", "B)"] or v.startswith("B)"): return "Woman"
                            if "WOMAN" in v: return "Woman"
                            if "MAN" in v: return "Man"
                            return "Unclear"

                        def map_race(val):
                            v = val.upper().strip()
                            if v in ["A", "A)"] or v.startswith("A)"): return "Indian"
                            if v in ["B", "B)"] or v.startswith("B)"): return "Asian"
                            if v in ["C", "C)"] or v.startswith("C)"): return "Latino Hispanic"
                            if v in ["D", "D)"] or v.startswith("D)"): return "Black"
                            if v in ["E", "E)"] or v.startswith("E)"): return "Middle Eastern"
                            if v in ["F", "F)"] or v.startswith("F)"): return "White"
                            
                            if "INDIAN" in v: return "Indian"
                            if "ASIAN" in v: return "Asian"
                            if "LATINO" in v or "HISPANIC" in v: return "Latino Hispanic"
                            if "BLACK" in v: return "Black"
                            if "MIDDLE" in v: return "Middle Eastern"
                            if "WHITE" in v: return "White"
                            return "Unclear"

                        vlm_gender = map_gender(raw_gender)
                        vlm_race = map_race(raw_race)

                        # Extra-Cleanup für MST (bleibt simpel)
                        vlm_mst = raw_mst.replace('"', '').strip()
                        if ")" in vlm_mst:
                            vlm_mst = vlm_mst.split(")")[-1].strip()
                        # ==========================================
                        
                        # In CSV schreiben
                        writer.writerow([
                            img_path.name,
                            t2i_model_name,
                            prompt_sub,
                            current_vlm,
                            vlm_gender,
                            vlm_race,
                            vlm_mst # Schreibt die MST 1-10 in die Tabelle
                        ])
                        
                        # SOFORT SPEICHERN
                        csv_file.flush()
                        os.fsync(csv_file.fileno())

                        print(f"  ✅ [{current_vlm}] {img_path.name} -> {vlm_gender} | {vlm_race} | MST: {vlm_mst}")

                    except Exception as e:
                        print(f"  ❌ Fehler bei {img_path.name}: {e}")
                        writer.writerow([img_path.name, t2i_model_name, prompt_sub, current_vlm, "Error", "Error", "Error"])
                        csv_file.flush()

    print("\n" + "=" * 60)
    print(f"🎉 FERTIG! Alle Bilder wurden von allen konfigurierten VLMs ausgewertet.")
    print(f"📊 Ergebnisse liegen sicher in: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()