"""
skin_metrics_analyse.py
=======================
Führt zwei rein mathematische Hautfarben-Analysen parallel aus:
1. K-Means-Ansatz (RGB Euclidean Distance zur Monk-Skala)
2. ITA-Ansatz (Individual Typology Angle im Lab-Farbraum mapped zu MST)

Speichert beide Ergebnisse übersichtlich nebeneinander in einer CSV.
"""

import os
import csv
import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
from sklearn.cluster import KMeans

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
INPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "skin_metrics_results.csv"

# Die offiziellen RGB-Werte der 10 Monk Skin Tones (von Google)
MONK_SCALE_RGB = {
    1: np.array([246, 237, 228]),
    2: np.array([243, 231, 219]),
    3: np.array([247, 234, 208]),
    4: np.array([234, 218, 186]),
    5: np.array([215, 189, 150]),
    6: np.array([160, 126, 86]),
    7: np.array([130, 92, 67]),
    8: np.array([96, 65, 52]),
    9: np.array([58, 49, 42]),
    10: np.array([41, 36, 32])
}

def extract_prompt_name(filename):
    """Extrahiert den Beruf/Prompt aus dem Dateinamen"""
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

# =============================================================
# METRIC 1: K-MEANS + RGB EUCLIDEAN DISTANCE
# =============================================================
def get_dominant_color_kmeans(image_rgb, k=3):
    """Sucht die dominanteste Farbe im Gesicht mittels Pixel-Clustering."""
    pixels = image_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    counts = np.bincount(kmeans.labels_)
    dominant_cluster_index = np.argmax(counts)
    return kmeans.cluster_centers_[dominant_cluster_index]

def find_closest_mst_rgb(dominant_rgb):
    """Berechnet den mathematischen Abstand im RGB-Raum zur MST Skala."""
    min_dist = float('inf')
    best_mst_num = "Error"
    
    for mst_num, mst_rgb in MONK_SCALE_RGB.items():
        dist = np.linalg.norm(dominant_rgb - mst_rgb)
        if dist < min_dist:
            min_dist = dist
            best_mst_num = mst_num
            
    return best_mst_num

# =============================================================
# METRIC 2: ITA (INDIVIDUAL TYPOLOGY ANGLE) IM LAB-RAUM
# =============================================================
def compute_ita_metrics(image_path):
    """Berechnet den ITA-Winkel und mappt ihn auf die MST-Skala."""
    img = cv2.imread(str(image_path))
    if img is None:
        return "Error", "Error"
        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    
    # Grobe Maske für Hautpixel im OpenCV-Lab-Raum
    skin_mask = (L > 40) & (a > 120) & (a < 175) & (b > 100) & (b < 165)
    
    if skin_mask.sum() < 50:
        return "No_Skin_Detected", "Error"
    
    # === HIER IST DER MATHEMATISCHE FIX ===
    # OpenCV skaliert L von 0-255. Wir müssen es für die Formel auf 0-100 umrechnen!
    L_mean_real = L[skin_mask].mean() * (100.0 / 255.0)
    
    # OpenCV b-Kanal Shifting korrigieren
    b_mean_real = b[skin_mask].mean() - 128.0  
    
    # ITA Formel anwenden
    if b_mean_real == 0: 
        b_mean_real = 0.001  
        
    ita = np.degrees(np.arctan((L_mean_real - 50.0) / b_mean_real))
    ita_val = round(ita, 2)
    # ========================================
    
    # Mapping von ITA-Winkel auf MST-Stufen
    if ita > 55:     ita_mst = 1
    elif ita > 41:   ita_mst = 2
    elif ita > 28:   ita_mst = 3
    elif ita > 19:   ita_mst = 4
    elif ita > 10:   ita_mst = 5
    elif ita > 0:    ita_mst = 6
    elif ita > -15:  ita_mst = 7
    elif ita > -30:  ita_mst = 8
    else:            ita_mst = 9
    
    return ita_val, ita_mst

# =============================================================
# MAIN PROCESSING LOOP
# =============================================================
def main():
    print("=" * 60)
    print("🎨 STARTE DUALE SKIN-METRICS-ANALYSE (RGB vs. ITA)")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Ordner {INPUT_DIR} existiert nicht.")
        return

    # Bereite die CSV-Datei vor
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # Tabellenkopf (Header) exakt wie von dir gewünscht
        writer.writerow([
            "Image_Name", 
            "T2I_Model", 
            "Prompt_Subject", 
            "MonkScale_RGB", 
            "ITA_Value", 
            "ITA_Scale_MST"
        ])

        total_processed = 0

        # Iteriere durch die Modellordner (flux, sd35, zimage, etc.)
        for model_folder in INPUT_DIR.iterdir():
            if not model_folder.is_dir():
                continue
            
            model_name = model_folder.name
            print(f"\n📂 Analysiere T2I-Modell: {model_name}")

            images = [f for f in model_folder.rglob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            for img_path in images:
                prompt_sub = extract_prompt_name(img_path.name)
                
                # Standard-Fallbacks falls etwas fehlschlägt
                mst_rgb_res = "Error"
                ita_val_res = "Error"
                ita_mst_res = "Error"
                
                # --- ANSATZ 1: K-Means (DeepFace Gesichtsausschnitt) ---
                try:
                    face_objs = DeepFace.extract_faces(img_path=str(img_path), enforce_detection=False)
                    if face_objs and len(face_objs) > 0:
                        face_img = face_objs[0]['face']
                        face_rgb = (face_img * 255).astype(np.uint8)
                        
                        dominant_rgb = get_dominant_color_kmeans(face_rgb)
                        mst_rgb_res = find_closest_mst_rgb(dominant_rgb)
                except Exception as e:
                    mst_rgb_res = "Detection_Error"

                # --- ANSATZ 2: ITA (Gesamtes Zuschnitt-Bild Lab-Maskierung) ---
                try:
                    ita_val_res, ita_mst_res = compute_ita_metrics(img_path)
                except Exception as e:
                    ita_val_res, ita_mst_res = "Calc_Error", "Error"

                # --- DATEN IN DIE CSV SCHREIBEN ---
                writer.writerow([
                    img_path.name,
                    model_name,
                    prompt_sub,
                    mst_rgb_res,
                    ita_val_res,
                    ita_mst_res
                ])
                
                print(f"  🔹 {img_path.name} -> RGB-MST: {mst_rgb_res} | ITA: {ita_val_res} (MST: {ita_mst_res})")
                total_processed += 1

    print("\n" + "=" * 60)
    print(f"🎉 FERTIG! {total_processed} Bilder mathematisch vermessen.")
    print(f"📊 Deine duale Auswertungstabelle liegt hier: {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()