"""
crop_persons.py
================
Nutzt YOLOv8n/v11n um Personen in den generierten Bildern zu finden,
schneidet sie aus und speichert sie in einer sauberen Ordnerstruktur.
Features:
- Resume-Funktion: Überspringt Bilder, die bereits zugeschnitten wurden!
"""

import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
print(f"Projekt-Wurzel: {PROJECT_ROOT}")

INPUT_DIR = PROJECT_ROOT / "outputs" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"

def main():
    print("=" * 60)
    print("✂️ YOLO PERSONEN-ZUSCHNITT GESTARTET (mit Smart-Skip)")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Eingabeordner {INPUT_DIR} existiert nicht.")
        return

    # YOLO Modell laden
    print("Lade YOLO Modell...")
    model = YOLO("yolo11n.pt")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_cropped = 0
    total_skipped = 0

    # Gehe durch alle Modell-Ordner (flux, qwen, zimage etc.)
    for model_folder in INPUT_DIR.iterdir():
        if not model_folder.is_dir():
            continue
            
        print(f"\n📂 Verarbeite Modell-Ordner: {model_folder.name}")
        
        # Erstelle den passenden Unterordner im Ausgabe-Verzeichnis
        save_folder = OUTPUT_DIR / model_folder.name
        save_folder.mkdir(parents=True, exist_ok=True)
        
        # Sammle alle Bilder in diesem Ordner
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_files.extend(model_folder.rglob(ext))
            
        for img_path in image_files:
            # 1. BERECHNE DEN ZIEL-DATEINAMEN VORAB
            save_name = f"{img_path.stem}_crop{img_path.suffix}"
            save_path = save_folder / save_name
            
            # 2. PRÜFE, OB DAS BILD SCHON EXISTIERT (Smart-Skip)
            if save_path.exists():
                # Wenn ja, überspringen und nichts tun!
                # (Du kannst das print auskommentieren, wenn es dir im Terminal zu viel wird)
                # print(f"  ⏭️ Überspringe bereits zugeschnittenes Bild: {img_path.name}")
                total_skipped += 1
                continue
            
            # Wenn das Bild neu ist, zähle es als verarbeitet und starte YOLO
            total_processed += 1
            
            # YOLO auf das Bild anwenden (verbose=False)
            results = model(str(img_path), verbose=False) 
            
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"  ❌ Fehler beim Laden von {img_path.name}: {e}")
                continue

            person_cropped = False
            # Ergebnisse durchgehen
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    # COCO Datensatz: Klasse 0 ist "person"
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        padding = 10
                        width, height = img.size
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(width, x2 + padding)
                        y2 = min(height, y2 + padding)
                        
                        # Bild zuschneiden und speichern
                        cropped_img = img.crop((x1, y1, x2, y2))
                        cropped_img.save(save_path)
                        
                        total_cropped += 1
                        person_cropped = True
                        
                        print(f"  ✅ Neu zugeschnitten: {img_path.name}")
                        break
                
                if person_cropped:
                    break
            
            if not person_cropped:
                print(f"  ⚠️ Keine Person gefunden in: {img_path.name}")

    print("\n" + "=" * 60)
    print(f"🎉 FERTIG!")
    print(f"   - Neu geprüft: {total_processed}")
    print(f"   - Neu zugeschnitten: {total_cropped}")
    print(f"   - Übersprungen (bereits fertig): {total_skipped}")
    print(f"📁 Die Dateien liegen in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()