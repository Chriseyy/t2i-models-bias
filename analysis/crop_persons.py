"""
crop_persons.py
================
Nutzt YOLOv8n um Personen in den generierten Bildern zu finden,
schneidet sie aus und speichert sie in einer sauberen Ordnerstruktur.
"""

import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# =============================================================
# PFADE DEFINIEREN
# =============================================================
# Passe PROJECT_ROOT an, je nachdem wo du das Skript speicherst. 
# Wenn es im Hauptordner liegt: Path(__file__).parent
# Wenn es im 'analysis' Ordner liegt: Path(__file__).parent.parent
PROJECT_ROOT = Path(__file__).parent.parent 
print(f"Projekt-Wurzel: {PROJECT_ROOT}")

INPUT_DIR = PROJECT_ROOT / "outputs" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"

def main():
    print("=" * 60)
    print("✂️ YOLO PERSONEN-ZUSCHNITT GESTARTET")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Eingabeordner {INPUT_DIR} existiert nicht.")
        return

    # YOLO Modell laden (Nano-Version: extrem schnell, braucht kaum RAM)
    print("Lade YOLOv11n Modell...")
    model = YOLO("yolo11n.pt")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_cropped = 0

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
                # YOLO sortiert standardmäßig nach Confidence Score (xyxy[0] ist die sicherste Box)
                boxes = r.boxes
                
                for box in boxes:
                    # COCO Datensatz: Klasse 0 ist "person"
                    if int(box.cls[0]) == 0:
                        # Bounding-Box Koordinaten extrahieren (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Einen kleinen Rand (Padding) hinzufügen
                        padding = 10
                        width, height = img.size
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(width, x2 + padding)
                        y2 = min(height, y2 + padding)
                        
                        # Bild zuschneiden
                        cropped_img = img.crop((x1, y1, x2, y2))
                        
                        # Speichern (z.B. doctor_seed101_crop.png)
                        save_name = f"{img_path.stem}_crop{img_path.suffix}"
                        save_path = save_folder / save_name
                        cropped_img.save(save_path)
                        
                        total_cropped += 1
                        person_cropped = True
                        
                        # !!! WICHTIG: Nach dem ersten Fund die Schleife abbrechen !!!
                        print(f"  ✅ Hauptsubjekt zugeschnitten: {img_path.name}")
                        break
                
                # Auch die äußere Schleife abbrechen
                if person_cropped:
                    break
            
            if not person_cropped:
                print(f"  ⚠️ Keine Person gefunden in: {img_path.name}")

    print("\n" + "=" * 60)
    print(f"🎉 FERTIG! {total_processed} Bilder geprüft, {total_cropped} Personen zugeschnitten.")
    print(f"📁 Die Dateien liegen in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()