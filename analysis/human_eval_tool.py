"""
human_eval_tool.py
==================
Professionelles GUI-Dashboard für die manuelle Evaluierung (Human-in-the-loop).
Erfasst Gender, Race und die Monk Skin Tone (MST) Skala.
Features: Live-Farb-Pipette (Hover) UND automatisches 3x3 Durchschnittsfarben-Raster!
"""

import os
import csv
import math
import random
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
INPUT_DIR = PROJECT_ROOT / "outputs" / "cropped_persons"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "human_evaluation.csv"

SAMPLE_SIZE = 50

# Die offiziellen HEX-Farben der Monk Skin Tone Skala (von Google)
MONK_HEX_COLORS = {
    "1": "#f6ede4", "2": "#f3e7db", "3": "#f7ead0", "4": "#eadaba", "5": "#d7bd96",
    "6": "#a07e56", "7": "#825c43", "8": "#604134", "9": "#3a312a", "10": "#292420"
}

def hex_to_rgb(hex_str):
    """Wandelt einen HEX-Code in ein RGB-Tuple um (für die Mathematik)"""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

class AnnotatorApp:
    def __init__(self, root, images):
        self.root = root
        self.images = images
        self.current_index = 0
        self.total_images = len(images)
        self.display_img = None 

        # Fenster konfigurieren
        self.root.title("Human Evaluation Dashboard - Bias Analyse")
        self.root.geometry("1250x950")
        self.root.configure(bg="#2d2d2d")

        # Layout-Frames erstellen
        self.left_frame = tk.Frame(root, bg="#2d2d2d", width=600)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        self.right_frame = tk.Frame(root, bg="#3d3d3d", width=600, relief="ridge", bd=2)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # UI Elemente Links (Bild & Counter)
        self.counter_label = tk.Label(self.left_frame, text="", fg="white", bg="#2d2d2d", font=("Arial", 14, "bold"))
        self.counter_label.pack(pady=5)

        # Der Bild-Container
        self.image_label = tk.Label(self.left_frame, bg="#2d2d2d")
        self.image_label.pack(pady=5)
        
        # Mausevents für das Bild binden (Die Live-Pipette)
        self.image_label.bind("<Motion>", self.on_image_hover)
        self.image_label.bind("<Leave>", self.on_image_leave)

        # --- NEU: Das 3x3 Farb-Raster kommt direkt unter das Bild ---
        self.palette_frame = tk.Frame(self.left_frame, bg="#2d2d2d")
        self.palette_frame.pack(pady=10)

        # UI Elemente Links Unten (Die Hover-Pipetten Anzeige)
        self.hover_frame = tk.Frame(self.left_frame, bg="#2d2d2d")
        self.hover_frame.pack(pady=5, fill="x")
        
        tk.Label(self.hover_frame, text="🔍 Live Pixel-Analyse:", fg="#4DA8DA", bg="#2d2d2d", font=("Arial", 12, "bold")).pack(anchor="center")
        
        self.hover_info_frame = tk.Frame(self.hover_frame, bg="#2d2d2d")
        self.hover_info_frame.pack(pady=5)
        
        self.hover_color_box = tk.Frame(self.hover_info_frame, width=40, height=40, bg="#2d2d2d", relief="solid", bd=1)
        self.hover_color_box.pack(side="left", padx=10)
        self.hover_color_box.pack_propagate(False)
        
        self.hover_text_label = tk.Label(self.hover_info_frame, text="Fahre mit der Maus über das Bild...", 
                                         fg="#a0a0a0", bg="#2d2d2d", font=("Arial", 11))
        self.hover_text_label.pack(side="left")

        # UI Elemente Rechts (Eingabe-Formular)
        tk.Label(self.right_frame, text="Wissenschaftliche Annotation", fg="white", bg="#3d3d3d", font=("Arial", 18, "bold")).pack(pady=10)

        self.var_gender = tk.StringVar(value="")
        self.var_race = tk.StringVar(value="")
        self.var_mst = tk.StringVar(value="")

        # 1. GENDER Sektion
        self.create_section("1. Perceived Gender", self.var_gender, ["Man", "Woman", "Unclear"], wrap=3)
        
        # 2. RACE Sektion
        self.create_section("2. Perceived Race/Ethnicity", self.var_race, 
                            ["Indian", "Asian", "Latino Hispanic", "Black", "Middle Eastern", "White", "Unclear"], wrap=3)
        
        # 3. MST Sektion
        self.create_mst_section("3. Perceived Monk Skin Tone (MST)")

        # Speichern Button
        self.save_btn = tk.Button(self.right_frame, text="Speichern & Nächstes Bild [ENTER]", 
                                  command=self.save_and_next, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white", height=2)
        self.save_btn.pack(pady=25, fill="x", padx=20)

        self.root.bind('<Return>', lambda event: self.save_and_next())
        self.root.bind('<Escape>', lambda event: self.root.quit())

        # CSV vorbereiten
        self.file_exists = OUTPUT_CSV.exists()
        self.csv_file = open(OUTPUT_CSV, mode='a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.csv_file)
        
        if not self.file_exists:
            self.writer.writerow(["Image_Name", "T2I_Model", "Human_Gender", "Human_Race", "Human_MST"])

        self.show_image()

    def get_closest_mst(self, r, g, b):
        min_dist = float('inf')
        best_mst = None
        for mst_val, hex_code in MONK_HEX_COLORS.items():
            mr, mg, mb = hex_to_rgb(hex_code)
            dist = math.sqrt((r - mr)**2 + (g - mg)**2 + (b - mb)**2)
            if dist < min_dist:
                min_dist = dist
                best_mst = mst_val
        return best_mst

    def on_image_hover(self, event):
        if self.display_img is None:
            return
            
        x, y = event.x, event.y
        w, h = self.display_img.size
        
        if 0 <= x < w and 0 <= y < h:
            pixel = self.display_img.getpixel((x, y))
            if isinstance(pixel, int): 
                pixel = (pixel, pixel, pixel)
            r, g, b = pixel[:3]
            
            hex_color = '#%02x%02x%02x' % (r, g, b)
            closest_mst = self.get_closest_mst(r, g, b)
            
            self.hover_color_box.config(bg=hex_color)
            self.hover_text_label.config(
                text=f"Maus-Position HEX: {hex_color.upper()}\nMathematisch nächster Treffer: MST {closest_mst}",
                fg="white"
            )

    def on_image_leave(self, event):
        self.hover_color_box.config(bg="#2d2d2d")
        self.hover_text_label.config(text="Fahre mit der Maus über das Bild...", fg="#a0a0a0")

    def create_section(self, title, variable, options, wrap=0):
        frame = tk.Frame(self.right_frame, bg="#3d3d3d")
        frame.pack(fill="x", padx=20, pady=10)
        tk.Label(frame, text=title, fg="#4DA8DA", bg="#3d3d3d", font=("Arial", 12, "bold")).pack(anchor="w")
        
        grid_frame = tk.Frame(frame, bg="#3d3d3d")
        grid_frame.pack(anchor="w", pady=5)
        
        for i, option in enumerate(options):
            row = i // wrap if wrap > 0 else 0
            col = i % wrap if wrap > 0 else i
            rb = tk.Radiobutton(grid_frame, text=option, variable=variable, value=option, 
                                bg="#3d3d3d", fg="white", selectcolor="#555555", font=("Arial", 11))
            rb.grid(row=row, column=col, sticky="w", padx=10, pady=2)

    def create_mst_section(self, title):
        frame = tk.Frame(self.right_frame, bg="#3d3d3d")
        frame.pack(fill="x", padx=20, pady=10)
        tk.Label(frame, text=title, fg="#4DA8DA", bg="#3d3d3d", font=("Arial", 12, "bold")).pack(anchor="w")
        
        grid_frame = tk.Frame(frame, bg="#3d3d3d")
        grid_frame.pack(anchor="w", pady=5)
        
        for i in range(1, 11):
            val_str = str(i)
            col = (i - 1) % 5
            row = (i - 1) // 5 * 3
            
            rb = tk.Radiobutton(grid_frame, text=f"MST {val_str}", variable=self.var_mst, value=val_str, 
                                bg="#3d3d3d", fg="white", selectcolor="#555555", font=("Arial", 10, "bold"))
            rb.grid(row=row, column=col, sticky="w", padx=5, pady=(5,0))
            
            color_box = tk.Frame(grid_frame, width=50, height=20, bg=MONK_HEX_COLORS[val_str], relief="sunken", bd=1)
            color_box.grid(row=row+1, column=col, padx=5, pady=(2, 0))
            color_box.pack_propagate(False)
            
            tk.Label(grid_frame, text=MONK_HEX_COLORS[val_str].upper(), bg="#3d3d3d", fg="#a0a0a0", 
                     font=("Arial", 8)).grid(row=row+2, column=col, padx=5, pady=(0, 5))

        rb_unclear = tk.Radiobutton(grid_frame, text="Unclear", variable=self.var_mst, value="Unclear", 
                                    bg="#3d3d3d", fg="white", selectcolor="#555555", font=("Arial", 10))
        rb_unclear.grid(row=2, column=4, sticky="w", padx=5, pady=2)

    def extract_image_palette(self, img):
        """Teilt das Bild in ein 3x3 Raster und extrahiert die Durchschnittsfarbe (HEX)."""
        for widget in self.palette_frame.winfo_children():
            widget.destroy()
            
        tk.Label(self.palette_frame, text="Durchschnittsfarben (3x3 Raster):", 
                 fg="#a0a0a0", bg="#2d2d2d", font=("Arial", 10, "italic")).grid(row=0, column=0, columnspan=3, pady=(0,5))
        
        w, h = img.size
        bw, bh = w // 3, h // 3
        
        for row in range(3):
            for col in range(3):
                box = (col*bw, row*bh, (col+1)*bw, (row+1)*bh)
                region = img.crop(box)
                avg_color = region.resize((1, 1)).getpixel((0, 0))
                
                if isinstance(avg_color, int): 
                    avg_color = (avg_color, avg_color, avg_color)
                    
                hex_color = '#%02x%02x%02x' % avg_color[:3]
                
                c_box = tk.Frame(self.palette_frame, width=30, height=30, bg=hex_color, relief="solid", bd=1)
                c_box.grid(row=row+1, column=col, padx=2, pady=2)
                c_box.pack_propagate(False)

    def show_image(self):
        if self.current_index >= self.total_images:
            self.counter_label.config(text="🎉 Fertig! Stichprobe erfolgreich evaluiert.")
            self.image_label.config(image='')
            self.on_image_leave(None)
            messagebox.showinfo("Erfolg", "Super! Du hast deine 50 Stichproben-Bilder bewertet. Die Daten liegen sicher in der CSV.")
            self.root.after(500, self.root.quit)
            return

        img_path = self.images[self.current_index]
        self.counter_label.config(text=f"Bild {self.current_index + 1} von {self.total_images}\nModell-Quelle: {img_path.parent.name}")

        img = Image.open(img_path)
        
        # --- NEU: 3x3 Raster extrahieren, BEVOR skaliert wird ---
        self.extract_image_palette(img)
        
        img.thumbnail((400, 450)) 
        self.display_img = img.copy() 
        
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        
        self.var_gender.set("")
        self.var_race.set("")
        self.var_mst.set("")
        
        self.on_image_leave(None)

    def save_and_next(self):
        if self.current_index >= self.total_images:
            return

        g = self.var_gender.get()
        r = self.var_race.get()
        m = self.var_mst.get()

        if not g or not r or not m:
            messagebox.showwarning("Fehlende Auswahl", "Bitte wähle für alle 3 Kategorien einen Wert aus!")
            return

        img_path = self.images[self.current_index]
        t2i_model = img_path.parent.name

        self.writer.writerow([img_path.name, t2i_model, g, r, m])
        self.csv_file.flush()

        self.current_index += 1
        self.show_image()

    def __del__(self):
        try:
            self.csv_file.close()
        except:
            pass

def main():
    print("=" * 60)
    print("🧠 ADVANCED HUMAN EVALUATION TOOL GESTARTET")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"❌ Fehler: Ordner {INPUT_DIR} existiert nicht.")
        return

    all_images = []
    for model_folder in INPUT_DIR.iterdir():
        if model_folder.is_dir():
            all_images.extend(list(model_folder.rglob("*.png")) + list(model_folder.rglob("*.jpg")))

    if not all_images:
        print("❌ Keine zugeschnittenen Bilder gefunden!")
        return

    processed_images = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) 
            for row in reader:
                if row: processed_images.add(row[0])

    pending_images = [img for img in all_images if img.name not in processed_images]

    if len(pending_images) > SAMPLE_SIZE:
        sample_images = random.sample(pending_images, SAMPLE_SIZE)
    else:
        sample_images = pending_images

    if not sample_images:
        print("✅ Du hast bereits alle Bilder (oder genug für deine Stichprobe) bewertet!")
        return

    print(f"Lade {len(sample_images)} verbleibende Stichproben-Bilder...")

    root = tk.Tk()
    app = AnnotatorApp(root, sample_images)
    root.mainloop()

if __name__ == "__main__":
    main()