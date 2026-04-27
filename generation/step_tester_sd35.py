"""
step_tester.py
==============
Beweis für die Masterarbeit: Semantische Stabilität über Iterationsschritte.
Generiert dasselbe Bild mit 5, 10, 15, 20, 25, 30, 35 und 40 Steps,
um den "Point of No Return" bei Diffusionsmodellen sichtbar zu machen.
"""

import os
import torch
import time
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/sd35")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter (Passe diese gerne an!)
PROMPT = "a photo of a doctor"
SEED = 101
# STEPS_TO_TEST = [1, 5, 10, 15, 20, 25, 28, 30, 35, 40]
STEPS_TO_TEST = [28]

def main():
    print("=" * 50)
    print("🔬 KI-ZEITLUPE STARTEN (Ablation Study)")
    print("=" * 50)
    
    # 3. Modell laden (SD 3.5)
    print("Lade Stable Diffusion 3.5...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ).to("cuda")
    
    print("\n🚀 Starte Generierungen...")
    
    # 4. Wir loopen durch unsere Step-Zahlen
    for steps in STEPS_TO_TEST:
        # GANZ WICHTIG: Generator für JEDEN Durchlauf neu mit demselben Seed initialisieren,
        # damit das Basis-Rauschen immer exakt zu 100% identisch ist!
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=steps,
            guidance_scale=3.5,   # 3.5 bei originalem SD3.5, 4.5 bei 4bit
            generator=generator,
        ).images[0]
        
        gen_time = time.time() - start_time
        
        # 5. Speichern mit dem Step im Dateinamen
        filename = OUTPUT_DIR / f"software_engineer_seed{SEED}_{steps:02d}steps.png"
        image.save(filename)
        
        print(f"✅ Bild mit {steps:02d} Steps gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
        
    print("\n🎉 Alle Testbilder generiert! Schau in den Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()