"""
step_tester_zimage.py
=====================
Ablation Study für Z-Image (Base, unkomprimiert).
Zeigt, wie das östliche Modell in Einzelschritten das Bild formt.
"""

import os
import torch
import time
from pathlib import Path
from diffusers import ZImagePipeline
from dotenv import load_dotenv

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/zimage")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter
PROMPT = "a photo of a doctor"
PROMPT = "a photo of a beautiful person"  # Für den Bias-Test wollen wir einen neutralen Prompt, damit die Vorurteile des Modells sichtbar werden.
NEGATIVE_PROMPT = ""  # Wie in deiner Config besprochen, leer lassen für sauberen Bias-Test
SEED = 103

# Z-Image ist ein 50-Step Base-Modell. Diese Spanne zeigt perfekt den Aufbau.
STEPS_TO_TEST = [1, 3, 5, 10, 20, 30, 40, 45, 50, 55, 60]

def main():
    print("=" * 60)
    print("🔬 KI-ZEITLUPE: Z-IMAGE (Ost-Modell / 6B Base)")
    print("=" * 60)
    
    # 3. Modell laden
    print("Lade Z-Image...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    
    # Da es "nur" 6 Milliarden Parameter hat (im Vergleich zu den 32B von FLUX),
    # passt es locker in deine 32GB VRAM. Wir schieben es komplett auf die GPU.
    pipe.to("cuda")
    
    # Anti-Freeze für Windows
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    print("\n🚀 Starte Generierungen...")
    
    # 4. Loop durch die Steps
    for steps in STEPS_TO_TEST:
        # Generator bei JEDEM Step neu mit Seed 103 starten für perfekte Vergleichbarkeit!
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=4.0,           # Der offizielle Z-Image Base Wert
            cfg_normalization=True,    
            generator=generator,
        ).images[0]
        
        gen_time = time.time() - start_time
        
        # 5. Speichern
        filename = OUTPUT_DIR / f"doctor_seed{SEED}_{steps:02d}steps.png"
        image.save(filename)
        
        print(f"✅ Bild mit {steps:02d} Steps gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
        
    print("\n🎉 Z-Image Testbilder generiert! Schau in den Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()