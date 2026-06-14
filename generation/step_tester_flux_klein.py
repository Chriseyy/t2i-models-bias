"""
step_tester_flux.py
===================
Ablation Study für FLUX.2-klein-9B (Destilliertes Modell).
Zeigt, wie das Modell in gewaltigen Sprüngen (1 bis 4 Steps) 
das Bild formt, im Gegensatz zu klassischen Diffusionsmodellen.
"""

import os
import torch
import time
from pathlib import Path
from diffusers import Flux2KleinPipeline
from dotenv import load_dotenv

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/flux_klein")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter
PROMPT = "a photo of a doctor"
SEED = 103
# Bei einem 4-Step-Modell testen wir die Einzelschritte und prüfen,
# ob mehr als 4 Steps (z.B. 6 oder 8) noch etwas verändern.
# STEPS_TO_TEST = [1, 2, 3, 4, 5, 6]  # not base model
STEPS_TO_TEST = [1, 3, 5, 10, 20, 30, 40, 45, 50, 55, 60]

def main():
    print("=" * 60)
    print("KI-ZEITLUPE: FLUX.2-KLEIN (Destilliert)")
    print("=" * 60)
    
    # 3. Modell laden
    print("Lade FLUX.2-klein-9B...")
    pipe = Flux2KleinPipeline.from_pretrained(
        # "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ).to("cuda")
    
    pipe.enable_model_cpu_offload()  # bei base wichtig

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    

    print("\nStarte Generierungen...")
    
    # 4. Loop durch die Steps
    for steps in STEPS_TO_TEST:
        # Wieder extrem wichtig: Generator bei JEDEM Step neu mit Seed 101 starten!
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=steps,
            # guidance_scale=1.0, # MUSS bei FLUX-klein 1.0 sein! be inciht base
            guidance_scale=4.0,  # bei base
            generator=generator,
        ).images[0]
        gen_time = time.time() - start_time
        
        # 5. Speichern
        filename = OUTPUT_DIR / f"doctor_seed{SEED}_{steps:02d}steps.png"
        image.save(filename)
        
        print(f"Bild mit {steps:02d} Steps gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
        
    print("\nFLUX Testbilder generiert! Schau in den Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()