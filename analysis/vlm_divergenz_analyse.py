"""
vlm_divergence_analysis.py
==========================
Wissenschaftliche Auswertung der Inter-Rater-Reliabilität zwischen verschiedenen VLMs.
Berechnet Konsens, Divergenz und den "Lone-Wolf"-Effekt für ALLE beteiligten Modelle.
Generiert am Ende automatisch wissenschaftliche Plots (Balkendiagramme) für die Masterarbeit.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================
# PFADE & EINSTELLUNGEN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent 
INPUT_CSV = PROJECT_ROOT / "outputs" / "ollama_results.csv"

# NEUER AUSGABE-ORDNER (Wird automatisch erstellt, falls er nicht existiert)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "vlm_analyse"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT_CSV = OUTPUT_DIR / "vlm_full_comparison.csv"
OUTPUT_DISPUTE_CSV = OUTPUT_DIR / "vlm_dispute_cases.csv"
PLOT_AGREEMENT = OUTPUT_DIR / "vlm_agreement_distribution.png"
PLOT_LONEWOLF = OUTPUT_DIR / "vlm_lonewolf_analysis.png"

# Die 3 Hauptmodelle für das Voting
M1 = 'gemma4:e4b'
M2 = 'blaifa/InternVL3_5:8B'  
M3 = 'qwen2.5vl:7b'

def main():
    print("=" * 75)
    print("UNIVERSELLE VLM-DIVERGENZ-ANALYSE GESTARTET")
    print("=" * 75)

    if not INPUT_CSV.exists():
        print(f"Fehler: Die Datei {INPUT_CSV} wurde nicht gefunden.")
        return

    # 1. Daten einlesen und filtern
    df = pd.read_csv(INPUT_CSV)
    df = df[df['VLM_Model'].isin([M1, M2, M3])]
    
    # 2. Daten Pivotieren (Modelle nebeneinander legen)
    pivot_df = df.pivot_table(
        index=['Image_Name', 'T2I_Model', 'Prompt_Subject'],
        columns='VLM_Model',
        values=['VLM_Gender', 'VLM_Race', 'VLM_MST'],
        aggfunc='first'
    )
    
    # Spaltennamen flach machen
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Prüfen, ob alle 3 Modelle vorhanden sind
    req_cols = [f"VLM_Gender_{M1}", f"VLM_Gender_{M2}", f"VLM_Gender_{M3}"]
    missing_cols = [c for c in req_cols if c not in pivot_df.columns]
    if missing_cols:
        print(f"Warnung: Es fehlen noch Daten für Modelle: {missing_cols}.")
        return
    
    # Nur Bilder behalten, die von allen 3 KIs bewertet wurden
    pivot_df = pivot_df.dropna(subset=req_cols)
    total_images = len(pivot_df)
    print(f"Datensatz geladen: {total_images} Bilder wurden von allen 3 VLMs analysiert.\n")

    # =============================================================
    # DATENSAMMLUNG FÜR DIE PLOTS
    # =============================================================
    plot_data_agreement = {'Dimension': [], 'Full Consensus': [], 'Majority': [], 'Total Divergence': []}
    plot_data_lonewolf = {'Dimension': [], 'Gemma4': [], 'InternVL': [], 'Qwen2.5': []}

    dimensions = [('Gender', 'VLM_Gender'), ('Race', 'VLM_Race'), ('MST', 'VLM_MST')]
    
    for dim_name, col_prefix in dimensions:
        
        c_m1 = f"{col_prefix}_{M1}" # Gemma
        c_m2 = f"{col_prefix}_{M2}" # InternVL
        c_m3 = f"{col_prefix}_{M3}" # Qwen

        # 1. Konsens-Analyse
        def calculate_agreement(row):
            votes = [str(row[c_m1]), str(row[c_m2]), str(row[c_m3])]
            unique_votes = len(set(votes))
            if unique_votes == 1: return "Full Consensus"
            elif unique_votes == 2: return "Majority"
            else: return "Total Divergence"
                
        # 2. Universelle Lone-Wolf Detektoren
        def lone_wolf_m1(row): return str(row[c_m2]) == str(row[c_m3]) and str(row[c_m1]) != str(row[c_m2])
        def lone_wolf_m2(row): return str(row[c_m1]) == str(row[c_m3]) and str(row[c_m2]) != str(row[c_m1])
        def lone_wolf_m3(row): return str(row[c_m1]) == str(row[c_m2]) and str(row[c_m3]) != str(row[c_m1])

        # Funktionen anwenden
        pivot_df[f'{dim_name}_Agreement'] = pivot_df.apply(calculate_agreement, axis=1)
        pivot_df[f'{dim_name}_LW_{M1}'] = pivot_df.apply(lone_wolf_m1, axis=1)
        pivot_df[f'{dim_name}_LW_{M2}'] = pivot_df.apply(lone_wolf_m2, axis=1)
        pivot_df[f'{dim_name}_LW_{M3}'] = pivot_df.apply(lone_wolf_m3, axis=1)

        # ---------------------------------------------------------
        # STATISTIKEN ZÄHLEN
        # ---------------------------------------------------------
        counts = pivot_df[f'{dim_name}_Agreement'].value_counts()
        consensus = counts.get("Full Consensus", 0)
        majority = counts.get("Majority", 0)
        total_div = counts.get("Total Divergence", 0)
        
        lw_m1 = pivot_df[f'{dim_name}_LW_{M1}'].sum()
        lw_m2 = pivot_df[f'{dim_name}_LW_{M2}'].sum()
        lw_m3 = pivot_df[f'{dim_name}_LW_{M3}'].sum()

        # Daten für Plots speichern
        plot_data_agreement['Dimension'].append(dim_name)
        plot_data_agreement['Full Consensus'].append(consensus / total_images * 100)
        plot_data_agreement['Majority'].append(majority / total_images * 100)
        plot_data_agreement['Total Divergence'].append(total_div / total_images * 100)

        plot_data_lonewolf['Dimension'].append(dim_name)
        plot_data_lonewolf['Gemma4'].append(lw_m1)
        plot_data_lonewolf['InternVL'].append(lw_m2)
        plot_data_lonewolf['Qwen2.5'].append(lw_m3)

        # Konsolen-Ausgabe
        print(f"--- Dimension: {dim_name.upper()} ---")
        print(f"  Voller Konsens (3/3):     {consensus} Bilder ({consensus/total_images*100:.1f}%)")
        print(f"  Mehrheit (2/3):           {majority} Bilder ({majority/total_images*100:.1f}%)")
        print(f"  Totale Divergenz (1/1/1): {total_div} Bilder ({total_div/total_images*100:.1f}%)\n")
        print(f"  Lone-Wolf (Wer weicht von der Mehrheit ab?):")
        print(f"     -> Gemma4:   {lw_m1}x")
        print(f"     -> InternVL: {lw_m2}x")
        print(f"     -> Qwen2.5:  {lw_m3}x\n")

    # =============================================================
    # PLOTS GENERIEREN & SPEICHERN
    # =============================================================
    print("Generiere Diagramme...")

    # PLOT 1: Agreement Distribution (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.5
    dims = plot_data_agreement['Dimension']
    
    p1 = ax.bar(dims, plot_data_agreement['Full Consensus'], bar_width, label='Voller Konsens (3/3)', color='#4CAF50')
    p2 = ax.bar(dims, plot_data_agreement['Majority'], bar_width, bottom=plot_data_agreement['Full Consensus'], label='Mehrheit (2/3)', color='#FFC107')
    p3 = ax.bar(dims, plot_data_agreement['Total Divergence'], bar_width, bottom=np.array(plot_data_agreement['Full Consensus']) + np.array(plot_data_agreement['Majority']), label='Totale Divergenz (1/1/1)', color='#F44336')

    ax.set_ylabel('Prozent der Bilder (%)')
    ax.set_title('Inter-Rater-Reliabilität der VLMs nach Dimensionen')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.savefig(PLOT_AGREEMENT, dpi=300)
    plt.close()

    # PLOT 2: Lone Wolf Analysis (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dims))
    width = 0.25

    ax.bar(x - width, plot_data_lonewolf['Gemma4'], width, label='Gemma 4', color='#1f77b4')
    ax.bar(x, plot_data_lonewolf['InternVL'], width, label='InternVL', color='#ff7f0e')
    ax.bar(x + width, plot_data_lonewolf['Qwen2.5'], width, label='Qwen 2.5', color='#2ca02c')

    ax.set_ylabel('Absolute Anzahl (Bilder)')
    ax.set_title('Lone-Wolf-Effekt: Welches VLM widerspricht der Mehrheit?')
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_LONEWOLF, dpi=300)
    plt.close()

    # =============================================================
    # EXPORTE FÜR DEN HUMAN-IN-THE-LOOP
    # =============================================================
    dispute_condition = (
        (pivot_df['Gender_Agreement'] != "Full Consensus") | 
        (pivot_df['Race_Agreement'] != "Full Consensus") | 
        (pivot_df['MST_Agreement'] != "Full Consensus")
    )
    dispute_df = pivot_df[dispute_condition]
    
    pivot_df.to_csv(OUTPUT_REPORT_CSV, index=False)
    dispute_df.to_csv(OUTPUT_DISPUTE_CSV, index=False)

    print("=" * 75)
    print(f" EXPORT ABGESCHLOSSEN!")
    print(f"  Alle Dateien gespeichert in: {OUTPUT_DIR}")
    print(f"  CSV: vlm_full_comparison.csv & vlm_dispute_cases.csv")
    print(f"  Plot 1: vlm_agreement_distribution.png")
    print(f"  Plot 2: vlm_lonewolf_analysis.png")
    print("=" * 75)

if __name__ == "__main__":
    main()