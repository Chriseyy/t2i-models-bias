"""
evaluate_results.py
===================
Generiert strukturierte Plots in Unterordnern.
NEU: Berücksichtigt mehrere Ollama-Modelle (VLMs) getrennt und vergleicht sie!
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent if 'analysis' in Path(__file__).parts else Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

HUMAN_CSV = OUTPUT_DIR / "human_evaluation.csv"
OLLAMA_CSV = OUTPUT_DIR / "ollama_results.csv"
DEEPFACE_CSV = OUTPUT_DIR / "deepface_results.csv"
SKIN_CSV = OUTPUT_DIR / "skin_metrics_results.csv"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

def extract_prompt(filename):
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

def generate_grouped_plots(df, category_col, t2i_model, output_folder, file_prefix="", hue_order=None):
    """
    Generiert den Master-Plot und Einzel-Plots. 
    file_prefix wird genutzt, um z.B. das VLM-Modell in den Dateinamen zu schreiben.
    """
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or category_col not in model_df.columns:
        return

    model_df['Prompt_Label'] = model_df['Prompt_Subject'] + " - " + t2i_model.lower()
    
    # Prefix für den Dateinamen säubern (z.B. gemma4:e4b -> gemma4_e4b)
    safe_prefix = file_prefix.replace(":", "_").replace("/", "_")
    if safe_prefix: safe_prefix += "_"

    # 1. MASTER-PLOT
    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=model_df, 
        x='Prompt_Label', 
        hue=category_col,
        hue_order=hue_order,
        palette="Set2"
    )
    title_prefix = f"[{file_prefix}] " if file_prefix else ""
    plt.title(f"{title_prefix}Master-Übersicht: {category_col} über alle Prompts ({t2i_model.upper()})")
    plt.ylabel("Anzahl (Count)")
    plt.xlabel("Prompt - Modell")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / f"{safe_prefix}{category_col}_MASTER_ALL_PROMPTS.png", dpi=300)
    plt.close()

    # 2. EINZEL-PLOTS
    prompts = model_df['Prompt_Subject'].unique()
    for prompt in prompts:
        prompt_df = model_df[model_df['Prompt_Subject'] == prompt]
        
        plt.figure(figsize=(8, 6))
        sns.countplot(
            data=prompt_df, 
            x=category_col, 
            order=hue_order,
            palette="Set2"
        )
        plt.title(f"{title_prefix}{category_col} | Prompt: {prompt} | Modell: {t2i_model.upper()}")
        plt.ylabel("Anzahl (Count)")
        plt.xlabel(category_col)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / f"{safe_prefix}{category_col}_SinglePrompt_{prompt}.png", dpi=300)
        plt.close()

def generate_vlm_comparison_plot(df, t2i_model, output_folder):
    """
    NEU: Ein Spezial-Plot, der vergleicht, wie die verschiedenen VLMs 
    (Gemma, Qwen, etc.) dasselbe T2I-Modell bewerten!
    """
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or 'VLM_Model' not in model_df.columns:
        return
        
    vlms = model_df['VLM_Model'].unique()
    if len(vlms) <= 1:
        return # Braucht keinen Vergleich, wenn es nur ein VLM gibt

    # Vergleiche VLM_Gender
    plt.figure(figsize=(10, 6))
    sns.countplot(data=model_df, x='VLM_Model', hue='VLM_Gender', hue_order=["Man", "Woman", "Unclear"], palette="Set2")
    plt.title(f"VLM Vergleich: Wie bewerten verschiedene KIs das Modell {t2i_model.upper()}? (Gender)")
    plt.ylabel("Anzahl der Bilder")
    plt.xlabel("Ollama VLM-Modell")
    plt.legend(title="VLM_Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / "VLM_COMPARISON_Gender.png", dpi=300)
    plt.close()

def main():
    print("=" * 60)
    print("📈 STARTE ORDNERSPEZIFISCHE PLOT-GENERIERUNG")
    print("=" * 60)

    dataframes = {}

    if HUMAN_CSV.exists():
        df_human = pd.read_csv(HUMAN_CSV)
        if 'Prompt_Subject' not in df_human.columns:
            df_human['Prompt_Subject'] = df_human['Image_Name'].apply(extract_prompt)
        dataframes['Human'] = df_human

    if OLLAMA_CSV.exists():
        dataframes['Ollama'] = pd.read_csv(OLLAMA_CSV)

    if DEEPFACE_CSV.exists():
        df_df = pd.read_csv(DEEPFACE_CSV)
        if 'DeepFace_Race' in df_df.columns:
            df_df['DeepFace_Race'] = df_df['DeepFace_Race'].str.title()
        dataframes['DeepFace'] = df_df

    if SKIN_CSV.exists():
        df_skin = pd.read_csv(SKIN_CSV)
        df_skin['ITA_Scale_MST'] = df_skin['ITA_Scale_MST'].astype(str)
        df_skin['MonkScale_RGB'] = df_skin['MonkScale_RGB'].astype(str)
        dataframes['Skin'] = df_skin

    all_models = set()
    for name, df in dataframes.items():
        all_models.update(df['T2I_Model'].unique())

    for model in all_models:
        model_out_dir = PLOTS_DIR / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 Erstelle Plots im Ordner: outputs/plots/{model}/")

        # 1. Human Eval Plots
        if 'Human' in dataframes:
            generate_grouped_plots(dataframes['Human'], 'Human_Gender', model, model_out_dir, hue_order=["Man", "Woman", "Unclear"])
            generate_grouped_plots(dataframes['Human'], 'Human_Race', model, model_out_dir)
            generate_grouped_plots(dataframes['Human'], 'Human_MST', model, model_out_dir, hue_order=[str(i) for i in range(1, 11)] + ["Unclear"])

        # 2. Ollama / VLM Plots (GETRENNT NACH VLM MODELL!)
        if 'Ollama' in dataframes:
            df_ollama = dataframes['Ollama']
            vlm_models = df_ollama['VLM_Model'].unique()
            
            for vlm in vlm_models:
                # Filtere Ollama Daten so, dass nur das aktuelle VLM (z.B. Gemma) übrig bleibt
                df_vlm_specific = df_ollama[df_ollama['VLM_Model'] == vlm].copy()
                
                # Wir übergeben den VLM-Namen als 'file_prefix', damit die Plots nicht überschrieben werden
                generate_grouped_plots(df_vlm_specific, 'VLM_Gender', model, model_out_dir, file_prefix=vlm, hue_order=["Man", "Woman", "Unclear"])
                generate_grouped_plots(df_vlm_specific, 'VLM_Race', model, model_out_dir, file_prefix=vlm)
                generate_grouped_plots(df_vlm_specific, 'VLM_MST', model, model_out_dir, file_prefix=vlm, hue_order=[str(i) for i in range(1, 11)] + ["Unclear"])
            
            # NEU: Ein Plot, der Gemma vs. Qwen vs. InternVL direkt nebeneinander stellt
            generate_vlm_comparison_plot(df_ollama, model, model_out_dir)

        # 3. DeepFace Plots
        if 'DeepFace' in dataframes:
            generate_grouped_plots(dataframes['DeepFace'], 'DeepFace_Gender', model, model_out_dir, hue_order=["Man", "Woman"])
            generate_grouped_plots(dataframes['DeepFace'], 'DeepFace_Race', model, model_out_dir)

        # 4. Skin Metrics Plots
        if 'Skin' in dataframes:
            generate_grouped_plots(dataframes['Skin'], 'MonkScale_RGB', model, model_out_dir, hue_order=[str(i) for i in range(1, 11)] + ["Error"])
            generate_grouped_plots(dataframes['Skin'], 'ITA_Scale_MST', model, model_out_dir, hue_order=[str(i) for i in range(1, 11)] + ["Error"])

    print("\n" + "=" * 60)
    print("🔗 VEREINE ALLE DATEN ZU EINER MASTER-METRIK-DATEI")
    
    if len(dataframes) > 0:
        master_df = list(dataframes.values())[0][['Image_Name', 'T2I_Model', 'Prompt_Subject']].drop_duplicates()
        
        # Für die Master CSV pivotieren wir die Ollama Tabelle, damit es übersichtlich bleibt
        if 'Ollama' in dataframes:
            df_ol = dataframes['Ollama'].copy()
            # Wir machen aus "VLM_Gender" -> "gemma4:e4b_VLM_Gender"
            df_ol_pivoted = df_ol.pivot_table(index=['Image_Name', 'T2I_Model', 'Prompt_Subject'], 
                                              columns='VLM_Model', 
                                              values=['VLM_Gender', 'VLM_Race', 'VLM_MST'], 
                                              aggfunc='first').reset_index()
            # Spaltennamen flachklopfen
            df_ol_pivoted.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in df_ol_pivoted.columns]
            dataframes['Ollama_Flat'] = df_ol_pivoted
            del dataframes['Ollama']

        for source, df in dataframes.items():
            cols_to_merge = [col for col in df.columns if col not in ['T2I_Model', 'Prompt_Subject']]
            master_df = pd.merge(master_df, df[cols_to_merge], on='Image_Name', how='left')

        master_csv_path = OUTPUT_DIR / "MASTER_ALL_METRICS_COMBINED.csv"
        master_df.to_csv(master_csv_path, index=False)
        print(f"✅ Master-CSV erstellt: {master_csv_path.name}")

    print("=" * 60)
    print("🎉 ALLES FERTIG!")

if __name__ == "__main__":
    main()