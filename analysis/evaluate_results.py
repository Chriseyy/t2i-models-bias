"""
evaluate_results.py
===================
1. MACRO-ANALYSE: Auswertung des gesamten KI-Datensatzes (Macro-Plots).
   -> Inklusive Vergleich ALLER KIs (Gemma, Qwen, InternVL + DeepFace)!
   -> ERWEITERUNG A: Kontinuierliche ITA-Hauttonverteilungen (Violin-Plots).
   -> ERWEITERUNG B: Divergierende Delta-Paritätsbalken für intuitive Bias-Ausschläge.
2. FAIR COMPARISON: Strikte Filterung auf die Human-Eval-Stichprobe mit 
   vollständigen paarweisen KI-Heatmaps und bereinigten Balkendiagrammen.
   -> ERWEITERUNG C: Globales KI-Mensch-Alignment (Genauigkeitsabgleich gegen Ground Truth).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================
# PFADE DEFINIEREN
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent if 'analysis' in Path(__file__).parts else Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

MACRO_DIR = PLOTS_DIR / "MACRO_ALL_DATA"
FAIR_DIR = PLOTS_DIR / "FAIR_COMPARISON"

HUMAN_CSV = OUTPUT_DIR / "human_evaluation.csv"
OLLAMA_CSV = OUTPUT_DIR / "ollama_results.csv"
DEEPFACE_CSV = OUTPUT_DIR / "deepface_results.csv"
SKIN_CSV = OUTPUT_DIR / "skin_metrics_results.csv"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

# =========================================================================
# HILFSFUNKTIONEN
# =========================================================================
def extract_prompt(filename):
    if "_seed" in filename:
        return filename.split("_seed")[0]
    return "unknown"

def clean_category_strings(df):
    for col in df.columns:
        if 'Race' in col or 'Gender' in col:
            df[col] = df[col].astype(str).str.title().str.strip()
            df[col] = df[col].replace('Nan', np.nan)
    return df

def get_clean_evaluator_name(col):
    """Konvertiert lange Spaltennamen in kurze, lesbare Bezeichner für die Plots"""
    col_lower = col.lower()
    if 'gemma' in col_lower: return 'Gemma4'
    if 'qwen' in col_lower: return 'Qwen2.5'
    if 'internvl' in col_lower: return 'InternVL'
    if 'deepface' in col_lower: return 'DeepFace'
    if 'human' in col_lower: return 'Human'
    return col

# =========================================================================
# NEU: EXKLUSIVE ERWEITERUNGS-PLOT-FUNKTIONEN (A, B & C)
# =========================================================================
def plot_continuous_ita_distribution(df, t2i_model, output_folder):
    """ERWEITERUNG A: Generiert hochpräzise Violin-Plots für kontinuierliche ITA-Hauttonwerte"""
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or 'ITA_Value' not in model_df.columns: return

    # Erzwinge numerische Datentypen für kontinuierliche Berechnungen
    model_df['ITA_Value'] = pd.to_numeric(model_df['ITA_Value'], errors='coerce')
    model_df = model_df.dropna(subset=['ITA_Value'])
    if model_df.empty: return

    plt.figure(figsize=(14, 7))
    # 'coolwarm' Farbpalette bildet den Übergang von hellen zu dunklen Hauttönen perfekt ab
    sns.violinplot(data=model_df, x='Prompt_Subject', y='ITA_Value', hue='Prompt_Subject', palette="coolwarm", inner="box", legend=False)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.6, label='ITA Paritätsgrenze (0°)')
    plt.title(f"Kontinuierliche ITA-Hauttonverteilung (Dichte & Streuung) | Modell: {t2i_model.upper()}")
    plt.ylabel("Individual Typology Angle (ITA-Wert in Grad)")
    plt.xlabel("Generierter Prompt")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_folder / "ITA_CONTINUOUS_VIOLIN_DISTRIBUTION.png", dpi=300)
    plt.close()

def plot_diverging_gender_bias(df, t2i_model, category_col, output_folder, file_prefix=""):
    """ERWEITERUNG B: Generiert ein horizontales, divergierendes Delta-Paritätsdiagramm"""
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or category_col not in model_df.columns: return

    # Berechne die relativen Anteile pro Prompt-Thema
    crosstab = pd.crosstab(model_df['Prompt_Subject'], model_df[category_col], normalize='index') * 100
    if 'Man' not in crosstab.columns: crosstab['Man'] = 0.0
    if 'Woman' not in crosstab.columns: crosstab['Woman'] = 0.0

    # Mathematische Delta-Parität: Überhang berechnen (% Männer - % Frauen)
    crosstab['Delta'] = crosstab['Man'] - crosstab['Woman']
    crosstab = crosstab.reset_index().sort_values(by='Delta')

    plt.figure(figsize=(12, 6))
    # Farbkodierung: Sanftes Grün für Frauen-Überhang, Erdiges Orange für Männer-Überhang
    colors = ['#4c9173' if x < 0 else '#df8a5a' for x in crosstab['Delta']]
    
    sns.barplot(data=crosstab, y='Prompt_Subject', x='Delta', palette=colors, hue='Prompt_Subject', legend=False)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=2.0)
    plt.title(f"Divergierende Geschlechterparität (Delta) | Metrik: {category_col} | Modell: {t2i_model.upper()}")
    plt.xlabel("← Frauen-Überhang (%)  |  Männer-Überhang (%) →")
    plt.ylabel("Prompt Thema")
    plt.xlim(-105, 105)
    plt.tight_layout()

    safe_prefix = file_prefix.replace(":", "_").replace("/", "_")
    if safe_prefix: safe_prefix += "_"
    plt.savefig(output_folder / f"{safe_prefix}{category_col}_DIVERGING_DELTA_PARITY.png", dpi=300)
    plt.close()

def plot_global_alignment_comparison(master_df, output_folder):
    """ERWEITERUNG C: Berechnet das methodische Gesamt-Alignment (Genauigkeit) aller KIs vs. Mensch"""
    if master_df.empty: return

    categories = ['Gender', 'Race', 'MST']
    alignment_rows = []

    for cat in categories:
        human_col = f'Human_{cat}'
        if human_col not in master_df.columns: continue

        eval_cols = {}
        if cat != 'MST' and f'DeepFace_{cat}' in master_df.columns:
            eval_cols['DeepFace'] = f'DeepFace_{cat}'

        vlm_cols = [c for c in master_df.columns if c.endswith(f'_VLM_{cat}')]
        for col in vlm_cols:
            clean_name = get_clean_evaluator_name(col)
            eval_cols[clean_name] = col

        # Berechne exakte mathematische Übereinstimmungsraten
        for eval_name, col_name in eval_cols.items():
            valid_sub = master_df.dropna(subset=[human_col, col_name])
            if valid_sub.empty: continue

            matches = (valid_sub[human_col].astype(str).str.lower().str.strip() == 
                       valid_sub[col_name].astype(str).str.lower().str.strip()).sum()
            accuracy = (matches / len(valid_sub)) * 100

            alignment_rows.append({
                'Kategorie': cat,
                'Evaluator': eval_name,
                'Uebereinstimmung': round(accuracy, 2)
            })

    if not alignment_rows: return
    df_align = pd.DataFrame(alignment_rows)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=df_align, x='Kategorie', y='Uebereinstimmung', hue='Evaluator', palette='Set2')
    plt.title("Methodische Kür: Globales KI-Alignment (Exakte Übereinstimmung mit Mensch)")
    plt.ylabel("Exakte Übereinstimmung zur Ground Truth (%)")
    plt.xlabel("Evaluierungs-Dimension")
    plt.ylim(0, 105)
    plt.legend(title="Maschinelle KI", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / "GLOBAL_KI_ALIGNMENT_TO_HUMAN.png", dpi=300)
    plt.close()

# =========================================================================
# PLOT FUNKTIONEN (MACRO & FAIR)
# =========================================================================
def generate_grouped_plots(df, category_col, t2i_model, output_folder, file_prefix="", hue_order=None):
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or category_col not in model_df.columns: return

    safe_prefix = file_prefix.replace(":", "_").replace("/", "_")
    if safe_prefix: safe_prefix += "_"

    plt.figure(figsize=(16, 7))
    sns.countplot(data=model_df, x='Prompt_Subject', hue=category_col, hue_order=hue_order, palette="Set2")
    title_prefix = f"[{file_prefix}] " if file_prefix else ""
    plt.title(f"{title_prefix}Verteilung: {category_col} über alle Prompts ({t2i_model.upper()})")
    plt.ylabel("Absolute Anzahl (Count)")
    plt.xlabel("Generierter Prompt")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / f"{safe_prefix}{category_col}_MASTER_ALL_PROMPTS.png", dpi=300)
    plt.close()

    prompts = model_df['Prompt_Subject'].unique()
    for prompt in prompts:
        prompt_df = model_df[model_df['Prompt_Subject'] == prompt]
        plt.figure(figsize=(8, 6))
        sns.countplot(data=prompt_df, x=category_col, order=hue_order, palette="Set2")
        plt.title(f"{title_prefix}{category_col} | Prompt: {prompt} | Modell: {t2i_model.upper()}")
        plt.ylabel("Absolute Anzahl (Count)")
        plt.xlabel(category_col)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_folder / f"{safe_prefix}{category_col}_SinglePrompt_{prompt}.png", dpi=300)
        plt.close()

def generate_vlm_comparison_plot(df, t2i_model, output_folder):
    model_df = df[df['T2I_Model'] == t2i_model].copy()
    if model_df.empty or 'VLM_Model' not in model_df.columns: return
    vlms = model_df['VLM_Model'].unique()
    if len(vlms) <= 1: return

    categories = [
        ('VLM_Gender', ["Man", "Woman", "Unclear"]), 
        ('VLM_Race', None), 
        ('VLM_MST', [str(i) for i in range(1, 11)] + ["Unclear"])
    ]

    for cat, order in categories:
        if cat not in model_df.columns: continue
        plt.figure(figsize=(12, 6))
        sns.countplot(data=model_df, x='VLM_Model', hue=cat, order=vlms, hue_order=order, palette="Set2")
        plt.title(f"VLM Vergleich (Generell): KIs über das gesamte Modell {t2i_model.upper()} ({cat})")
        plt.ylabel("Absolute Anzahl (Bilder)")
        plt.xlabel("Ollama VLM-Modell")
        plt.legend(title=cat, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_folder / f"VLM_COMPARISON_GENERAL_{cat}.png", dpi=300)
        plt.close()

def generate_all_ai_comparison_plot(df_ollama, df_deepface, t2i_model, output_folder):
    if df_ollama is None or df_deepface is None: return
    
    ol_model = df_ollama[df_ollama['T2I_Model'] == t2i_model].copy()
    df_model = df_deepface[df_deepface['T2I_Model'] == t2i_model].copy()
    
    if ol_model.empty or df_model.empty: return

    combined_rows = []
    
    for _, row in ol_model.iterrows():
        img = row['Image_Name']
        vlm = row['VLM_Model']
        safe_vlm = vlm.replace(":", "_").replace("/", "_")
        if 'VLM_Gender' in row: combined_rows.append({'Image_Name': img, 'Evaluator': safe_vlm, 'Category': 'Gender', 'Prediction': row['VLM_Gender']})
        if 'VLM_Race' in row: combined_rows.append({'Image_Name': img, 'Evaluator': safe_vlm, 'Category': 'Race', 'Prediction': row['VLM_Race']})

    for _, row in df_model.iterrows():
        img = row['Image_Name']
        if 'DeepFace_Gender' in row: combined_rows.append({'Image_Name': img, 'Evaluator': 'DeepFace', 'Category': 'Gender', 'Prediction': row['DeepFace_Gender']})
        if 'DeepFace_Race' in row: combined_rows.append({'Image_Name': img, 'Evaluator': 'DeepFace', 'Category': 'Race', 'Prediction': row['DeepFace_Race']})

    combined_df = pd.DataFrame(combined_rows).dropna(subset=['Prediction'])
    if combined_df.empty: return

    gender_df = combined_df[combined_df['Category'] == 'Gender']
    if not gender_df.empty:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=gender_df, x='Evaluator', hue='Prediction', hue_order=["Man", "Woman", "Unclear"], palette="Set2")
        plt.title(f"Alle KIs im Vergleich (Generell): Gender für Modell {t2i_model.upper()}")
        plt.ylabel("Absolute Anzahl (Bilder)")
        plt.xlabel("Evaluator (VLMs + DeepFace)")
        plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(output_folder / "ALLE_KIS_COMPARISON_GENERAL_Gender.png", dpi=300)
        plt.close()

    race_df = combined_df[combined_df['Category'] == 'Race']
    if not race_df.empty:
        plt.figure(figsize=(14, 7))
        sns.countplot(data=race_df, x='Evaluator', hue='Prediction', palette="Set2")
        plt.title(f"Alle KIs im Vergleich (Generell): Race für Modell {t2i_model.upper()}")
        plt.ylabel("Absolute Anzahl (Bilder)")
        plt.xlabel("Evaluator (VLMs + DeepFace)")
        plt.legend(title="Race", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(output_folder / "ALLE_KIS_COMPARISON_GENERAL_Race.png", dpi=300)
        plt.close()

def plot_evaluator_comparison(df, model_name, category, output_folder):
    """FAIR COMPARISON: Zeigt die Modelle direkt nebeneinander (Max 30 Bilder hoch)"""
    eval_cols = []
    
    if category in ['Gender', 'Race']:
        if f'DeepFace_{category}' in df.columns: 
            eval_cols.append(f'DeepFace_{category}')
        vlm_cols = [c for c in df.columns if c.endswith(f'_VLM_{category}')]
        eval_cols.extend(vlm_cols)
        eval_order = ['DeepFace', 'Gemma4', 'Qwen2.5', 'InternVL']
    elif category == 'MST':
        vlm_cols = [c for c in df.columns if c.endswith(f'_VLM_{category}')]
        eval_cols.extend(vlm_cols)
        eval_order = ['Gemma4', 'Qwen2.5', 'InternVL']
        
    if len(eval_cols) < 2: return
        
    melted_df = df[['Image_Name'] + eval_cols].melt(id_vars='Image_Name', var_name='Evaluator', value_name='Prediction')
    melted_df = melted_df.dropna(subset=['Prediction'])
    
    melted_df['Evaluator'] = melted_df['Evaluator'].apply(get_clean_evaluator_name)
    
    melted_df['Evaluator'] = pd.Categorical(melted_df['Evaluator'], categories=eval_order, ordered=True)
    melted_df = melted_df.sort_values('Evaluator')

    plt.figure(figsize=(12, 6))
    sns.countplot(data=melted_df, x='Evaluator', hue='Prediction', palette="Set2")
    plt.title(f"KI-Vergleich: Modell-Metriken ({category} | {model_name.upper()})")
    plt.ylabel(f"Anzahl der Bilder (Max {len(df)})")
    plt.xlabel("Maschinelle Evaluatoren")
    plt.xticks(rotation=0)
    plt.legend(title=f"Erkannte(s) {category}", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    safe_category = category.replace("/", "_")
    plt.savefig(output_folder / f"VERGLEICH_KI_Modelle_{safe_category}.png", dpi=300)
    plt.close()

def generate_heatmap(merged_df, col1, col2, title, filename_prefix, output_folder, label1, label2):
    """Generischer Heatmap-Plotter für beliebige Evaluatoren-Paare"""
    if col1 not in merged_df.columns or col2 not in merged_df.columns: return
    valid_data = merged_df.dropna(subset=[col1, col2]).copy()
    if valid_data.empty: return
    
    confusion_matrix = pd.crosstab(valid_data[col1], valid_data[col2])
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=True)
    plt.title(title, pad=20)
    plt.ylabel(f"Bewertung von: {label1}")
    plt.xlabel(f"Bewertung von: {label2}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_folder / f"{filename_prefix}.png", dpi=300)
    plt.close()

def export_model_statistics(model_df, model_name, output_folder):
    summary_rows = []
    prompts = model_df['Prompt_Subject'].unique()
    for prompt in prompts:
        pdf = model_df[model_df['Prompt_Subject'] == prompt].copy()
        n_images = len(pdf)
        if n_images == 0: continue
        row = {'Prompt': prompt, 'Analyzed_Images': n_images}
        if 'Human_Gender' in pdf.columns:
            modes = pdf['Human_Gender'].mode()
            row['Majority_Human_Gender'] = modes[0] if not modes.empty else "N/A"
        if 'Human_Race' in pdf.columns:
            modes = pdf['Human_Race'].mode()
            row['Majority_Human_Race'] = modes[0] if not modes.empty else "N/A"

        if 'Human_Gender' in pdf.columns and 'DeepFace_Gender' in pdf.columns:
            disagreements = (pdf['Human_Gender'] != pdf['DeepFace_Gender']).sum()
            row['ErrorRate_DeepFace_Gender_%'] = round((disagreements / n_images) * 100, 1)
        if 'Human_Race' in pdf.columns and 'DeepFace_Race' in pdf.columns:
            disagreements = (pdf['Human_Race'] != pdf['DeepFace_Race']).sum()
            row['ErrorRate_DeepFace_Race_%'] = round((disagreements / n_images) * 100, 1)

        vlm_race_cols = [c for c in pdf.columns if c.endswith('_VLM_Race')]
        for vlm_col in vlm_race_cols:
            vlm_name = vlm_col.replace('_VLM_Race', '')
            if 'Human_Race' in pdf.columns:
                disagreements = (pdf['Human_Race'] != pdf[vlm_col]).sum()
                row[f'ErrorRate_{vlm_name}_Race_%'] = round((disagreements / n_images) * 100, 1)

        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_folder / f"{model_name}_PAPER_SUMMARY.csv", index=False)


# =========================================================================
# MAIN PROCESSING
# =========================================================================
def main():
    print("=" * 70)
    print("📈 STARTE ZWEISTUFIGE AUSWERTUNG (MACRO & FAIR COMPARISON)")
    print("=" * 70)

    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    FAIR_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # TEIL 1: MACRO DATEN LADEN (Alle Bilder, Duplikate entfernt)
    # ---------------------------------------------------------
    raw_dataframes = {}

    if OLLAMA_CSV.exists():
        df_ol = pd.read_csv(OLLAMA_CSV)
        if 'Prompt_Subject' not in df_ol.columns:
            df_ol['Prompt_Subject'] = df_ol['Image_Name'].apply(extract_prompt)
        df_ol = df_ol.drop_duplicates(subset=['Image_Name', 'T2I_Model', 'VLM_Model'], keep='last')
        df_ol = clean_category_strings(df_ol)
        raw_dataframes['Ollama'] = df_ol

    if DEEPFACE_CSV.exists():
        df_df = pd.read_csv(DEEPFACE_CSV)
        if 'Prompt_Subject' not in df_df.columns:
            df_df['Prompt_Subject'] = df_df['Image_Name'].apply(extract_prompt)
        df_df = df_df.drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
        df_df = clean_category_strings(df_df)
        raw_dataframes['DeepFace'] = df_df

    if SKIN_CSV.exists():
        df_skin = pd.read_csv(SKIN_CSV)
        if 'Prompt_Subject' not in df_skin.columns:
            df_skin['Prompt_Subject'] = df_skin['Image_Name'].apply(extract_prompt)
        df_skin = df_skin.drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
        df_skin['ITA_Scale_MST'] = df_skin['ITA_Scale_MST'].astype(str)
        df_skin['MonkScale_RGB'] = df_skin['MonkScale_RGB'].astype(str)
        raw_dataframes['Skin'] = df_skin

    all_models_macro = set()
    for name, df in raw_dataframes.items():
        all_models_macro.update(df['T2I_Model'].unique())

    print(f"\n[TEIL 1] Generiere MACRO Plots für {len(all_models_macro)} Modelle...")
    for model in all_models_macro:
        model_out_dir = MACRO_DIR / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> MACRO: {model.upper()}")

        if 'Ollama' in raw_dataframes:
            df_ollama = raw_dataframes['Ollama']
            vlm_models = df_ollama['VLM_Model'].unique()
            for vlm in vlm_models:
                df_vlm_specific = df_ollama[df_ollama['VLM_Model'] == vlm].copy()
                safe_vlm = vlm.replace(":", "_").replace("/", "_")
                generate_grouped_plots(df_vlm_specific, 'VLM_Gender', model, model_out_dir, file_prefix=safe_vlm, hue_order=["Man", "Woman", "Unclear"])
                generate_grouped_plots(df_vlm_specific, 'VLM_Race', model, model_out_dir, file_prefix=safe_vlm)
                generate_grouped_plots(df_vlm_specific, 'VLM_MST', model, model_out_dir, file_prefix=safe_vlm, hue_order=[str(i) for i in range(1, 11)] + ["Unclear"])
                
                # INTEGRIERTE ERWEITERUNG B: Divergierende Delta-Parität für Ollama VLMs auf Macro-Ebene
                plot_diverging_gender_bias(df_vlm_specific, model, 'VLM_Gender', model_out_dir, file_prefix=safe_vlm)
            
            generate_vlm_comparison_plot(df_ollama, model, model_out_dir)

        if 'DeepFace' in raw_dataframes:
            generate_grouped_plots(raw_dataframes['DeepFace'], 'DeepFace_Gender', model, model_out_dir, hue_order=["Man", "Woman"])
            generate_grouped_plots(raw_dataframes['DeepFace'], 'DeepFace_Race', model, model_out_dir)
            
            # INTEGRIERTE ERWEITERUNG B: Divergierende Delta-Parität für DeepFace auf Macro-Ebene
            plot_diverging_gender_bias(raw_dataframes['DeepFace'], model, 'DeepFace_Gender', model_out_dir, file_prefix="DeepFace")

        df_ol_ref = raw_dataframes.get('Ollama')
        df_df_ref = raw_dataframes.get('DeepFace')
        generate_all_ai_comparison_plot(df_ol_ref, df_df_ref, model, model_out_dir)

        if 'Skin' in raw_dataframes:
            generate_grouped_plots(raw_dataframes['Skin'], 'MonkScale_RGB', model, model_out_dir, hue_order=[str(i) for i in range(1, 11)] + ["Error"])
            generate_grouped_plots(raw_dataframes['Skin'], 'ITA_Scale_MST', model, model_out_dir, hue_order=[str(i) for i in range(1, 11)] + ["Error"])
            
            # INTEGRIERTE ERWEITERUNG A: Kontinuierliche ITA-Hauttonverteilungen (Violin-Plots) auf Macro-Ebene
            plot_continuous_ita_distribution(raw_dataframes['Skin'], model, model_out_dir)

    # ---------------------------------------------------------
    # TEIL 2: FAIR COMPARISON (Strikt auf Human Eval gefiltert)
    # ---------------------------------------------------------
    print("\n[TEIL 2] Generiere FAIR COMPARISON (Gefiltert auf menschliche Stichprobe)...")
    
    if not HUMAN_CSV.exists():
        print("⚠️ HINWEIS: human_evaluation.csv fehlt! Fair Comparison wird übersprungen.")
        return

    df_human = pd.read_csv(HUMAN_CSV).drop_duplicates(subset=['Image_Name', 'T2I_Model'], keep='last')
    if 'Prompt_Subject' not in df_human.columns:
        df_human['Prompt_Subject'] = df_human['Image_Name'].apply(extract_prompt)
    df_human = clean_category_strings(df_human)
    
    valid_images = set(df_human['Image_Name'].unique())

    master_df = df_human.copy()
    
    if 'Ollama' in raw_dataframes:
        df_ol_filtered = raw_dataframes['Ollama'][raw_dataframes['Ollama']['Image_Name'].isin(valid_images)].copy()
        df_ol_pivoted = df_ol_filtered.pivot_table(index=['Image_Name', 'T2I_Model', 'Prompt_Subject'], 
                                                   columns='VLM_Model', 
                                                   values=['VLM_Gender', 'VLM_Race', 'VLM_MST'], 
                                                   aggfunc='first').reset_index()
        df_ol_pivoted.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in df_ol_pivoted.columns]
        cols_to_merge = [c for c in df_ol_pivoted.columns if c not in ['Prompt_Subject']]
        master_df = pd.merge(master_df, df_ol_pivoted[cols_to_merge], on=['Image_Name', 'T2I_Model'], how='left')

    for source in ['DeepFace', 'Skin']:
        if source in raw_dataframes:
            df_filtered = raw_dataframes[source][raw_dataframes[source]['Image_Name'].isin(valid_images)]
            cols_to_merge = [c for c in df_filtered.columns if c not in ['Prompt_Subject']]
            master_df = pd.merge(master_df, df_filtered[cols_to_merge], on=['Image_Name', 'T2I_Model'], how='left')

    master_df.to_csv(FAIR_DIR / "MASTER_ALL_METRICS_FAIR_SUBSET.csv", index=False)

    # INTEGRIERTE ERWEITERUNG C: Globaler Alignment-Abgleich aller KIs gegen den Menschen über das gesamte Subset
    plot_global_alignment_comparison(master_df, FAIR_DIR)

    all_models_fair = df_human['T2I_Model'].unique()

    for model in all_models_fair:
        model_out_dir = FAIR_DIR / model
        model_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> FAIR: {model.upper()}")
        
        model_data = master_df[master_df['T2I_Model'] == model].copy()

        # A) Richter-Vergleiche generieren (Balkendiagramme)
        plot_evaluator_comparison(model_data, model, 'Gender', model_out_dir)
        plot_evaluator_comparison(model_data, model, 'Race', model_out_dir)
        plot_evaluator_comparison(model_data, model, 'MST', model_out_dir)

        # B) Vollständig Paarweise Heatmaps generieren
        for cat in ['Gender', 'Race', 'MST']:
            cat_cols = {}
            if f'Human_{cat}' in model_data.columns:
                cat_cols['Human'] = f'Human_{cat}'
            if cat != 'MST' and f'DeepFace_{cat}' in model_data.columns:
                cat_cols['DeepFace'] = f'DeepFace_{cat}'
                
            vlm_cols = [c for c in model_data.columns if c.endswith(f'_VLM_{cat}')]
            for col in vlm_cols:
                clean_name = get_clean_evaluator_name(col)
                cat_cols[clean_name] = col
                
            eval_names = list(cat_cols.keys())
            for i in range(len(eval_names)):
                for j in range(i + 1, len(eval_names)):
                    name1 = eval_names[i]
                    name2 = eval_names[j]
                    
                    title = f"Heatmap: {name1} vs {name2} ({cat} | {model.upper()})"
                    filename = f"HEATMAP_{name1}_vs_{name2}_{cat}"
                    
                    generate_heatmap(model_data, cat_cols[name1], cat_cols[name2], 
                                     title, filename, model_out_dir, name1, name2)

        # C) Summary CSV generieren
        export_model_statistics(model_data, model, model_out_dir)

    print("\n" + "=" * 70)
    print("🎉 ALLES FERTIG! Makro-Plots und Fair-Comparison erfolgreich erstellt.")

if __name__ == "__main__":
    main()