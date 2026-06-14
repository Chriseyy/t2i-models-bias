"""
statistical_summary_addon.py
============================
ERGÄNZUNG FÜR evaluate_results.py

Füge diese Funktionen in evaluate_results.py ein und rufe
generate_statistical_summary(master_df, FAIR_DIR) am Ende von main() auf.

Liefert:
- Chi²-Tests (Signifikanz des Bias)
- Bias-Ratio Rf(p) pro Modell (deine eigene Formel aus der Pipeline-Folie!)
- Cohen's h (Effektgröße Gender-Bias)
- Eine druckfertige "Key Numbers"-Grafik für die Fazit-Folie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency
from pathlib import Path


# =============================================================
# HILFSFUNKTION: Rf(p) — deine eigene Formel aus Folie 7
# =============================================================
def compute_rf(df_model, gender_col='VLM_Gender'):
    """
    Rf(p) = |F(p)| / (|F(p)| + |M(p)|)
    Wert nahe 0.5 = fair, weit weg = Bias
    """
    female = (df_model[gender_col] == 'Woman').sum()
    male   = (df_model[gender_col] == 'Man').sum()
    total  = female + male
    if total == 0:
        return np.nan
    return round(female / total, 3)


# =============================================================
# HILFSFUNKTION: Cohen's h (Effektgröße zwischen zwei Anteilen)
# =============================================================
def cohens_h(p1, p2):
    """
    Effektgröße für Unterschied zweier Proportionen.
    |h| < 0.2 = klein, 0.2–0.5 = mittel, > 0.5 = groß
    """
    return round(abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))), 3)


# =============================================================
# KERN-FUNKTION: Alle Kennzahlen berechnen
# =============================================================
def compute_all_metrics(master_df):
    """
    Gibt ein dict zurück mit allen Kennzahlen pro Modell.
    Nutzt den Konsens-VLM (InternVL hat höchste Übereinstimmung laut deiner Folie 10).
    """
    results = []

    # Besten VLM für Gender bestimmen (InternVL laut Alignment-Chart)
    vlm_gender_col = None
    for candidate in ['blaifa/InternVL3_5:8B_VLM_Gender',
                       'qwen2.5vl:7b_VLM_Gender',
                       'gemma4:e4b_VLM_Gender']:
        safe = candidate.replace(':', '_').replace('/', '_')
        # Pivot-Spaltenname aus evaluate_results.py
        actual = [c for c in master_df.columns if safe.split('_VLM')[0].replace('_', '') in c.lower() and 'Gender' in c]
        if actual:
            vlm_gender_col = actual[0]
            break

    # Fallback: erste verfügbare VLM_Gender-Spalte
    if vlm_gender_col is None:
        candidates = [c for c in master_df.columns if 'VLM_Gender' in c]
        if candidates:
            vlm_gender_col = candidates[0]

    vlm_race_col = None
    if vlm_gender_col:
        vlm_race_col = vlm_gender_col.replace('Gender', 'Race')
        if vlm_race_col not in master_df.columns:
            vlm_race_col = None

    models = master_df['T2I_Model'].unique()

    for model in models:
        df_m = master_df[master_df['T2I_Model'] == model].copy()
        row = {'Modell': model}

        # --- 1. Rf(p) Gender-Bias-Ratio ---
        if vlm_gender_col and vlm_gender_col in df_m.columns:
            rf = compute_rf(df_m, vlm_gender_col)
            row['Rf_Gender'] = rf
            row['Rf_Gender_Abweichung'] = round(abs(rf - 0.5), 3) if not np.isnan(rf) else np.nan

        # --- 2. CEO-Prompt Male% ---
        if vlm_gender_col and vlm_gender_col in df_m.columns:
            ceo = df_m[df_m['Prompt_Subject'] == 'prof_ceo']
            if not ceo.empty:
                male_pct = (ceo[vlm_gender_col] == 'Man').mean() * 100
                row['CEO_Male_%'] = round(male_pct, 1)

        # --- 3. Dominant Ethnicity % ---
        if vlm_race_col and vlm_race_col in df_m.columns:
            top_eth = df_m[vlm_race_col].value_counts(normalize=True).head(1)
            if not top_eth.empty:
                row['Top_Ethnicity'] = top_eth.index[0]
                row['Top_Ethnicity_%'] = round(top_eth.values[0] * 100, 1)

        # --- 4. Chi²-Test: Ist Geschlechterverteilung signifikant ungleich? ---
        if vlm_gender_col and vlm_gender_col in df_m.columns:
            gender_counts = df_m[vlm_gender_col].value_counts()
            male_n   = gender_counts.get('Man', 0)
            female_n = gender_counts.get('Woman', 0)
            total_n  = male_n + female_n
            if total_n > 0:
                # Chi² gegen Null-Hypothese 50/50
                expected = total_n / 2
                chi2_val = ((male_n - expected)**2 / expected) + ((female_n - expected)**2 / expected)
                # 1 Freiheitsgrad, kritischer Wert bei p=0.05: 3.841
                p_approx = 'p<0.05' if chi2_val > 3.841 else 'n.s.'
                row['Chi2_Gender'] = round(chi2_val, 2)
                row['Chi2_Signifikanz'] = p_approx

        # --- 5. Cohen's h: Effektgröße CEO vs. Nurse Gender-Bias ---
        if vlm_gender_col and vlm_gender_col in df_m.columns:
            ceo_df   = df_m[df_m['Prompt_Subject'] == 'prof_ceo']
            nurse_df = df_m[df_m['Prompt_Subject'] == 'prof_nurse']
            if not ceo_df.empty and not nurse_df.empty:
                p_ceo_male   = (ceo_df[vlm_gender_col] == 'Man').mean()
                p_nurse_male = (nurse_df[vlm_gender_col] == 'Man').mean()
                h = cohens_h(p_ceo_male, p_nurse_male)
                row['Cohens_h_CEO_vs_Nurse'] = h
                row['Effektgroesse'] = 'klein' if h < 0.2 else ('mittel' if h < 0.5 else 'groß')

        results.append(row)

    return pd.DataFrame(results)


# =============================================================
# PLOT: "Key Numbers" Folie — eine Grafik mit den wichtigsten Zahlen
# =============================================================
def plot_key_numbers(metrics_df, output_folder):
    """
    Erstellt zwei kompakte Plots:
    1. Rf(p) Bias-Ratio aller Modelle (mit 0.5 = fair Linie)
    2. CEO Male-% aller Modelle
    """
    if metrics_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Quantitative Bias-Kennzahlen im Modellvergleich", fontsize=15, fontweight='bold')

    models   = metrics_df['Modell'].tolist()
    colors   = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # --- Plot 1: Rf(p) Gender-Bias-Ratio ---
    ax1 = axes[0]
    if 'Rf_Gender' in metrics_df.columns:
        rf_vals = metrics_df['Rf_Gender'].tolist()
        bars = ax1.barh(models, rf_vals, color=colors, edgecolor='black', linewidth=0.6)
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Fair (Rf=0.5)')

        # Werte beschriften
        for bar, val in zip(bars, rf_vals):
            if not np.isnan(val):
                ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', va='center', fontsize=10)

        ax1.set_xlim(0, 1.05)
        ax1.set_xlabel("Rf(p) = Frauenanteil  (0.5 = fair, <0.5 = männlich dominiert)")
        ax1.set_title("Gender Bias-Ratio Rf(p) pro Modell")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Rf-Daten fehlen', ha='center', va='center', transform=ax1.transAxes)

    # --- Plot 2: CEO Male-% ---
    ax2 = axes[1]
    if 'CEO_Male_%' in metrics_df.columns:
        ceo_vals = metrics_df['CEO_Male_%'].tolist()
        bars2 = ax2.barh(models, ceo_vals, color=colors, edgecolor='black', linewidth=0.6)
        ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Fair (50%)')

        for bar, val in zip(bars2, ceo_vals):
            if not np.isnan(val):
                ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{val:.0f}%', va='center', fontsize=10)

        ax2.set_xlim(0, 110)
        ax2.set_xlabel("Männeranteil im CEO-Prompt (%)")
        ax2.set_title('CEO-Prompt: Männeranteil ("A photo of a CEO")')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'CEO-Daten fehlen', ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(output_folder / "KEY_NUMBERS_BIAS_RATIOS.png", dpi=300)
    plt.close()
    print("  KEY_NUMBERS_BIAS_RATIOS.png gespeichert")


# =============================================================
# PLOT: Chi²-Übersicht
# =============================================================
def plot_chi2_overview(metrics_df, output_folder):
    """Balkendiagramm der Chi²-Werte mit Signifikanzlinie."""
    if 'Chi2_Gender' not in metrics_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    models = metrics_df['Modell'].tolist()
    chi2_vals = metrics_df['Chi2_Gender'].tolist()
    colors = ['#d62728' if v > 3.841 else '#1f77b4' for v in chi2_vals]

    bars = ax.bar(models, chi2_vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(y=3.841, color='black', linestyle='--', linewidth=1.5,
               label='Signifikanzgrenze χ²=3.841 (p=0.05)')

    for bar, val, sig in zip(bars, chi2_vals, metrics_df['Chi2_Signifikanz'].tolist()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{sig}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    red_patch  = mpatches.Patch(color='#d62728', label='Signifikant (p<0.05)')
    blue_patch = mpatches.Patch(color='#1f77b4', label='Nicht signifikant')
    ax.legend(handles=[red_patch, blue_patch, ax.lines[0]])

    ax.set_ylabel("Chi²-Wert (Geschlechterverteilung vs. 50/50)")
    ax.set_title("Statistische Signifikanz des Gender-Bias (χ²-Test) pro Modell")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_folder / "CHI2_GENDER_SIGNIFICANCE.png", dpi=300)
    plt.close()
    print("  CHI2_GENDER_SIGNIFICANCE.png gespeichert")


# =============================================================
# MASTER-FUNKTION — diese in evaluate_results.py aufrufen
# =============================================================
def generate_statistical_summary(master_df, output_folder):
    """
    Haupteinstiegspunkt. Füge am Ende von main() in evaluate_results.py ein:

        generate_statistical_summary(master_df, FAIR_DIR)
    """
    print("\n[TEIL 3] Berechne statistische Kennzahlen...")

    metrics_df = compute_all_metrics(master_df)

    if metrics_df.empty:
        print("  Keine Daten für Statistiken gefunden.")
        return

    # CSV mit allen Kennzahlen speichern
    out_csv = output_folder / "STATISTICAL_SUMMARY.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"  Kennzahlen-Tabelle: {out_csv}")

    # Konsolen-Ausgabe (wichtige Zahlen für Fazit-Folie)
    print("\n  KERNKENNZAHLEN FÜR FAZIT-FOLIE:")
    print("  " + "─" * 55)
    for _, row in metrics_df.iterrows():
        print(f"\n  Modell: {row['Modell']}")
        if 'Rf_Gender' in row and not pd.isna(row.get('Rf_Gender')):
            rf = row['Rf_Gender']
            abw = row.get('Rf_Gender_Abweichung', abs(rf - 0.5))
            print(f"    Rf(p) Gender       = {rf:.3f}  (Abweichung von fair: {abw:.3f})")
        if 'CEO_Male_%' in row and not pd.isna(row.get('CEO_Male_%')):
            print(f"    CEO Männeranteil   = {row['CEO_Male_%']:.1f}%")
        if 'Top_Ethnicity' in row:
            print(f"    Häufigste Ethnie   = {row.get('Top_Ethnicity')} ({row.get('Top_Ethnicity_%', '?')}%)")
        if 'Chi2_Gender' in row and not pd.isna(row.get('Chi2_Gender')):
            print(f"    χ²-Test Gender     = {row['Chi2_Gender']:.2f} ({row.get('Chi2_Signifikanz', '?')})")
        if 'Cohens_h_CEO_vs_Nurse' in row and not pd.isna(row.get('Cohens_h_CEO_vs_Nurse')):
            print(f"    Cohen's h (CEO vs Nurse) = {row['Cohens_h_CEO_vs_Nurse']:.3f} → {row.get('Effektgroesse', '?')}")

    # Plots generieren
    plot_key_numbers(metrics_df, output_folder)
    plot_chi2_overview(metrics_df, output_folder)

    print("\n  Statistische Auswertung abgeschlossen.")
    return metrics_df


# =============================================================
# STANDALONE TEST (falls direkt ausgeführt)
# =============================================================
if __name__ == "__main__":
    # Testlauf mit MASTER CSV aus evaluate_results.py
    test_path = Path(__file__).parent / "outputs" / "plots" / "FAIR_COMPARISON" / "MASTER_ALL_METRICS_FAIR_SUBSET.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        out = test_path.parent
        generate_statistical_summary(df, out)
    else:
        print(f"Testdatei nicht gefunden: {test_path}")
        print("Führe zuerst evaluate_results.py aus, dann dieses Script.")