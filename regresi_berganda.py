import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  PALETTE VINTAGE (Putih & Coklat Susu)
# ============================================================
CREAM       = "#FAF3E0"      # latar utama (cream/putih gading)
MILK_BROWN  = "#C8A97E"      # coklat susu terang – aksen
DARK_BROWN  = "#7B4F2E"      # coklat tua – judul / teks penting
MED_BROWN   = "#A0724A"      # coklat sedang – elemen grafis
LIGHT_TAN   = "#E8D5B7"      # tan muda – background panel/scatter
WARM_WHITE  = "#FFF8F0"      # putih hangat – background figure
LINE_COLOR  = "#7B4F2E"      # warna garis regresi
GRID_COLOR  = "#D9C4A8"      # warna grid

# Palet urutan kontinu untuk heatmap
heatmap_cmap = sns.color_palette(
    [CREAM, LIGHT_TAN, MILK_BROWN, MED_BROWN, DARK_BROWN], as_cmap=True
)

# ============================================================
#  1. LOAD & PREPROCESSING DATA
# ============================================================
df = pd.read_csv('/mnt/user-data/uploads/dataset_2.csv')
df.dropna(subset=['Area_SqFt', 'Rooms', 'Furnishing'], inplace=True)
df.reset_index(drop=True, inplace=True)

le_furnishing = LabelEncoder()
df['Furnishing_enc'] = le_furnishing.fit_transform(df['Furnishing'])   # 0=Furnished,1=Semi,2=Unfurnished

# Variabel bebas (X) dan terikat (Y)
features  = ['Area_SqFt', 'Rooms', 'Build_Year', 'Furnishing_enc']
labels    = ['Luas (sqft)', 'Jumlah Kamar', 'Tahun Bangun', 'Furnishing']
target    = 'Price'

X = df[features]
y = df[target]

# ============================================================
#  2. SPLIT & FIT MODEL
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrik
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
coef = model.coef_
intercept = model.intercept_

# ============================================================
#  3. FIGURE UTAMA  (3 baris × 2 kolom)
# ============================================================
fig = plt.figure(figsize=(18, 20), facecolor=WARM_WHITE)
fig.patch.set_facecolor(WARM_WHITE)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.55, wspace=0.38,
                       left=0.07, right=0.96,
                       top=0.91, bottom=0.06)

# ──────────────────────────────────────────────────────────
# JUDUL UTAMA
# ──────────────────────────────────────────────────────────
fig.text(0.50, 0.965, "Regresi Berganda – Harga Properti India",
         ha='center', va='top',
         fontsize=22, fontweight='bold', color=DARK_BROWN,
         fontfamily='serif')
fig.text(0.50, 0.945, "Dataset Properti India  |  1 091 observasi  |  Variabel Y: Price  |  Variabel X: Area, Rooms, Build_Year, Furnishing",
         ha='center', va='top',
         fontsize=11, color=MED_BROWN, fontstyle='italic',
         fontfamily='serif')

# Garis dekoratif bawah judul
line = plt.Line2D([0.07, 0.93], [0.935, 0.935],
                  transform=fig.transFigure,
                  color=MILK_BROWN, linewidth=1.5)
fig.add_artist(line)

# ──────────────────────────────────────────────────────────
# HELPER: styling umum ax
# ──────────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor(LIGHT_TAN)
    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=DARK_BROWN, pad=10, fontfamily='serif')
    ax.tick_params(colors=DARK_BROWN, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(MILK_BROWN)
        spine.set_linewidth(1.2)
    ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.xaxis.label.set_color(DARK_BROWN)
    ax.yaxis.label.set_color(DARK_BROWN)

# ==============================================================
#  PANEL 1 (baris 0, kolom 0) : HEATMAP KORELASI
# ==============================================================
ax_heat = fig.add_subplot(gs[0, 0])
ax_heat.set_facecolor(LIGHT_TAN)

corr_df = df[features + [target]].copy()
corr_df.columns = labels + ['Price']
corr_mat = corr_df.corr()

mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)   # tampilkan penuh (simetri)
sns.heatmap(corr_mat,
            ax=ax_heat,
            cmap=heatmap_cmap,
            annot=True, fmt=".2f",
            annot_kws={"size": 9, "color": DARK_BROWN, "fontfamily": "serif"},
            linewidths=0.8, linecolor=CREAM,
            cbar_kws={"shrink": 0.75},
            square=True)

ax_heat.set_title("Heatmap Korelasi Antar Variabel",
                  fontsize=13, fontweight='bold',
                  color=DARK_BROWN, pad=10, fontfamily='serif')
ax_heat.tick_params(colors=DARK_BROWN, labelsize=8.5, rotation=30)
cbar = ax_heat.collections[0].colorbar
cbar.ax.tick_params(colors=DARK_BROWN, labelsize=8)
for spine in ax_heat.spines.values():
    spine.set_edgecolor(MILK_BROWN)

# ==============================================================
#  PANEL 2 (baris 0, kolom 1) : SCATTER – Luas vs Harga
# ==============================================================
ax_s1 = fig.add_subplot(gs[0, 1])
style_ax(ax_s1, "Scatter: Luas (sqft) vs Harga")
ax_s1.scatter(df['Area_SqFt'], df[target],
              color=MILK_BROWN, edgecolors=DARK_BROWN,
              alpha=0.55, s=30, linewidth=0.5, label='Data aktual')
# Garis tren
m, b = np.polyfit(df['Area_SqFt'].fillna(0), df[target], 1)
xs = np.linspace(df['Area_SqFt'].min(), df['Area_SqFt'].max(), 200)
ax_s1.plot(xs, m*xs + b, color=DARK_BROWN, linewidth=1.8,
           linestyle='--', label='Tren linear')
ax_s1.set_xlabel("Luas (sqft)", fontsize=10, fontfamily='serif')
ax_s1.set_ylabel("Harga (USD)", fontsize=10, fontfamily='serif')
ax_s1.legend(fontsize=8, facecolor=CREAM, edgecolor=MILK_BROWN,
             labelcolor=DARK_BROWN)

# ==============================================================
#  PANEL 3 (baris 1, kolom 0) : SCATTER – Jumlah Kamar vs Harga
# ==============================================================
ax_s2 = fig.add_subplot(gs[1, 0])
style_ax(ax_s2, "Scatter: Jumlah Kamar vs Harga")

jitter = np.random.default_rng(7).uniform(-0.2, 0.2, len(df))
ax_s2.scatter(df['Rooms'] + jitter, df[target],
              color=MED_BROWN, edgecolors=DARK_BROWN,
              alpha=0.50, s=30, linewidth=0.5, label='Data aktual')
# Box-plot ringkas per kamar digantikan titik median
for r in sorted(df['Rooms'].dropna().unique()):
    med = df.loc[df['Rooms'] == r, target].median()
    ax_s2.plot(r, med, marker='D', color=DARK_BROWN, markersize=8, zorder=5)
ax_s2.set_xlabel("Jumlah Kamar", fontsize=10, fontfamily='serif')
ax_s2.set_ylabel("Harga (USD)", fontsize=10, fontfamily='serif')
med_patch = mpatches.Patch(color=DARK_BROWN, label='Median per kamar')
ax_s2.legend(handles=[
    mpatches.Patch(color=MED_BROWN, label='Data aktual'),
    med_patch], fontsize=8,
    facecolor=CREAM, edgecolor=MILK_BROWN, labelcolor=DARK_BROWN)

# ==============================================================
#  PANEL 4 (baris 1, kolom 1) : SCATTER – Tahun Bangun vs Harga
# ==============================================================
ax_s3 = fig.add_subplot(gs[1, 1])
style_ax(ax_s3, "Scatter: Tahun Bangun vs Harga")
ax_s3.scatter(df['Build_Year'], df[target],
              color=LIGHT_TAN, edgecolors=DARK_BROWN,
              alpha=0.60, s=30, linewidth=0.6, label='Data aktual')
m2, b2 = np.polyfit(df['Build_Year'], df[target], 1)
xs2 = np.linspace(df['Build_Year'].min(), df['Build_Year'].max(), 200)
ax_s3.plot(xs2, m2*xs2 + b2, color=DARK_BROWN, linewidth=1.8,
           linestyle='--', label='Tren linear')
ax_s3.set_xlabel("Tahun Bangun", fontsize=10, fontfamily='serif')
ax_s3.set_ylabel("Harga (USD)", fontsize=10, fontfamily='serif')
ax_s3.legend(fontsize=8, facecolor=CREAM, edgecolor=MILK_BROWN,
             labelcolor=DARK_BROWN)

# ==============================================================
#  PANEL 5 (baris 2, kolom 0) : ACTUAL vs PREDICTED
# ==============================================================
ax_ap = fig.add_subplot(gs[2, 0])
style_ax(ax_ap, "Actual vs Predicted (Data Test)")
ax_ap.scatter(y_test, y_pred,
              color=MILK_BROWN, edgecolors=DARK_BROWN,
              alpha=0.55, s=30, linewidth=0.5)
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
ax_ap.plot([lo, hi], [lo, hi], color=DARK_BROWN,
           linewidth=2, linestyle='-', label='Ideal (y=x)')
ax_ap.set_xlabel("Harga Aktual (USD)", fontsize=10, fontfamily='serif')
ax_ap.set_ylabel("Harga Prediksi (USD)", fontsize=10, fontfamily='serif')
ax_ap.legend(fontsize=8, facecolor=CREAM, edgecolor=MILK_BROWN,
             labelcolor=DARK_BROWN)
ax_ap.text(0.04, 0.94, f"R² = {r2:.4f}",
           transform=ax_ap.transAxes,
           fontsize=10, color=DARK_BROWN, fontfamily='serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor=CREAM,
                     edgecolor=MILK_BROWN, alpha=0.9))

# ==============================================================
#  PANEL 6 (baris 2, kolom 1) : RINGKASAN MODEL (teks)
# ==============================================================
ax_info = fig.add_subplot(gs[2, 1])
ax_info.set_facecolor(CREAM)
for spine in ax_info.spines.values():
    spine.set_edgecolor(MILK_BROWN)
    spine.set_linewidth(1.5)
ax_info.axis('off')

furnish_map = {0: "Furnished", 1: "Semi-Furnished", 2: "Unfurnished"}

info_lines = [
    ("PERSAMAAN REGRESI BERGANDA", True, DARK_BROWN, 13),
    ("", False, DARK_BROWN, 9),
    ("Y = a + b₁X₁ + b₂X₂ + b₃X₃ + b₄X₄", True, MED_BROWN, 11),
    ("", False, DARK_BROWN, 7),
    (f"Intercept  (a)         =  {intercept:,.0f}", False, DARK_BROWN, 10),
    (f"b₁ · Luas (sqft)       =  {coef[0]:.2f}", False, DARK_BROWN, 10),
    (f"b₂ · Jumlah Kamar     =  {coef[1]:,.0f}", False, DARK_BROWN, 10),
    (f"b₃ · Tahun Bangun      =  {coef[2]:,.0f}", False, DARK_BROWN, 10),
    (f"b₄ · Furnishing (enc)  =  {coef[3]:,.0f}", False, DARK_BROWN, 10),
    ("", False, DARK_BROWN, 7),
    ("── Metrik Evaluasi ─────────────────", False, MILK_BROWN, 9),
    (f"R² Score      : {r2:.4f}", True, DARK_BROWN, 11),
    (f"MAE           : {mae:,.0f} USD", False, DARK_BROWN, 10),
    (f"RMSE          : {rmse:,.0f} USD", False, DARK_BROWN, 10),
    ("", False, DARK_BROWN, 7),
    ("── Info Dataset ─────────────────────", False, MILK_BROWN, 9),
    (f"Total data    : 1 091 baris (setelah dropna)", False, DARK_BROWN, 9.5),
    (f"Data train    : {len(X_train)} baris (80%)", False, DARK_BROWN, 9.5),
    (f"Data test     : {len(X_test)} baris (20%)", False, DARK_BROWN, 9.5),
    (f"Sumber        : Kaggle – India Property Dataset", False, DARK_BROWN, 9.5),
]

y_pos = 0.97
for text, bold, color, fs in info_lines:
    fw = 'bold' if bold else 'normal'
    ax_info.text(0.05, y_pos, text,
                 transform=ax_info.transAxes,
                 fontsize=fs, fontweight=fw,
                 color=color, fontfamily='serif',
                 verticalalignment='top')
    y_pos -= 0.057 if text else 0.025

# ── Footer ──
fig.text(0.50, 0.025,
         "Regresi Berganda  |  Python – scikit-learn  |  Tema Vintage",
         ha='center', fontsize=9, color=MILK_BROWN,
         fontstyle='italic', fontfamily='serif')

plt.savefig('/mnt/user-data/outputs/regresi_berganda_vintage.png',
            dpi=160, bbox_inches='tight',
            facecolor=WARM_WHITE)
print("Gambar disimpan.")
print(f"R² = {r2:.4f}  |  MAE = {mae:,.0f}  |  RMSE = {rmse:,.0f}")
print(f"Intercept = {intercept:,.0f}")
print(f"Koefisien: Area={coef[0]:.2f}, Rooms={coef[1]:.2f}, Build_Year={coef[2]:.2f}, Furnishing={coef[3]:.2f}")
