import streamlit as st
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
#  PALETTE VINTAGE
# ============================================================
CREAM      = "#FAF3E0"
MILK_BROWN = "#C8A97E"
DARK_BROWN = "#7B4F2E"
MED_BROWN  = "#A0724A"
LIGHT_TAN  = "#E8D5B7"
WARM_WHITE = "#FFF8F0"
GRID_COLOR = "#D9C4A8"

heatmap_cmap = sns.color_palette(
    [CREAM, LIGHT_TAN, MILK_BROWN, MED_BROWN, DARK_BROWN], as_cmap=True
)

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Regresi Berganda – Harga Properti",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
#  CSS VINTAGE
# ============================================================
st.markdown(f"""
<style>
    /* Background utama */
    .stApp {{
        background-color: {WARM_WHITE};
        font-family: Georgia, serif;
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {LIGHT_TAN};
        border-right: 2px solid {MILK_BROWN};
    }}
    /* Judul besar */
    h1 {{
        color: {DARK_BROWN} !important;
        font-family: Georgia, serif !important;
        font-weight: bold;
        border-bottom: 2px solid {MILK_BROWN};
        padding-bottom: 8px;
    }}
    h2, h3 {{
        color: {MED_BROWN} !important;
        font-family: Georgia, serif !important;
    }}
    /* Metric cards */
    [data-testid="metric-container"] {{
        background-color: {CREAM};
        border: 1.5px solid {MILK_BROWN};
        border-radius: 8px;
        padding: 10px;
    }}
    [data-testid="stMetricValue"] {{
        color: {DARK_BROWN} !important;
        font-family: Georgia, serif !important;
        font-weight: bold;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MED_BROWN} !important;
        font-family: Georgia, serif !important;
    }}
    /* Upload area */
    [data-testid="stFileUploadDropzone"] {{
        background-color: {CREAM} !important;
        border: 2px dashed {MILK_BROWN} !important;
        border-radius: 10px;
    }}
    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {MILK_BROWN};
        border-radius: 6px;
    }}
    /* Divider */
    hr {{
        border-color: {MILK_BROWN};
    }}
    /* Info/success box */
    .stAlert {{
        background-color: {CREAM};
        border-left-color: {MILK_BROWN};
        color: {DARK_BROWN};
        font-family: Georgia, serif;
    }}
    p, li {{
        color: {DARK_BROWN};
        font-family: Georgia, serif;
    }}
    .stSelectbox label, .stSlider label {{
        color: {DARK_BROWN} !important;
        font-family: Georgia, serif !important;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  HEADER
# ============================================================
st.markdown(f"""
<div style='text-align:center; padding: 20px 0 5px 0;'>
    <h1 style='font-size:2.2rem; color:{DARK_BROWN}; margin-bottom:4px;'>
        🏠 Regresi Berganda – Harga Properti India
    </h1>
    <p style='color:{MED_BROWN}; font-style:italic; font-size:1rem;'>
        Implementasi Multiple Linear Regression | Tema Vintage | scikit-learn
    </p>
</div>
<hr style='border: 1.5px solid {MILK_BROWN}; margin: 0 0 20px 0;'>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"<h2 style='color:{DARK_BROWN};'>⚙️ Konfigurasi</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(f"<p style='color:{MED_BROWN}; font-weight:bold;'>📂 Upload Dataset</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<p style='color:{MED_BROWN}; font-weight:bold;'>🔧 Parameter Model</p>", unsafe_allow_html=True)
    test_size = st.slider("Ukuran Data Test (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random State", value=42, step=1)

    st.markdown("---")
    st.markdown(f"""
    <div style='background:{CREAM}; border:1px solid {MILK_BROWN}; border-radius:8px; padding:12px;'>
        <p style='color:{DARK_BROWN}; font-size:0.85rem; margin:0;'>
        <b>Variabel Model:</b><br>
        🎯 Y &nbsp;→ <code>Price</code><br>
        📐 X₁ → <code>Area_SqFt</code><br>
        🛏 X₂ → <code>Rooms</code><br>
        📅 X₃ → <code>Build_Year</code><br>
        🪑 X₄ → <code>Furnishing</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  HELPER
# ============================================================
def style_ax(ax, title):
    ax.set_facecolor(LIGHT_TAN)
    ax.set_title(title, fontsize=12, fontweight='bold',
                 color=DARK_BROWN, pad=10, fontfamily='serif')
    ax.tick_params(colors=DARK_BROWN, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(MILK_BROWN)
        spine.set_linewidth(1.2)
    ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.xaxis.label.set_color(DARK_BROWN)
    ax.yaxis.label.set_color(DARK_BROWN)

# ============================================================
#  KONTEN UTAMA
# ============================================================
if uploaded_file is None:
    # Landing page saat belum upload
    st.markdown(f"""
    <div style='text-align:center; padding: 60px 20px;
                background:{CREAM}; border: 2px dashed {MILK_BROWN};
                border-radius:16px; margin-top:30px;'>
        <p style='font-size:3rem; margin:0;'>📁</p>
        <h3 style='color:{DARK_BROWN};'>Upload file dataset_2.csv di sidebar kiri</h3>
        <p style='color:{MED_BROWN}; font-style:italic;'>
            Dataset harus memiliki kolom:<br>
            <code>Area_SqFt, Rooms, Build_Year, Furnishing, Price</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
#  LOAD & PREPROCESSING
# ============================================================
df_raw = pd.read_csv(uploaded_file)
df = df_raw.copy()
df.dropna(subset=['Area_SqFt', 'Rooms', 'Furnishing'], inplace=True)
df.reset_index(drop=True, inplace=True)

le = LabelEncoder()
df['Furnishing_enc'] = le.fit_transform(df['Furnishing'])

features = ['Area_SqFt', 'Rooms', 'Build_Year', 'Furnishing_enc']
labels   = ['Luas (sqft)', 'Jumlah Kamar', 'Tahun Bangun', 'Furnishing']
target   = 'Price'

X = df[features]
y = df[target]

# ============================================================
#  MODEL
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=int(random_state)
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred  = model.predict(X_test)
r2      = r2_score(y_test, y_pred)
mae     = mean_absolute_error(y_test, y_pred)
rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
coef    = model.coef_
intercept = model.intercept_

# ============================================================
#  SECTION 1 – INFO DATASET
# ============================================================
st.markdown(f"<h2>📊 Ringkasan Dataset</h2>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Baris (raw)", f"{len(df_raw):,}")
c2.metric("Baris Valid", f"{len(df):,}")
c3.metric("Data Train", f"{len(X_train):,}")
c4.metric("Data Test", f"{len(X_test):,}")

with st.expander("👁 Lihat sampel data (5 baris pertama)"):
    st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# ============================================================
#  SECTION 2 – METRIK MODEL
# ============================================================
st.markdown(f"<h2>📈 Hasil Model Regresi Berganda</h2>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2:.4f}", help="Semakin mendekati 1, semakin baik model")
m2.metric("MAE", f"{mae:,.0f} USD", help="Mean Absolute Error")
m3.metric("RMSE", f"{rmse:,.0f} USD", help="Root Mean Squared Error")

# Persamaan regresi
st.markdown(f"""
<div style='background:{CREAM}; border:1.5px solid {MILK_BROWN};
            border-radius:10px; padding:16px 20px; margin: 16px 0;'>
    <p style='color:{MED_BROWN}; font-size:0.85rem; margin:0 0 6px 0;'>PERSAMAAN REGRESI BERGANDA</p>
    <p style='color:{DARK_BROWN}; font-size:1.05rem; font-weight:bold; font-family:monospace; margin:0;'>
        Y = {intercept:,.0f}
        + ({coef[0]:.2f} × Area_SqFt)
        + ({coef[1]:,.0f} × Rooms)
        + ({coef[2]:,.0f} × Build_Year)
        + ({coef[3]:,.0f} × Furnishing)
    </p>
</div>
""", unsafe_allow_html=True)

# Tabel koefisien
coef_df = pd.DataFrame({
    'Variabel'  : ['Intercept (a)', 'X₁ – Luas (sqft)', 'X₂ – Jumlah Kamar', 'X₃ – Tahun Bangun', 'X₄ – Furnishing'],
    'Koefisien' : [intercept] + list(coef),
    'Interpretasi': [
        'Harga dasar saat semua X = 0',
        f'Setiap +1 sqft → harga naik {coef[0]:,.2f} USD',
        f'Setiap +1 kamar → harga naik {coef[1]:,.0f} USD',
        f'Setiap +1 tahun → harga naik {coef[2]:,.0f} USD',
        f'Encoding furnishing → pengaruh {coef[3]:,.0f} USD',
    ]
})
st.dataframe(coef_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
#  SECTION 3 – HEATMAP
# ============================================================
st.markdown(f"<h2>🔥 Heatmap Korelasi</h2>", unsafe_allow_html=True)

corr_df = df[features + [target]].copy()
corr_df.columns = labels + ['Price']
corr_mat = corr_df.corr()

fig_heat, ax_heat = plt.subplots(figsize=(8, 5.5), facecolor=WARM_WHITE)
ax_heat.set_facecolor(WARM_WHITE)
sns.heatmap(corr_mat, ax=ax_heat, cmap=heatmap_cmap,
            annot=True, fmt=".2f",
            annot_kws={"size": 10, "color": DARK_BROWN, "fontfamily": "serif"},
            linewidths=0.8, linecolor=CREAM,
            cbar_kws={"shrink": 0.8}, square=True)
ax_heat.set_title("Korelasi Antar Variabel", fontsize=13,
                  fontweight='bold', color=DARK_BROWN, pad=12, fontfamily='serif')
ax_heat.tick_params(colors=DARK_BROWN, labelsize=9, rotation=30)
cbar = ax_heat.collections[0].colorbar
cbar.ax.tick_params(colors=DARK_BROWN, labelsize=8)
for spine in ax_heat.spines.values():
    spine.set_edgecolor(MILK_BROWN)
fig_heat.patch.set_facecolor(WARM_WHITE)
plt.tight_layout()
st.pyplot(fig_heat, use_container_width=True)
plt.close(fig_heat)

st.markdown("---")

# ============================================================
#  SECTION 4 – SCATTER PLOTS (3 plot)
# ============================================================
st.markdown(f"<h2>🔵 Scatter Plot</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# --- Scatter 1: Luas vs Harga ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 4), facecolor=WARM_WHITE)
    style_ax(ax1, "Luas (sqft) vs Harga")
    ax1.scatter(df['Area_SqFt'], df[target],
                color=MILK_BROWN, edgecolors=DARK_BROWN,
                alpha=0.5, s=20, linewidth=0.4, label='Data aktual')
    m_, b_ = np.polyfit(df['Area_SqFt'].fillna(0), df[target], 1)
    xs_ = np.linspace(df['Area_SqFt'].min(), df['Area_SqFt'].max(), 200)
    ax1.plot(xs_, m_*xs_ + b_, color=DARK_BROWN, linewidth=1.8,
             linestyle='--', label='Tren linear')
    ax1.set_xlabel("Luas (sqft)", fontsize=9, fontfamily='serif')
    ax1.set_ylabel("Harga (USD)", fontsize=9, fontfamily='serif')
    ax1.legend(fontsize=7, facecolor=CREAM, edgecolor=MILK_BROWN, labelcolor=DARK_BROWN)
    fig1.patch.set_facecolor(WARM_WHITE)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

# --- Scatter 2: Jumlah Kamar vs Harga ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor=WARM_WHITE)
    style_ax(ax2, "Jumlah Kamar vs Harga")
    jitter = np.random.default_rng(7).uniform(-0.2, 0.2, len(df))
    ax2.scatter(df['Rooms'] + jitter, df[target],
                color=MED_BROWN, edgecolors=DARK_BROWN,
                alpha=0.45, s=20, linewidth=0.4)
    for r in sorted(df['Rooms'].dropna().unique()):
        med = df.loc[df['Rooms'] == r, target].median()
        ax2.plot(r, med, marker='D', color=DARK_BROWN, markersize=7, zorder=5)
    ax2.set_xlabel("Jumlah Kamar", fontsize=9, fontfamily='serif')
    ax2.set_ylabel("Harga (USD)", fontsize=9, fontfamily='serif')
    ax2.legend(handles=[
        mpatches.Patch(color=MED_BROWN, label='Data aktual'),
        mpatches.Patch(color=DARK_BROWN, label='Median')],
        fontsize=7, facecolor=CREAM, edgecolor=MILK_BROWN, labelcolor=DARK_BROWN)
    fig2.patch.set_facecolor(WARM_WHITE)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# --- Scatter 3: Tahun Bangun vs Harga ---
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 4), facecolor=WARM_WHITE)
    style_ax(ax3, "Tahun Bangun vs Harga")
    ax3.scatter(df['Build_Year'], df[target],
                color=LIGHT_TAN, edgecolors=DARK_BROWN,
                alpha=0.55, s=20, linewidth=0.5, label='Data aktual')
    m2_, b2_ = np.polyfit(df['Build_Year'], df[target], 1)
    xs2_ = np.linspace(df['Build_Year'].min(), df['Build_Year'].max(), 200)
    ax3.plot(xs2_, m2_*xs2_ + b2_, color=DARK_BROWN, linewidth=1.8,
             linestyle='--', label='Tren linear')
    ax3.set_xlabel("Tahun Bangun", fontsize=9, fontfamily='serif')
    ax3.set_ylabel("Harga (USD)", fontsize=9, fontfamily='serif')
    ax3.legend(fontsize=7, facecolor=CREAM, edgecolor=MILK_BROWN, labelcolor=DARK_BROWN)
    fig3.patch.set_facecolor(WARM_WHITE)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

st.markdown("---")

# ============================================================
#  SECTION 5 – ACTUAL vs PREDICTED
# ============================================================
st.markdown(f"<h2>🎯 Actual vs Predicted</h2>", unsafe_allow_html=True)

col_ap, col_resid = st.columns(2)

with col_ap:
    fig_ap, ax_ap = plt.subplots(figsize=(6, 5), facecolor=WARM_WHITE)
    style_ax(ax_ap, "Actual vs Predicted (Data Test)")
    ax_ap.scatter(y_test, y_pred,
                  color=MILK_BROWN, edgecolors=DARK_BROWN,
                  alpha=0.55, s=30, linewidth=0.5)
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    ax_ap.plot([lo, hi], [lo, hi], color=DARK_BROWN,
               linewidth=2, linestyle='-', label='Ideal (y = x)')
    ax_ap.set_xlabel("Harga Aktual (USD)", fontsize=10, fontfamily='serif')
    ax_ap.set_ylabel("Harga Prediksi (USD)", fontsize=10, fontfamily='serif')
    ax_ap.legend(fontsize=8, facecolor=CREAM, edgecolor=MILK_BROWN, labelcolor=DARK_BROWN)
    ax_ap.text(0.04, 0.94, f"R² = {r2:.4f}",
               transform=ax_ap.transAxes, fontsize=10,
               color=DARK_BROWN, fontfamily='serif',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=CREAM,
                         edgecolor=MILK_BROWN, alpha=0.9))
    fig_ap.patch.set_facecolor(WARM_WHITE)
    plt.tight_layout()
    st.pyplot(fig_ap, use_container_width=True)
    plt.close(fig_ap)

with col_resid:
    # Residual plot
    residuals = y_test.values - y_pred
    fig_res, ax_res = plt.subplots(figsize=(6, 5), facecolor=WARM_WHITE)
    style_ax(ax_res, "Residual Plot")
    ax_res.scatter(y_pred, residuals,
                   color=MED_BROWN, edgecolors=DARK_BROWN,
                   alpha=0.55, s=30, linewidth=0.5)
    ax_res.axhline(0, color=DARK_BROWN, linewidth=1.8, linestyle='--')
    ax_res.set_xlabel("Harga Prediksi (USD)", fontsize=10, fontfamily='serif')
    ax_res.set_ylabel("Residual (Actual – Predicted)", fontsize=10, fontfamily='serif')
    fig_res.patch.set_facecolor(WARM_WHITE)
    plt.tight_layout()
    st.pyplot(fig_res, use_container_width=True)
    plt.close(fig_res)

# ============================================================
#  FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; padding: 10px 0;'>
    <p style='color:{MILK_BROWN}; font-style:italic; font-size:0.85rem;'>
        Regresi Berganda &nbsp;|&nbsp; Python – scikit-learn &nbsp;|&nbsp; Tema Vintage
    </p>
</div>
""", unsafe_allow_html=True)
