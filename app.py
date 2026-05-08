import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Saham BEI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# KONSTANTA
# ============================================================
SAHAM = {
    "BBCA.JK": "Bank Central Asia (Perbankan)",
    "TLKM.JK": "Telkom Indonesia (Telekomunikasi)",
    "ADRO.JK": "Adaro Energy (Energi/Tambang)",
    "GOTO.JK": "GoTo Gojek Tokopedia (Teknologi)",
    "BMRI.JK": "Bank Mandiri (Perbankan)",
}
SEKTOR = {
    "BBCA.JK": "🏦 Perbankan",
    "TLKM.JK": "📡 Telekomunikasi",
    "ADRO.JK": "⛏️ Energi/Tambang",
    "GOTO.JK": "💻 Teknologi",
    "BMRI.JK": "🏦 Perbankan",
}
TIME_STEP   = 60
SPLIT_RATIO = 0.8
DATA_DIR    = "dataset/"
MODEL_DIR   = "saved_model/"

# ============================================================
# FUNGSI HELPER
# ============================================================
@st.cache_data
def load_data(ticker):
    fname = os.path.join(DATA_DIR, ticker.replace('.', '_') + '.csv')
    df = pd.read_csv(fname, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna()
    df = df[df['Volume'] > 0]
    return df

@st.cache_resource
def load_models(ticker):
    rnn_path  = os.path.join(MODEL_DIR, f"rnn_{ticker.replace('.','_')}.h5")
    lstm_path = os.path.join(MODEL_DIR, f"lstm_{ticker.replace('.','_')}.h5")
    rnn  = load_model(rnn_path,  compile=False)
    lstm = load_model(lstm_path, compile=False)
    return rnn, lstm

def preprocess(df, time_step=TIME_STEP, split_ratio=SPLIT_RATIO):
    series = df['Open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series)
    n_train = int(len(scaled) * split_ratio)
    train_scaled = scaled[:n_train]
    valid_scaled = scaled[n_train:]

    def make_xy(data, ts):
        X, y = [], []
        for i in range(ts, len(data)):
            X.append(data[i-ts:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_tr, y_tr = make_xy(train_scaled, time_step)
    X_v,  y_v  = make_xy(valid_scaled, time_step)
    X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
    X_v  = X_v.reshape(X_v.shape[0],   X_v.shape[1],  1)
    y_tr = y_tr.reshape(-1, 1)
    y_v  = y_v.reshape(-1, 1)
    return X_tr, y_tr, X_v, y_v, scaler, n_train

def hitung_metrik(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / (y_true_inv + 1e-8))) * 100
    r2   = r2_score(y_true_inv, y_pred_inv)
    return rmse, mae, mape, r2

def prediksi_besok(df, model, scaler):
    last_open   = df['Open'].values[-TIME_STEP:]
    last_scaled = scaler.transform(last_open.reshape(-1, 1))
    X_input     = last_scaled.reshape(1, TIME_STEP, 1)
    pred        = scaler.inverse_transform(model.predict(X_input, verbose=0))[0, 0]
    return pred

def get_mape_label(mape):
    if mape < 5:   return "🟢 Sangat Baik"
    elif mape < 10: return "🟡 Baik"
    elif mape < 20: return "🟠 Cukup"
    else:           return "🔴 Kurang Baik"

def get_r2_label(r2):
    if r2 > 0.95:  return "🟢 Sangat Baik"
    elif r2 > 0.85: return "🟡 Baik"
    elif r2 > 0.70: return "🟠 Cukup"
    else:           return "🔴 Kurang Baik"

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Bursa_Efek_Indonesia_Logo.svg/320px-Bursa_Efek_Indonesia_Logo.svg.png", width=180)
    st.markdown("---")
    st.markdown("## ⚙️ Pengaturan")

    ticker_selected = st.selectbox(
        "📌 Pilih Saham",
        options=list(SAHAM.keys()),
        format_func=lambda x: f"{x} — {SAHAM[x].split('(')[0].strip()}"
    )

    model_selected = st.selectbox(
        "🤖 Pilih Model",
        options=["Keduanya (RNN & LSTM)", "Simple RNN", "LSTM"]
    )

    st.markdown("---")
    st.markdown("### 📊 Informasi Saham")
    st.markdown(f"**Ticker:** `{ticker_selected}`")
    st.markdown(f"**Emiten:** {SAHAM[ticker_selected].split('(')[0].strip()}")
    st.markdown(f"**Sektor:** {SEKTOR[ticker_selected]}")
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown(
        "Aplikasi prediksi harga pembukaan saham BEI menggunakan "
        "**Deep Learning** (RNN & LSTM).  \n\n"
        "*Penulisan Ilmiah — Program Studi Informatika*"
    )

# ============================================================
# HEADER UTAMA
# ============================================================
st.title("📈 Prediksi Harga Pembukaan Saham BEI")
st.markdown("**Multi-Sektor Bursa Efek Indonesia — Model RNN & LSTM**")
st.markdown("---")

# ============================================================
# LOAD DATA & MODEL
# ============================================================
with st.spinner(f"Memuat data dan model untuk {ticker_selected}..."):
    try:
        df = load_data(ticker_selected)
        rnn_model, lstm_model = load_models(ticker_selected)
        X_tr, y_tr, X_v, y_v, scaler, n_train = preprocess(df)

        y_pred_rnn  = rnn_model.predict(X_v,   verbose=0)
        y_pred_lstm = lstm_model.predict(X_v,   verbose=0)

        rmse_rnn,  mae_rnn,  mape_rnn,  r2_rnn  = hitung_metrik(y_v, y_pred_rnn,  scaler)
        rmse_lstm, mae_lstm, mape_lstm, r2_lstm = hitung_metrik(y_v, y_pred_lstm, scaler)

        pred_rnn_besok  = prediksi_besok(df, rnn_model,  scaler)
        pred_lstm_besok = prediksi_besok(df, lstm_model, scaler)
        harga_terakhir  = df['Open'].values[-1]
        tanggal_terakhir = df['Date'].iloc[-1]

        data_loaded = True
    except Exception as e:
        st.error(f"❌ Gagal memuat data/model: {e}")
        st.info("Pastikan folder `dataset/` dan `saved_model/` sudah tersedia dan notebook sudah dijalankan.")
        data_loaded = False

if data_loaded:

    # ============================================================
    # TAB NAVIGASI
    # ============================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Ringkasan",
        "📊 Eksplorasi Data",
        "🔮 Prediksi",
        "📉 Evaluasi Model",
        "⚖️ Perbandingan"
    ])

    # ============================================================
    # TAB 1 — RINGKASAN
    # ============================================================
    with tab1:
        st.subheader(f"📌 {ticker_selected} — {SAHAM[ticker_selected]}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Harga Open Terakhir",  f"Rp {harga_terakhir:,.0f}",  f"Per {tanggal_terakhir.date()}")
        col2.metric("📅 Total Data",           f"{len(df):,} hari",          f"{df['Date'].iloc[0].date()} – {df['Date'].iloc[-1].date()}")
        col3.metric("🤖 RNN — Prediksi Besok", f"Rp {pred_rnn_besok:,.0f}",  f"{((pred_rnn_besok-harga_terakhir)/harga_terakhir*100):+.2f}%")
        col4.metric("🧠 LSTM — Prediksi Besok",f"Rp {pred_lstm_besok:,.0f}", f"{((pred_lstm_besok-harga_terakhir)/harga_terakhir*100):+.2f}%")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📋 Ringkasan Metrik RNN")
            st.markdown(f"| Metrik | Nilai | Status |")
            st.markdown(f"|--------|-------|--------|")
            st.markdown(f"| RMSE | Rp {rmse_rnn:,.2f} | — |")
            st.markdown(f"| MAE  | Rp {mae_rnn:,.2f}  | — |")
            st.markdown(f"| MAPE | {mape_rnn:.2f}% | {get_mape_label(mape_rnn)} |")
            st.markdown(f"| R²   | {r2_rnn:.4f}   | {get_r2_label(r2_rnn)} |")

        with col_b:
            st.markdown("#### 📋 Ringkasan Metrik LSTM")
            st.markdown(f"| Metrik | Nilai | Status |")
            st.markdown(f"|--------|-------|--------|")
            st.markdown(f"| RMSE | Rp {rmse_lstm:,.2f} | — |")
            st.markdown(f"| MAE  | Rp {mae_lstm:,.2f}  | — |")
            st.markdown(f"| MAPE | {mape_lstm:.2f}% | {get_mape_label(mape_lstm)} |")
            st.markdown(f"| R²   | {r2_lstm:.4f}   | {get_r2_label(r2_lstm)} |")

        st.markdown("---")
        winner = "LSTM" if rmse_lstm < rmse_rnn else "Simple RNN"
        st.success(f"✅ **Model Terbaik untuk {ticker_selected}: {winner}** (berdasarkan RMSE terendah)")

    # ============================================================
    # TAB 2 — EKSPLORASI DATA
    # ============================================================
    with tab2:
        st.subheader("📊 Eksplorasi Data Historis")

        # Statistik deskriptif
        st.markdown("#### 📋 Statistik Deskriptif")
        st.dataframe(
            df[['Open','High','Low','Close','Volume']].describe().round(2),
            use_container_width=True
        )

        st.markdown("---")

        # Grafik harga + volume
        st.markdown("#### 📈 Riwayat Harga Open & Volume")
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = ax1.twinx()
        ax1.plot(df['Date'], df['Open'], color='#1f77b4', linewidth=1.3, label='Harga Open', zorder=2)
        ax2.bar(df['Date'], df['Volume'], alpha=0.15, color='gray', width=1.5, label='Volume')
        ax1.set_ylabel('Harga Open (IDR)', color='#1f77b4')
        ax2.set_ylabel('Volume', color='gray')
        ax1.set_title(f'Riwayat Harga Open — {ticker_selected}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Tabel data mentah
        st.markdown("#### 🗂️ Data Mentah")
        n_show = st.slider("Tampilkan N baris terakhir:", 10, 100, 30)
        st.dataframe(df.tail(n_show).reset_index(drop=True), use_container_width=True)

        # Download data
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data CSV",
            data=csv_data,
            file_name=f"{ticker_selected.replace('.','_')}_data.csv",
            mime="text/csv"
        )

    # ============================================================
    # TAB 3 — PREDIKSI
    # ============================================================
    with tab3:
        st.subheader("🔮 Prediksi Harga Open Hari Berikutnya")

        col1, col2, col3 = st.columns(3)
        col1.metric("📅 Tanggal Data Terakhir",  str(tanggal_terakhir.date()))
        col2.metric("💰 Harga Open Terakhir",     f"Rp {harga_terakhir:,.2f}")
        col3.metric("⏳ Lookback Window",          f"{TIME_STEP} hari")

        st.markdown("---")

        colA, colB = st.columns(2)
        with colA:
            delta_rnn = pred_rnn_besok - harga_terakhir
            pct_rnn   = (delta_rnn / harga_terakhir) * 100
            st.markdown("### 🤖 Simple RNN")
            st.metric(
                label="Prediksi Harga Open",
                value=f"Rp {pred_rnn_besok:,.2f}",
                delta=f"Rp {delta_rnn:+,.2f} ({pct_rnn:+.2f}%)"
            )
            trend_rnn = "📈 Naik" if delta_rnn > 0 else "📉 Turun"
            st.info(f"Tren prediksi RNN: **{trend_rnn}** dari harga terakhir")

        with colB:
            delta_lstm = pred_lstm_besok - harga_terakhir
            pct_lstm   = (delta_lstm / harga_terakhir) * 100
            st.markdown("### 🧠 LSTM")
            st.metric(
                label="Prediksi Harga Open",
                value=f"Rp {pred_lstm_besok:,.2f}",
                delta=f"Rp {delta_lstm:+,.2f} ({pct_lstm:+.2f}%)"
            )
            trend_lstm = "📈 Naik" if delta_lstm > 0 else "📉 Turun"
            st.info(f"Tren prediksi LSTM: **{trend_lstm}** dari harga terakhir")

        st.markdown("---")

        # Grafik 60 hari terakhir + prediksi
        st.markdown("#### 📉 Grafik 60 Hari Terakhir + Prediksi Besok")
        last_60_dates  = df['Date'].values[-TIME_STEP:]
        last_60_prices = df['Open'].values[-TIME_STEP:]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(last_60_dates, last_60_prices, color='#1f77b4', linewidth=1.5, label='Harga Aktual (60 hari)')
        ax.scatter([last_60_dates[-1]], [last_60_prices[-1]], color='#1f77b4', s=60, zorder=5)

        # Titik prediksi (hari berikutnya)
        from datetime import timedelta
        next_date = pd.to_datetime(last_60_dates[-1]) + timedelta(days=1)
        ax.scatter([next_date], [pred_rnn_besok],  color='#ff7f0e', s=120, zorder=6, label=f'Prediksi RNN: Rp {pred_rnn_besok:,.0f}',  marker='*')
        ax.scatter([next_date], [pred_lstm_besok], color='#2ca02c', s=120, zorder=6, label=f'Prediksi LSTM: Rp {pred_lstm_besok:,.0f}', marker='*')
        ax.axvline(x=last_60_dates[-1], color='gray', linestyle=':', linewidth=1.5, label='Batas prediksi')
        ax.set_title(f'60 Hari Terakhir + Prediksi Besok — {ticker_selected}', fontweight='bold')
        ax.set_ylabel('Harga Open (IDR)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.warning("⚠️ **Disclaimer:** Prediksi ini hanya untuk keperluan akademis dan tidak dapat dijadikan saran investasi.")

    # ============================================================
    # TAB 4 — EVALUASI MODEL
    # ============================================================
    with tab4:
        st.subheader("📉 Evaluasi Performa Model")

        # Inverse transform
        y_v_inv      = scaler.inverse_transform(y_v)
        pred_rnn_inv = scaler.inverse_transform(y_pred_rnn)
        pred_lstm_inv= scaler.inverse_transform(y_pred_lstm)

        start_valid  = n_train + TIME_STEP
        end_valid    = start_valid + len(y_v_inv)
        date_valid   = df['Date'].values[start_valid:end_valid]
        min_len      = min(len(date_valid), len(pred_rnn_inv), len(pred_lstm_inv), len(y_v_inv))
        date_valid   = date_valid[:min_len]
        pred_rnn_inv = pred_rnn_inv[:min_len]
        pred_lstm_inv= pred_lstm_inv[:min_len]
        y_v_inv      = y_v_inv[:min_len]

        # Pilih model untuk ditampilkan
        show_models = []
        if model_selected in ["Keduanya (RNN & LSTM)", "Simple RNN"]:
            show_models.append(("Simple RNN", pred_rnn_inv, '#ff7f0e', rmse_rnn, mae_rnn, mape_rnn, r2_rnn))
        if model_selected in ["Keduanya (RNN & LSTM)", "LSTM"]:
            show_models.append(("LSTM", pred_lstm_inv, '#2ca02c', rmse_lstm, mae_lstm, mape_lstm, r2_lstm))

        for model_name, pred_inv, color, rmse, mae, mape, r2 in show_models:
            st.markdown(f"#### Model: {model_name}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMSE",    f"Rp {rmse:,.2f}")
            m2.metric("MAE",     f"Rp {mae:,.2f}")
            m3.metric("MAPE",    f"{mape:.2f}%",   get_mape_label(mape))
            m4.metric("R²",      f"{r2:.4f}",      get_r2_label(r2))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(date_valid, y_v_inv,   label='Aktual',              color='#1f77b4', linewidth=1.8)
            ax.plot(date_valid, pred_inv,  label=f'Prediksi {model_name}', color=color, linewidth=1.5, linestyle='--')
            ax.fill_between(date_valid, y_v_inv.flatten(), pred_inv.flatten(), alpha=0.1, color=color)
            ax.set_title(f'Aktual vs Prediksi {model_name} — {ticker_selected}', fontweight='bold')
            ax.set_ylabel('Harga Open (IDR)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("---")

        # Download hasil evaluasi
        df_hasil = pd.DataFrame({
            'Date':        date_valid,
            'Aktual':      y_v_inv.flatten(),
            'Pred_RNN':    pred_rnn_inv.flatten(),
            'Pred_LSTM':   pred_lstm_inv.flatten(),
        })
        csv_eval = df_hasil.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Hasil Prediksi CSV",
            data=csv_eval,
            file_name=f"{ticker_selected.replace('.','_')}_hasil_prediksi.csv",
            mime="text/csv"
        )

    # ============================================================
    # TAB 5 — PERBANDINGAN RNN vs LSTM SEMUA SAHAM
    # ============================================================
    with tab5:
        st.subheader("⚖️ Perbandingan RNN vs LSTM — Semua Saham")
        st.info("Memuat data semua saham, harap tunggu...")

        rows = []
        progress = st.progress(0)
        for idx, (tkr, nama) in enumerate(SAHAM.items()):
            try:
                df_t = load_data(tkr)
                rnn_t, lstm_t = load_models(tkr)
                _, _, Xv_t, yv_t, sc_t, nt_t = preprocess(df_t)
                yp_rnn_t  = rnn_t.predict(Xv_t,  verbose=0)
                yp_lstm_t = lstm_t.predict(Xv_t, verbose=0)
                rr, mr, mpr, r2r = hitung_metrik(yv_t, yp_rnn_t,  sc_t)
                rl, ml, mpl, r2l = hitung_metrik(yv_t, yp_lstm_t, sc_t)
                sektor = nama.split('(')[-1].replace(')', '')
                rows.append({'Ticker': tkr, 'Sektor': sektor, 'Model': 'Simple RNN',
                             'RMSE': round(rr,2), 'MAE': round(mr,2), 'MAPE (%)': round(mpr,2), 'R²': round(r2r,4)})
                rows.append({'Ticker': tkr, 'Sektor': sektor, 'Model': 'LSTM',
                             'RMSE': round(rl,2), 'MAE': round(ml,2), 'MAPE (%)': round(mpl,2), 'R²': round(r2l,4)})
            except Exception as e:
                st.warning(f"⚠️ {tkr}: {e}")
            progress.progress((idx+1) / len(SAHAM))

        progress.empty()

        df_comp = pd.DataFrame(rows)

        st.markdown("#### 📋 Tabel Perbandingan Lengkap")
        st.dataframe(df_comp, use_container_width=True)

        # Download tabel perbandingan
        csv_comp = df_comp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Tabel Perbandingan CSV",
            data=csv_comp,
            file_name="perbandingan_semua_saham.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("#### 📊 Grafik Perbandingan Metrik")

        metrics  = ['RMSE', 'MAE', 'MAPE (%)', 'R²']
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Perbandingan Metrik — Simple RNN vs LSTM (Semua Saham BEI)',
                     fontsize=13, fontweight='bold')

        for ax, metric in zip(axes.flatten(), metrics):
            tickers_list = list(SAHAM.keys())
            x     = np.arange(len(tickers_list))
            width = 0.35
            vals_rnn  = [float(df_comp[(df_comp['Ticker']==t)&(df_comp['Model']=='Simple RNN')][metric].values[0])
                         for t in tickers_list]
            vals_lstm = [float(df_comp[(df_comp['Ticker']==t)&(df_comp['Model']=='LSTM')][metric].values[0])
                         for t in tickers_list]
            b1 = ax.bar(x - width/2, vals_rnn,  width, label='Simple RNN', color='#1f77b4', alpha=0.85)
            b2 = ax.bar(x + width/2, vals_lstm, width, label='LSTM',       color='#ff7f0e', alpha=0.85)
            for bar, val in zip(list(b1)+list(b2), vals_rnn+vals_lstm):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)
            ax.set_title(f'Metrik: {metric}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(tickers_list, rotation=15, fontsize=9)
            ax.set_ylabel(metric)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tentukan model terbaik per saham
        st.markdown("---")
        st.markdown("#### 🏆 Model Terbaik per Saham (berdasarkan RMSE)")
        for tkr in SAHAM.keys():
            rnn_rmse  = df_comp[(df_comp['Ticker']==tkr)&(df_comp['Model']=='Simple RNN')]['RMSE'].values
            lstm_rmse = df_comp[(df_comp['Ticker']==tkr)&(df_comp['Model']=='LSTM')]['RMSE'].values
            if len(rnn_rmse) and len(lstm_rmse):
                best = "🧠 LSTM" if float(lstm_rmse[0]) < float(rnn_rmse[0]) else "🤖 Simple RNN"
                st.markdown(f"- **{tkr}** ({SAHAM[tkr].split('(')[0].strip()}): {best}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "📈 Prediksi Harga Saham BEI | Deep Learning: RNN & LSTM | "
    "Penulisan Ilmiah — Program Studi Informatika"
    "</div>",
    unsafe_allow_html=True
)