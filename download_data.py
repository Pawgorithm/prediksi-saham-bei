import yfinance as yf
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

SAHAM = {
    "BBCA.JK": "Bank Central Asia (Perbankan)",
    "TLKM.JK": "Telkom Indonesia (Telekomunikasi)",
    "ADRO.JK": "Adaro Energy (Energi/Tambang)",
    "GOTO.JK": "GoTo Gojek Tokopedia (Teknologi)",
    "BMRI.JK": "Bank Mandiri (Perbankan)",
}

print("Mendownload data saham BEI dari Yahoo Finance...")
print("=" * 65)

for ticker, nama in SAHAM.items():
    try:
        df = yf.download(
            ticker,
            start="2020-01-01",
            end="2026-05-07",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            print(f"❌ {ticker}: Data kosong, coba cek koneksi internet")
            continue

        # Rapikan kolom
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.index.name = "Date"
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Simpan ke CSV
        fname = f"dataset/{ticker.replace('.', '_')}.csv"
        df.to_csv(fname, index=False)

        print(f"✅ {ticker} ({nama})")
        print(f"   Jumlah data : {len(df):,} baris")
        print(f"   Periode     : {df['Date'].iloc[0]} s/d {df['Date'].iloc[-1]}")
        print(f"   Tersimpan   : {fname}")
        print()

    except Exception as e:
        print(f"❌ {ticker}: Error — {e}")

print("=" * 65)
print("Download selesai! Silakan jalankan notebook kembali.")