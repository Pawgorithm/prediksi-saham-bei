# Prediksi Harga Pembukaan Saham BEI dengan RNN dan LSTM

## Deskripsi
Proyek Penulisan Ilmiah — Prediksi harga pembukaan saham (open price) 
di Bursa Efek Indonesia menggunakan Deep Learning (RNN & LSTM).

## Dataset
| Ticker   | Emiten                        | Sektor           | Periode           |
|----------|-------------------------------|------------------|-------------------|
| BBCA.JK  | Bank Central Asia             | Perbankan        | Jan 2020 – Mei 2026 |
| TLKM.JK  | Telkom Indonesia              | Telekomunikasi   | Jan 2020 – Mei 2026 |
| ADRO.JK  | Adaro Energy                  | Energi & Tambang | Jan 2020 – Mei 2026 |
| GOTO.JK  | GoTo Gojek Tokopedia          | Teknologi        | Jan 2020 – Mei 2026 |
| BMRI.JK  | Bank Mandiri                  | Perbankan        | Jan 2020 – Mei 2026 |

## Struktur Folder
```
project/
├── Prediksi_Saham_BEI_RNN_LSTM.ipynb   ← Notebook utama
├── requirements.txt
├── README.md
└── dataset/
    ├── BBCA_JK.csv
    ├── TLKM_JK.csv
    ├── ADRO_JK.csv
    ├── GOTO_JK.csv
    └── BMRI_JK.csv
```

## Cara Menjalankan
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Jalankan Jupyter Notebook:
   ```
   jupyter notebook Prediksi_Saham_BEI_RNN_LSTM.ipynb
   ```

## Model
- **Simple RNN**: 4 layer (64-64-32-32 unit) + Dropout 0.2
- **LSTM**: 2 layer (64-64 unit) + Dense 32 + Dropout 0.2

## Metrik Evaluasi
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)  
- MAPE (Mean Absolute Percentage Error)
- R² (Koefisien Determinasi)
