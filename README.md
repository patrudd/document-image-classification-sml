# Document Image Classification — Tugas 2 SML 2026

Klasifikasi 32.003 citra dokumen hasil scan ke 16 kelas menggunakan **PyTorch** dan transfer learning **ConvNeXt-Tiny**. Metrik kompetisi adalah multiclass accuracy. Implementasi memisahkan data train/validation secara stratified, menyimpan checkpoint terbaik berdasarkan validation accuracy, lalu menghasilkan submission `id,label` untuk 7.994 citra test.

## Ringkasan pendekatan

- **Baseline utama:** ConvNeXt-Tiny pretrained ImageNet dari `torchvision`.
- **Input:** citra TIFF dikonversi ke RGB, diputar mengikuti EXIF, diberi padding putih hingga persegi, lalu di-resize 224×224. Padding dipilih agar header, tabel, dan tata letak halaman tidak terpotong.
- **Validasi:** 15% stratified holdout dari `train.csv`; test kompetisi tidak pernah digunakan untuk memilih model.
- **Augmentasi:** rotasi kecil, autocontrast, serta jitter brightness/contrast. Horizontal flip tidak dipakai karena dapat merusak orientasi tulisan.
- **Optimasi:** AdamW, differential learning rate, cosine annealing, class-weighted cross entropy, label smoothing, mixed precision, dan early stopping.
- **Pemilihan model:** checkpoint dengan validation accuracy terbaik.

## Struktur repository

```text
document-image-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                         # tidak di-commit
├── checkpoints/                  # tidak di-commit
├── notebooks/
│   ├── 01_end_to_end_kaggle.py
│   └── 01_end_to_end_kaggle.ipynb
├── scripts/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── train.py
│   └── predict.py
├── reports/
│   └── figures/
└── submissions/
```

## Data

Kompetisi: `document-image-classification-sd234504` (private competition).

Di Kaggle Notebook, gunakan **Add Input → Competition Data**. Kode otomatis mencari:

1. `/kaggle/input/document-image-classification-sd234504`
2. `/kaggle/input/competitions/document-image-classification-sd234504`
3. `data/` untuk eksekusi lokal

Untuk lokal, letakkan `train.csv`, `test.csv`, `sample_submission.csv`, folder `train/`, dan folder `test/` di `data/`. Dataset dan checkpoint sengaja diabaikan Git agar mematuhi aturan tugas.

## Menjalankan secara lokal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train --data-dir data --epochs 8 --batch-size 32
python -m scripts.predict --data-dir data --checkpoint checkpoints/convnext_tiny_best.pt
```

Output penting:

- `checkpoints/convnext_tiny_best.pt`: bobot model terbaik + konfigurasi + riwayat training.
- `reports/validation_metrics.json`: accuracy, classification report, confusion matrix, dan learning history.
- `submissions/submission_convnext.csv`: file yang diunggah ke Kaggle.

## Menjalankan di Kaggle

1. Buka tab **Code** pada kompetisi dan buat notebook baru.
2. Aktifkan **GPU** dan **Internet**. Internet diperlukan sekali untuk bobot pretrained ImageNet.
3. Tambahkan competition data sebagai input.
4. Upload atau import `notebooks/01_end_to_end_kaggle.ipynb`.
5. Jalankan semua cell dan pastikan validation accuracy serta file submission muncul.
6. Unduh checkpoint dan submission dari `/kaggle/working/`.

Jika memori GPU tidak cukup, turunkan `batch_size` dari 32 menjadi 16. Jangan mengubah validation split ketika membandingkan eksperimen agar skor tetap adil.

## Strategi submission (batas 5/hari)

Gunakan submission secara hemat:

1. ConvNeXt-Tiny baseline yang lolos validasi lokal.
2. Perbaikan resolusi 288 jika GPU dan waktu cukup.
3. Ensemble hanya setelah dua model benar-benar memiliki error validasi yang berbeda.

Catat validation accuracy, konfigurasi, public score, dan nama file setiap submission. Public leaderboard hanya berbobot 10%; kualitas repository dan interpretasi berbobot 70%, sehingga eksperimen harus reproducible dan terdokumentasi.

## Interpretasi yang wajib dilaporkan setelah training

- Bandingkan train vs validation accuracy/loss untuk mendeteksi overfitting.
- Gunakan confusion matrix dan per-class recall untuk pasangan kelas yang mirip, misalnya `form`–`questionnaire`, `scientific_publication`–`scientific_report`, serta `letter`–`memo`.
- Jelaskan bahwa class weighting disiapkan untuk ketidakseimbangan, tetapi keputusan akhirnya harus berdasarkan distribusi aktual `train.csv`.
- Jangan menyatakan public/private score sebelum submission benar-benar dinilai Kaggle.

## Reproducibility dan batasan

Seed default adalah 42. PyTorch/CUDA tetap dapat memiliki sedikit variasi numerik. Bobot pretrained berasal dari ImageNet, bukan dari test set kompetisi. Hasil leaderboard belum dicantumkan karena belum ada training dan submission yang selesai dijalankan pada data kompetisi.

