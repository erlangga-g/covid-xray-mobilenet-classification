# Klasifikasi Citra X-Ray Dada (COVID-19 vs Normal) dengan MobileNet

Proyek kerja kelompok **Kelompok 7** untuk mengklasifikasikan citra rontgen (*X-ray*) dada ke dalam dua kelas (**COVID-19** dan **Normal**) menggunakan *transfer learning* arsitektur **MobileNet** berbasis **TensorFlow / Keras**.

## 👥 Anggota Kelompok 7
- **Muhammad Erlangga Gunawan** (`I.2210161`)
- **Gerald Dwight Andgha C** (`I.2210326`)
- **Riko Wahyudi** (`I.2210473`)
- **Muhamad Taufiqurrahman** (`I.2211117`)

## 📂 Dataset & Referensi
- **Dataset Normal:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Dataset COVID-19:** [COVID-19 Image Data Collection — GitHub](https://github.com/ieee8023/covid-chestxray-dataset)
- **Referensi:** [luchonaveiro/covid-19-xray](https://github.com/luchonaveiro/covid-19-xray.git)

## 🧪 Ringkasan Pipeline & Hasil
- **Ukuran & Preprocessing Citra:** Resize ke `128x128` (3 *channel* RGB), normalisasi piksel (`1./255`), serta augmentasi pada data latih (`rotation_range=40`, `zoom_range=0.1`) dengan pembagian validasi `20%` dari folder train.
- **Pembagian Data:**
  - `training_set`: `1.544` gambar
  - `validation_set`: `386` gambar
  - `test_set`: `327` gambar (`0`: `covid`, `1`: `normal`)
- **Arsitektur Model:** *Base model* `MobileNet` (`weights='imagenet'`, `include_top=False`, `trainable=False`) + `Flatten` + `Dense(256, activation='relu')` + `Dropout(0.5)` + `Dense(1, activation='sigmoid')`.
- **Pelatihan & Evaluasi:** Dilatih selama `10` *epoch* (`batch_size=32`, optimizer `Adam`, loss `binary_crossentropy`). Pada data uji (`327` citra), model mencapai akurasi sekitar **99%** (`Test akurasi ~0.9906` pada evaluasi batch, dengan *classification report* `0.99` di seluruh metrik rata-rata).

## 📝 Catatan Penggunaan
- Kode notebook dirancang untuk dijalankan di lingkungan Python/TensorFlow (seperti Google Colab atau Jupyter/editor lokal).
- Bagian pengujian interaktif menggunakan `tkinter.filedialog` lebih sesuai dijalankan di lingkungan lokal yang mendukung GUI, bukan di *notebook* cloud *headless*.
- Proyek ini merupakan eksperimen akademik/pembelajaran dan **bukan** alat diagnosis medis klinis.

## 📄 Lisensi
Proyek ini menggunakan lisensi **MIT** (lihat file `LICENSE`).
