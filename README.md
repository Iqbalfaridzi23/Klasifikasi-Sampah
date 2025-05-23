# Laporan Proyek Machine Learning - Iqbal Alfaridzi Balman
## Domain Proyek

Permasalahan pengelolaan sampah menjadi salah satu tantangan besar di kota-kota metropolitan dan kawasan padat penduduk. Menurut data dari Kementerian Lingkungan Hidup dan Kehutanan Indonesia (KLHK), volume sampah nasional mencapai lebih dari 60 juta ton per tahun, namun hanya sebagian kecil yang berhasil dipilah dan didaur ulang dengan benar. Salah satu hambatan utama adalah proses pemilahan yang masih dilakukan secara manual, memakan waktu, dan rentan terhadap kesalahan manusia.

Pemanfaatan teknologi artificial intelligence, khususnya deep learning, memberikan peluang besar untuk mengotomatisasi proses ini. Dengan menggunakan model klasifikasi berbasis citra, sistem dapat secara otomatis mengenali jenis sampah hanya dari gambar, sehingga proses pemilahan bisa lebih cepat, akurat, dan efisien. Hal ini tidak hanya membantu rumah tangga dan institusi dalam mengelola limbah, tetapi juga mendukung pengurangan sampah yang berakhir di Tempat Pembuangan Akhir (TPA).

Referensi :
-Kementerian Lingkungan Hidup dan Kehutanan (KLHK). "Data Sampah Nasional 2022." https://sipsn.menlhk.go.id/sipsn/
-Dataset: Kaggle - Waste Classification Dataset

## Business Understanding

### Problem Statements

- Kurangnya sistem pemilahan sampah otomatis menyebabkan rendahnya efektivitas pengolahan limbah.
- Dibutuhkan model klasifikasi berbasis gambar yang akurat untuk mengenali jenis sampah (organik atau daur ulang).

### Goals

- Mengembangkan sistem klasifikasi citra berbasis CNN untuk mengenali jenis sampah.
- Mencapai akurasi minimal 85% untuk menjamin keandalan sistem dalam implementasi nyata.

### Solution statements
- Mengimplementasikan Convolutional Neural Network (CNN) dengan preprocessing berupa augmentasi dan normalisasi gambar.
- Menggunakan teknik regularisasi seperti dropout dan early stopping untuk mencegah overfitting.

## Data Understanding
Dataset yang digunakan berasal dari Kaggle - Waste Classification Dataset(https://www.kaggle.com/datasets/techsash/waste-classification-data) dan terdiri dari total lebih dari 22.000 gambar yang dikategorikan ke dalam dua kelas utama:
- organic : sampah dapur, sisa makanan, daun, dll.
- recyclable : plastik, kertas, logam, dan bahan daur ulang lainnya.

## Data Preparation
Pada tahap ini dilakukan beberapa proses persiapan data sebelum dimasukkan ke dalam model:
- Resize Gambar: Semua gambar diubah ukurannya menjadi 128x128 piksel agar memiliki dimensi seragam.
- Rescaling: Seluruh piksel gambar dinormalisasi ke skala 0-1 menggunakan rescale=1./255 untuk mempercepat proses pembelajaran dan stabilisasi pelatihan.
- ImageDataGenerator: Digunakan ImageDataGenerator dari Keras untuk membaca data dari direktori dan membuat batch image-label.
- class_mode='binary': Karena ini adalah klasifikasi biner (organik vs daur ulang), digunakan mode biner.
- Visualisasi Gambar: Ditampilkan 9 gambar acak dari batch pelatihan untuk menunjukkan representasi kelas.

## Modeling
Model yang digunakan adalah Convolutional Neural Network (CNN) dengan arsitektur sebagai berikut:
- Conv2D(32, 3x3, relu) + MaxPooling2D(2x2)
- Conv2D(64, 3x3, relu) + MaxPooling2D(2x2)
- Conv2D(128, 3x3, relu) + MaxPooling2D(2x2)
- Flatten
- Dense(128, relu)
- Dropout(0.5) untuk mencegah overfitting
- Dense(1, sigmoid) karena tugas klasifikasi biner

Parameter pelatihan:
- Optimizer: Adam
- Loss function: binary_crossentropy
- Metrics: accuracy
- Epochs: 5

Model dilatih menggunakan data training dan divalidasi dengan data testing.

## Evaluation
Model klasifikasi menunjukkan **akurasi total sebesar 91%**, yang menunjukkan performa cukup baik dalam mengklasifikasikan data ke dalam dua kelas: `0` dan `R`.
#### 🔍 Detail Evaluasi per Kelas:
- **Kelas 0**:
  - **Recall tinggi (0.97)** → Artinya model sangat baik dalam mengenali sampel kelas 0.
  - **Precision lebih rendah (0.88)** → Namun masih terdapat cukup banyak prediksi kelas 0 yang salah.

- **Kelas R**:
  - **Precision tinggi (0.96)** → Model sangat yakin ketika memprediksi kelas ini.
  - **Recall rendah (0.82)** → Masih banyak sampel kelas R yang gagal dikenali.

#### 📈 Analisis Faktor yang Mempengaruhi Hasil:
1. **Distribusi kelas tidak seimbang (class imbalance)** dapat menyebabkan model bias ke kelas mayoritas.
2. **Kualitas data latih**, termasuk pencampuran gambar atau label yang ambigu, bisa mengganggu pembelajaran model.
3. **Arsitektur model dan parameter pelatihan** (misalnya epoch atau learning rate) juga bisa memengaruhi kemampuan generalisasi.
4. **Threshold 0.5** mungkin belum optimal — bisa dicoba tuning untuk meningkatkan trade-off precision dan recall.
5. **Augmentasi data** yang kurang tepat dapat mempengaruhi kemampuan model dalam menghadapi data baru.

![Screenshot 2025-05-08 152324](https://github.com/user-attachments/assets/ae5e37d4-551d-44ce-b2f2-84bf99dd92b7)



