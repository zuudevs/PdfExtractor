# MinerU Literature Extractor — Modular CPU Version

Paket modular Python untuk mengekstrak PDF menggunakan MinerU 4.x dan menyusun hasilnya menjadi database literatur. Paket ini dibuat dari notebook `PdfExtractor.ipynb` yang diberikan, dengan backend inferensi CPU yang diatur secara eksplisit.

## Apa yang diekstrak

- **Teks dan blok**: teks per halaman/blok, judul, heading, caption, formula bila dikenali, bounding box jika disediakan MinerU, serta JSON mentah.
- **Tabel**: caption, teks/HTML bila tersedia, halaman, bounding box, referensi aset gambar, dan JSON sumber.
- **Figure/gambar/chart**: tipe, caption/teks terkait, halaman, bounding box, referensi aset, dan JSON sumber.
- **Metadata dokumen**: SHA-256, metadata PDF, jumlah halaman, kandidat judul dan DOI, status pemrosesan, serta peringatan audit.
- **Database**: SQLite dengan tabel teks, elemen, tabel, figure, aset, dan full-text search FTS5 jika tersedia.
- **Ekspor**: CSV dan JSONL agar mudah dipakai untuk analisis atau migrasi.

## CPU mode yang digunakan

Paket ini sengaja **tidak memasang `mineru[full]`**. `requirements.txt` memasang paket dasar MinerU 4.x dan `pypdf`, lalu kode menetapkan konfigurasi sebelum MinerU diimpor:

- `MINERU_MODEL_SMALL_BACKEND=onnx`
- `MINERU_MODEL_VLM_ENGINE=llama-cpp`
- `MINERU_TABLE_DEVICE=cpu`
- `CUDA_VISIBLE_DEVICES=-1`

Kombinasi ONNX + llama.cpp adalah konfigurasi CPU yang didokumentasikan MinerU 4.x untuk `standard` / `advanced`. CPU biasanya lebih lambat daripada GPU. Untuk `standard`, sediakan RAM yang memadai; dokumentasi MinerU menyebut sekitar 8 GB sebagai titik awal untuk konfigurasi CPU dan merekomendasikan 16 GB atau lebih untuk beban lebih berat. Kebutuhan sebenarnya bergantung pada PDF dan proses lain yang berjalan.

## Struktur paket

```text
PdfExtractor/
├── requirements.txt
├── README.md
├── run_extraction.py
├── PdfExtractor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   ├── model_setup.py
│   ├── normalizer.py
│   ├── pipeline.py
│   ├── runner.py
│   ├── scanner.py
│   └── utils.py
└── tests/
    └── test_normalizer.py
```

## Instalasi lokal (Linux/macOS/Windows WSL)

Disarankan menggunakan Python 3.12 dengan virtual environment.

```bash
python -m venv .venv
```

Aktifkan environment, kemudian:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MinerU mendukung Python 3.10–3.14, tetapi dukungan engine dan paket turunannya dapat berbeda antarsistem operasi. Jalankan dari environment yang sama untuk instalasi dan ekstraksi.

## Google Colab + Google Drive

1. Unggah dan ekstrak ZIP paket ini ke `/content/PdfExtractor` atau salin folder paket ke runtime Colab.
2. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Instal dependensi dari folder paket:

   ```python
   %cd /content/PdfExtractor
   %pip install -r requirements.txt
   ```

   Jika Colab meminta restart session setelah instalasi, lakukan restart lalu mount Drive kembali.

4. Jalankan ekstraksi dari sel notebook dengan perintah shell berikut. Ganti lokasi input dan output sesuai folder Drive Anda:

   ```bash
   !python run_extraction.py \
     --input-dir "/content/drive/MyDrive/Paper_PDF" \
     --output-dir "/content/drive/MyDrive/literature_mineru_db" \
     --tier standard \
     --prepare-models
   ```

`--prepare-models` mengunduh dan memverifikasi model untuk backend CPU yang sama sebelum PDF diproses. Jalankan pada penggunaan pertama atau ketika model belum ada di cache. Model akan disimpan pada lokasi cache MinerU, bukan di folder PDF sumber. Model memerlukan ruang disk dan akses internet ketika pertama kali diunduh.

Jika model CPU yang sesuai sudah diunduh dan diverifikasi, penggunaan berikutnya dapat melewati unduh/validasi model:

```bash
!python run_extraction.py \
  --input-dir "/content/drive/MyDrive/Paper_PDF" \
  --output-dir "/content/drive/MyDrive/literature_mineru_db" \
  --tier standard
```

> Catatan: ketika proses dijalankan dari Colab, file `.py` dijalankan sebagai proses Python tersendiri. Pengaturan backend CPU diterapkan di dalam proses tersebut sebelum modul MinerU diimpor.

## Opsi CLI

```bash
python run_extraction.py --help
```

Opsi yang umum:

- `--input-dir`: folder input yang berisi PDF.
- `--output-dir`: folder untuk hasil ekstraksi dan database; gunakan folder terpisah dari input.
- `--tier`: `flash`, `basic`, `standard`, atau `advanced`; default `standard`.
- `--ocr-mode`: `auto`, `txt`, atau `ocr`; default `auto`.
- `--non-recursive`: hanya scan folder input, tanpa subfolder.
- `--no-resume`: proses ulang semua PDF, termasuk hasil valid yang sebelumnya tersimpan.
- `--stop-on-error`: hentikan batch jika sebuah PDF gagal; defaultnya lanjut ke PDF berikutnya.
- `--model-source`: `auto`, `huggingface`, atau `modelscope`.
- `--prepare-models`: unduh dan verifikasi model untuk tier/backend CPU yang dipilih.

## Hasil output

```text
literature_mineru_db/
├── documents/
│   └── <nama_dokumen>__<id>/
│       ├── markdown.md
│       ├── middle_json.json
│       ├── structured_content.json
│       ├── images/
│       ├── _SUCCESS.json atau _ERROR.json
└── database/
    ├── literature_database.sqlite
    ├── literature_documents.csv
    ├── literature_tables.csv
    ├── literature_figures.csv
    ├── literature_text_blocks.csv
    ├── literature_elements.jsonl
    ├── literature_fulltext.jsonl
    ├── literature_extraction_audit.csv
    └── literature_image_inventory.csv
```

Nomor `page_idx` dari JSON dipertahankan sebagai zero-based dan `page_number` diekspor sebagai one-based. Teks/caption/HTML mengikuti informasi yang disediakan MinerU; pipeline tidak menjamin OCR atau deteksi tabel sempurna. Periksa `literature_extraction_audit.csv` dan bandingkan beberapa halaman dengan PDF sumber sebelum memakai hasil untuk analisis ilmiah.

## Troubleshooting

- **Kernel / proses restart atau berhenti mendadak**: mode CPU menghindari penggunaan CUDA, tetapi `standard` masih intensif dan dapat memerlukan banyak RAM. Coba satu PDF dulu dan periksa RAM tersedia.
- **Download model gagal**: coba `--model-source modelscope` jika sumber default/Hugging Face tidak dapat diakses di jaringan Anda.
- **`mineru-kit` tidak ditemukan saat `--prepare-models`**: pastikan `requirements.txt` dipasang pada environment Python yang sama dengan perintah `python run_extraction.py`.
- **PDF gagal diproses**: periksa file `_ERROR.json` di folder dokumen dan kolom `error_message`/`warnings_json` pada audit CSV.
- **Aset gambar hilang**: cek referensi yang disimpan di `middle_json.json` dan `literature_image_inventory.csv`; beberapa gambar dapat tertanam dalam struktur JSON alih-alih menjadi file terpisah.

## Referensi resmi

- MinerU Quick Start: https://opendatalab.github.io/MinerU/quick_start/
- MinerU Tiers and Runtimes: https://opendatalab.github.io/MinerU/usage/tiers/
- MinerU Model Downloads and Configuration: https://github.com/opendatalab/MinerU/blob/master/docs/en/usage/model_source.md
- MinerU Python SDK: https://opendatalab.github.io/MinerU/usage/sdk_api/
