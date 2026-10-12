# PdfExtractor

Paket Python modular untuk mengekstrak PDF dengan MinerU 4.x dan membuat database literatur. Google Colab dapat menjalankan paket dari GitHub sebagai runner; PDF dan hasil ekstraksi tetap disimpan di Google Drive.

## Fitur

- Teks/heading/caption per blok, halaman, bounding box jika tersedia.
- Tabel dan figure/gambar beserta caption, halaman, bounding box dan referensi aset jika tersedia.
- Menyimpan keluaran mentah MinerU: `markdown.md`, `middle_json.json`, `structured_content.json`, dan folder aset.
- Metadata dokumen, SHA-256, jumlah halaman, status, serta audit kegagalan.
- SQLite, CSV, JSONL, dan inventory gambar.
- Resume hasil valid untuk menghindari ekstraksi ulang.

## Mode runtime

### CPU
Konfigurasi MinerU 4.x: small-model backend `onnx`, VLM `llama-cpp`, dan table models `cpu`. Instal paket dasar dari `requirements.txt`.

### GPU NVIDIA (Google Colab T4, bila runtime tersedia)
Konfigurasi: small-model backend `torch` dan table models `cuda`. Untuk tier `standard`/`advanced`, engine `vllm` dipakai secara default pada GPU NVIDIA, termasuk Tesla T4. Pilihan ini memerlukan wheel vLLM yang cocok dengan runtime Python/CUDA dan GPU. Pada Colab yang dilaporkan dalam issue, proses mati dengan SIGSEGV segera setelah memilih `llama-cpp-engine`, sehingga notebook tidak lagi memilih llama-cpp secara otomatis pada T4. Jangan menganggap penggantian engine sudah tervalidasi sampai satu PDF percobaan berhasil diekstrak.

Backend mode dipilih melalui CLI `--device cpu|gpu`; engine VLM dipilih melalui `--vlm-engine auto|llama-cpp|vllm`. Jangan mencampur cache model untuk backend berbeda: script menyiapkan dan memverifikasi model sesuai kombinasi yang dipilih.

## Struktur

```text
PdfExtractor/
├── requirements.txt
├── README.md
├── run_extraction.py
├── PdfExtractor/
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
```

## Instalasi

CPU:

```bash
python -m pip install -r requirements.txt
python run_extraction.py --input-dir ./pdf --output-dir ./literature_db --device cpu --tier standard --prepare-models
```

GPU NVIDIA/Linux (pastikan runtime CUDA aktif terlebih dahulu). Pilih salah satu jalur instalasi berikut pada environment bersih.

VLM dengan vLLM untuk throughput tinggi, bila wheel/runtime kompatibel:

```bash
python -m pip install -r requirements.txt "mineru[full]>=4.0,<5"
python run_extraction.py --input-dir ./pdf --output-dir ./literature_db --device gpu --tier standard --vlm-engine vllm --prepare-models
```

Jika vLLM gagal dipasang atau diimpor, jangan otomatis beralih ke `llama-cpp` untuk Standard: pada runtime Colab yang dilaporkan, engine itu mengalami SIGSEGV saat inisialisasi. Untuk menguji jalur GPU tanpa VLM terlebih dahulu, gunakan tier `basic`:

```bash
python -m pip install -r requirements.txt "mineru[torch]>=4.0,<5"
python run_extraction.py --input-dir ./pdf --output-dir ./literature_db --device gpu --tier basic --prepare-models
```

Tier `basic` bukan pengganti setara untuk Standard karena tidak menjalankan VLM.

Jika preflight vLLM menghasilkan `TypeError: Config() got an unexpected keyword argument 'deprecated'`, jangan lanjutkan ekstraksi. Error ini muncul saat mengimpor internal TorchDynamo, sebelum MinerU memproses PDF. Notebook yang diperbarui akan mencetak signature dan lokasi modul `Config`, serta perintah untuk memasang ulang wheel PyTorch yang sama dengan versi/CUDA saat ini. Jalankan perintah itu pada sel baru, restart runtime, lalu jalankan notebook dari awal. Ini memperbaiki kemungkinan file internal PyTorch yang tidak sinkron tanpa sengaja mengganti versi Torch; bila error tetap ada, kirim signature/lokasi modul dan versi `vllm` untuk menentukan pasangan versi yang benar.

Pilihan utama CLI:

- `--input-dir`: folder PDF sumber.
- `--output-dir`: folder output yang terpisah dari input.
- `--device`: `cpu` atau `gpu` (NVIDIA CUDA; bukan GPU AMD Windows).
- `--vlm-engine`: `auto`, `vllm`, atau `llama-cpp`; mode GPU dengan `llama-cpp` tetap menggunakan Torch/CUDA untuk small models dan CUDA untuk model tabel.
- `--tier`: `flash`, `basic`, `standard`, atau `advanced`.
- `--ocr-mode`: `auto`, `txt`, atau `ocr`.
- `--prepare-models`: unduh dan verifikasi model untuk backend yang dipilih.
- `--model-source`: `auto`, `huggingface`, atau `modelscope`.
- `--no-resume`: proses ulang hasil yang sudah valid.
- `--stop-on-error`: hentikan batch bila satu PDF gagal.
- Jika kode native crash saat parsing, worker terisolasi menulis `documents/<document_id>/_ERROR.json` (termasuk exit code/signal dan cuplikan log) serta `_WORKER.log`; jika proses utama mati sebelum worker, notebook menyimpan `OUTPUT_DIR/_RUN_ERROR.json`.

## Google Colab runner

Gunakan notebook `runner.ipynb`. Isi URL repo GitHub publik pada sel konfigurasi, pilih runtime GPU di Colab jika ingin `DEVICE_MODE="gpu"`, dan sesuaikan folder input/output Google Drive. Untuk private repo, autentikasi GitHub perlu disiapkan terpisah; jangan menanam token akses ke notebook publik.

## Output

```text
OUTPUT_DIR/
├── documents/<nama>__<id>/
│   ├── markdown.md
│   ├── middle_json.json
│   ├── structured_content.json
│   ├── images/
│   ├── _SUCCESS.json / _ERROR.json
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

## Catatan kualitas

Pipeline mempertahankan teks, HTML/JSON, caption, dan referensi yang disediakan MinerU; ia tidak menjamin OCR/tabel sempurna. Audit CSV dan cek sampel terhadap PDF asli sebelum memakai hasil dalam analisis ilmiah. Mode Standard/VLM dapat memerlukan memori dan GPU yang memadai. Jalankan satu PDF percobaan sebelum batch besar.

## Referensi resmi MinerU 4.x

- https://opendatalab.github.io/MinerU/usage/tiers/
- https://opendatalab.github.io/MinerU/quick_start/extension_modules/
- https://opendatalab.github.io/MinerU/usage/model_source/
