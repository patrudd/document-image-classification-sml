# Model artifact

Model terbaik adalah ConvNeXt-Tiny hasil fine-tuning selama 8 epoch. Checkpoint
asli bernama `convnext_tiny_best.pt` dan berukuran 111.399.911 byte (sekitar
106 MiB), sehingga tidak disimpan langsung dalam riwayat Git karena melewati
batas file 100 MiB GitHub.

Checkpoint tersedia sebagai aset pada
[GitHub Release v1.0-kaggle](https://github.com/patrudd/document-image-classification-sml/releases/tag/v1.0-kaggle).
Setelah diunduh, letakkan file di `models/convnext_tiny_best.pt`, lalu jalankan:

```bash
python -m scripts.predict \
  --data-dir data \
  --checkpoint models/convnext_tiny_best.pt
```

Integritas file dapat diperiksa dengan SHA-256 yang tercatat dalam
`model_metadata.json`.
