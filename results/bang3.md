**Bảng 3. Kết quả thí nghiệm**

| Cách chia   | RMSE (trung bình ± SD)   | R² (trung bình ± SD)   | Độ tương đồng trung vị test–train   | Chênh lệch RMSE so với random   |
|:------------|:-------------------------|:-----------------------|:------------------------------------|:--------------------------------|
| Random      | 0.687 ± 0.019            | 0.726 ± 0.013          | 0.82 ± 0.00                         | —                               |
| Scaffold    | 0.794 ± 0.046            | 0.628 ± 0.028          | 0.74 ± 0.01                         | +0.107 (+16%)                   |
| Time        | 1.398 ± 0.003            | -0.294 ± 0.006         | 0.51 ± 0.00                         | +0.711 (+103%)                  |

- Dữ liệu: ChEMBL CHEMBL203 (EGFR), IC50 '=' nM → pIC50; 26600 bản ghi thô → 10689 hợp chất duy nhất (gộp trung vị).
- Random/Scaffold dùng toàn bộ 10689 hợp chất; Time split bỏ 39 hợp chất thiếu document_year, train < 2022 ≤ test (năm 1989–2025).
- ECFP4 (r=2, 2048 bit); RandomForestRegressor {'n_estimators': 500, 'n_jobs': -1}, giống nhau cho mọi cách chia; 5 seed [0, 1, 2, 3, 4].
- Time split là tất định: SD của Time chỉ phản ánh seed của random forest, không phản ánh biến thiên của cách chia.
- Chênh lệch RMSE = RMSE trung bình (cách chia) − RMSE trung bình (Random).
- RMSE theo bin 0.7-1.0: Random 0.64, Scaffold 0.68, Time 1.16 — ở Time, sai số cao ngay cả khi test rất giống train, nên gap của Time không chỉ do độ tương đồng thấp (xem results/rmse_by_similarity_bin.csv).
- Lưu ý (gây nhiễu cho Time split): CHEMBL203 gộp cả assay EGFR đột biến; tỉ lệ bản ghi đột biến tăng từ 2% (≤2010) lên 50% (2022+), nên tập test của Time một phần là endpoint khác (xem src/diagnose_time.py).
