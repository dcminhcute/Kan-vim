# Hướng dẫn chạy TOÀN BỘ thí nghiệm của đồ án

Chỉ cần chạy **một lệnh** (máy thuê) hoặc bấm **một nút** (Kaggle). Script `scripts/run_plan.py` tự làm hết các bước sau, **theo thứ tự ưu tiên**:

| Bước | Nội dung | Số lần train |
|---|---|---|
| 0 | Đo tốc độ GPU, **tự chọn số ảnh train / số epoch** để toàn bộ kế hoạch vừa `--budget_hours` | - |
| 1 | Dò learning rate cho baseline (2e-5, 5e-5, 1e-4), chọn theo validation | 3 (ngắn) |
| 2 | Baseline Vim-Ti (100% token) trên CIFAR-100, lưu làm teacher | 1 |
| 3 | 6 phương pháp (KANPool, MLP Pool, Avg Pool, Max Pool, KAN Prune, MLP Prune) giữ 50% | 6 |
| 4 | MTR (training-free, không train) giữ 75/50/25% | 0 (chỉ đánh giá) |
| 5 | 6 phương pháp giữ 75% và 25% | 12 |
| 6 | Baseline + 6 phương pháp (50%) với seed 1 và 2 → trung bình ± độ lệch chuẩn | 14 |
| 7 | Đối chứng Static drop, Random drop + giảm token nhiều tầng (KAN/MLP Prune 3 tầng) | 4 |
| 8 | Dataset thứ hai **Tiny-ImageNet**: baseline + 6 phương pháp (50%) + MTR | 7 |
| 9 | Ablation KAN (KANPool, KAN Prune): grid 3/8, hidden 16/64, bỏ L1, bỏ distillation | 12 |
| 10 | Báo cáo `BAO_CAO.md` + `ket_qua.csv` + hình | - |

Tổng cộng khoảng **59 lần train**. Mỗi lần train theo quy trình chuẩn: tách **validation** cố định từ tập train → chọn model theo val, **early stopping** → **test 1 lần duy nhất** ở cuối. Chống overfitting: augmentation, label smoothing, weight decay, mixup, gradient clipping.

**Bị ngắt / hết giờ:** chạy lại đúng lệnh cũ. Phần đã xong được giữ, lần train đang dở chạy tiếp từ epoch cuối. Bước nào chưa kịp thì báo cáo vẫn được tạo với những gì đã có.

> **Về thời gian:** 59 lần train là khối lượng lớn. Script tự thu nhỏ số ảnh/epoch để vừa `--budget_hours`, nhưng nếu ngân sách quá nhỏ thì số ảnh train sẽ rất ít (kết quả kém tin cậy). Thực tế:
> - Máy thuê **2 GPU** (ví dụ 2×RTX 4090) chạy song song 2 lần train một lúc → nhanh gấp đôi.
> - Kaggle (miễn phí, T4, bản chậm): chỉ đủ cho các bước đầu trong vài ngày; các bước sau chạy tiếp ở các phiên sau.

---

## Cách A: Thuê GPU (khuyên dùng)

Ví dụ với **RunPod** (vast.ai làm tương tự).

1. Vào [runpod.io](https://www.runpod.io), đăng ký, nạp **25–40 USD** (thẻ Visa/Mastercard).
2. **Pods → Deploy**:
   - GPU: **2 × RTX 4090** (khoảng 0,8–1,4 USD/giờ cho 2 GPU). Hoặc 1 × A100 / H100.
   - Template: **RunPod Pytorch 2.1** (image `runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04`). Đúng bản 2.1 + CUDA 11.8 thì cài được kernel CUDA chính thức của Vim (nhanh hơn nhiều).
   - Container disk 40 GB, Volume 30 GB → **Deploy**.
3. Khi pod chạy: **Connect → Jupyter Lab**. Kéo thả file `D:\đồ án\kan-vim-v3.zip` vào `/workspace`.
4. Mở **Terminal** trong Jupyter Lab, dán:
   ```bash
   cd /workspace && unzip -o kan-vim-v3.zip && cd kan-vim
   bash scripts/setup_rent.sh
   bash scripts/run_rent.sh --budget_hours 30
   ```
   `--budget_hours 30` = "tôi muốn toàn bộ xong trong khoảng 30 giờ". Script tự chọn quy mô. Muốn xong sớm hơn thì giảm số này (đổi lại ít ảnh train hơn).
5. Xem tiến độ (đóng trình duyệt vẫn chạy):
   ```bash
   tail -f /workspace/kan-vim/run_plan.log
   ```
   Sau Bước 0, log in `KẾ HOẠCH: ...` và thời gian ước tính của từng bước.
6. Xem kết quả tạm thời bất cứ lúc nào:
   ```bash
   cd /workspace/kan-vim && python scripts/report_plan.py results_plan
   ```
   rồi mở `results_plan/BAO_CAO.md` trong Jupyter Lab.
7. Khi log in `XONG TOÀN BỘ`: tải `/workspace/kan-vim/results_plan.zip` về máy (chuột phải → Download).
8. **Stop + Terminate pod** để không bị trừ tiền tiếp.

---

## Cách B: Kaggle (miễn phí, chậm)

1. Tài khoản Kaggle → **Settings → Phone verification** (bắt buộc để dùng GPU + Internet).
2. **Datasets → New Dataset** → upload `D:\đồ án\kan-vim-v3.zip`, đặt tên `kan-vim` → **Create**.
3. **Code → New Notebook → File → Import Notebook** → chọn `D:\đồ án\kan-vim\notebooks\kaggle_plan.ipynb`.
4. Thanh bên phải: **Accelerator → GPU T4 x2**, **Internet → On**, **Add Input** → dataset `kan-vim` của bạn.
5. **Save Version → Save & Run All (Commit) → Save**. Kaggle chạy nền, tắt máy cũng được.
6. Sau khoảng 11 giờ notebook tự dừng an toàn. Mở log của version:
   - `XONG TOÀN BỘ` → tab **Output** → tải `results_plan.zip`.
   - `ĐÃ TẠM DỪNG` → **chạy tiếp**:
     1. Mở lại notebook (**Edit**).
     2. **Add Input → Your Work → Notebooks** → chọn chính notebook này (version mới nhất) → Add. (Từ lần thứ 3: ở mục Input, đổi sang version mới nhất.)
     3. **Save Version → Save & Run All**. Cell 1 in `khôi phục kết quả phiên trước`.

Hạn mức Kaggle: 30 giờ GPU/tuần. Mỗi phiên vẫn tạo `results_plan.zip` với những gì đã xong.

---

## Chỉ chạy một phần

```bash
python scripts/run_plan.py --profile gpu --phases 1-4      # chỉ bước 1-4 (cốt lõi + MTR)
python scripts/run_plan.py --profile gpu --phases 8        # chỉ Tiny-ImageNet
python scripts/run_plan.py --profile gpu --phases 1-3,5    # bỏ qua MTR, seed, ...
python scripts/run_plan.py --profile gpu --train_subset 20000 --epochs 30   # ép cấu hình
python scripts/run_plan.py --profile gpu --replan          # đo và chọn lại cấu hình (xoá plan.json)
```
Lưu ý: cấu hình (số ảnh, epoch) được chọn **một lần** ở lần chạy đầu và ghi vào `results_plan/plan.json` để mọi lần train dùng cùng cấu hình (so sánh công bằng). Lần chạy đầu nên để `--phases` mặc định (1-9) để ngân sách tính cho toàn bộ kế hoạch.

## Kiểm tra trên máy mình (không cần GPU)

```bash
cd "D:\đồ án\kan-vim"
python scripts/run_plan.py --profile cpu-test --out results_cpu_test
```
Chạy cả 9 bước với dữ liệu giả + backbone Vim rất nhỏ (khoảng 30–60 phút), chỉ để chắc chắn mọi thứ chạy hết và sinh `BAO_CAO.md`. Con số **không có ý nghĩa**.

---

## Kết quả (`results_plan.zip`)

| File | Nội dung |
|---|---|
| `BAO_CAO.md` | 11 mục: thiết lập, dò LR, kết quả chính, độ ổn định qua seed, các mức giữ, đối chứng + multi-stage + MTR, Tiny-ImageNet, ablation KAN, kiểm tra overfitting, chi phí tính toán, **so sánh rút gọn KAN vs MLP / học vs cố định / gộp vs loại bỏ / có train vs MTR** |
| `ket_qua.csv` | Mọi lần chạy, mở bằng Excel |
| `fig_acc_vs_gmacs.png` | Đánh đổi độ chính xác – GMACs ở 3 mức giữ |
| `fig_curves.png`, `fig_curves_d2.png` | Train loss, val loss, val top-1 theo epoch (nhận xét overfitting) |
| `fig_test_top1.png`, `fig_lr_search.png` | So sánh các phương pháp; dò LR |
| `plan.json`, `logs/*.log` | Cấu hình đã chọn; log đầy đủ từng lần train |

Gửi lại file zip này để viết phần kết quả, thảo luận và hướng phát triển.
