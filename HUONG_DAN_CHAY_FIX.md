# Chạy lại nhánh Prune đã sửa

Bản này sửa lỗi gradient của KAN Prune / MLP Prune (token bị loại không nhận gradient, nhiễu Gumbel làm việc chọn token khi train gần như ngẫu nhiên). Chỉ cần chạy lại phần Prune trên CIFAR-100; teacher (baseline đã fine-tune) có sẵn trong `teacher/`, không phải train lại.

**14 lần train, khoảng 3,5–4 giờ trên 2 × RTX 4090.**

## Thuê máy
Giống lần trước: vast.ai, **2 × RTX 4090**, template image `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`, disk 50 GB. Không chọn RTX 5090 / RTX PRO (Blackwell).

## Chạy
Upload `kan-vim-v4.zip` lên `/workspace`, rồi:
```bash
cd /workspace && python -m zipfile -e kan-vim-v4.zip . && cd kan-vim    # image không có sẵn unzip
bash scripts/setup_rent.sh          # 10–30 phút, cài kernel Vim chính thức (và pytest)
python -m pytest -q                 # phải thấy "49 passed"
bash scripts/run_fix.sh             # chạy nền
tail -f run_fix.log                 # xem tiến độ (Ctrl+C để thoát xem, không dừng chương trình)
```
Sau vài phút, log phải có dòng `backbone implementation: official`. Nếu là `pure` thì vẫn chạy được nhưng chậm hơn khoảng 25%.

Máy khởi động lại hoặc mất kết nối: chạy lại `bash scripts/run_fix.sh`, phần đã xong được giữ, job dở chạy tiếp.

## Thứ tự chạy
| Nhóm | Nội dung | Số job |
|---|---|---|
| A | KAN Prune, MLP Prune, giữ 50%, seed 0/1/2 | 6 |
| B | Kiểm tra learning rate: lr bộ chấm điểm 1e-4; lr backbone 5e-5 | 4 |
| C | KAN Prune, MLP Prune, giữ 75% và 25% | 4 |

## Kết quả
Bảng so sánh bản mới / bản cũ cập nhật sau mỗi job: `results_fix/KET_QUA_FIX.md`.
Khi log in `XONG TOÀN BỘ`: tải `results_fix.zip` về máy, kiểm tra mở được, rồi **Destroy** máy.

## Đợt 3: sửa độ ổn định của Prune + bổ sung seed (lr backbone 5e-5)

Lý do: ở các run Prune, val và cả train loss xấu dần ở nửa sau (vd KAN Prune seed 1: val 82,92 -> 74,06), còn Pool/Drop thì tăng đều.
Thử 3 cách sửa, mỗi cách cho KAN và MLP Prune:
- `wd0`: bộ chấm điểm không weight decay (`--wd_reducer 0`). Với `--prune_norm`, việc chọn token không phụ thuộc độ lớn trọng số
  của bộ chấm điểm, nên weight decay chỉ làm trọng số co lại và mỗi bước cập nhật xáo trộn thứ hạng token ngày càng mạnh.
- `g03`: nhiễu Gumbel ban đầu 0.3 thay vì 1 (`--gumbel 0.3`). Nhiễu 1 lớn hơn độ lệch của điểm (1 sau chuẩn hoá) nên nửa đầu gần như chọn ngẫu nhiên.
- `frz`: đóng băng bộ chấm điểm 1/3 số epoch cuối (`--freeze_reducer 0.33`), nhiễu về 0 đúng lúc đóng băng.

Mỗi epoch log thêm `raw_std` (độ lệch điểm trước chuẩn hoá), `scorer_norm` (độ lớn trọng số bộ chấm điểm),
`flip` (tỉ lệ token được giữ khác epoch trước, trên một batch val cố định).

```bash
cd /workspace/kan-vim
python -m pytest -q                 # phải thấy "55 passed"
# (a) sửa độ ổn định: 12 job (seed 1 rồi seed 0), ~4 giờ trên 2 GPU
bash scripts/run_fix.sh --stab_round --patience 0 --out results_stab
# (b) thêm seed 1/2 cho KANPool, MLP Pool, Static Drop ở lr 5e-5: 6 job, ~2 giờ.
#     Cần thư mục results_lr cũ nằm trong kan-vim/ (giải nén results_lr.zip vào đây) để bỏ qua các job đã xong.
bash scripts/run_fix.sh --lr_round --prune_norm --patience 0 --out results_lr
```
Chạy (a) xong rồi mới chạy (b) (cùng dùng 2 GPU). Bảng: `results_stab/KET_QUA_STAB.md`, `results_lr/KET_QUA_LR.md`.
Tải `results_stab.zip` và `results_lr.zip` về.
