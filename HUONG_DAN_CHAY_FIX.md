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
