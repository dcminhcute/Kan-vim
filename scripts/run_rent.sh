#!/usr/bin/env bash
# Chạy toàn bộ quy trình ở chế độ nền (tắt trình duyệt / mất kết nối vẫn chạy):
#   bash scripts/run_rent.sh                     # ngân sách mặc định 16 giờ
#   bash scripts/run_rent.sh --budget_hours 10   # muốn xong trong ~10 giờ
# Xem tiến độ:  tail -f run_plan.log      Chạy lại lệnh này nếu máy bị khởi động lại: tự chạy tiếp.
cd "$(dirname "$0")/.."
nohup python scripts/run_plan.py --profile gpu "$@" >> run_plan.log 2>&1 &
echo "Đang chạy nền (PID $!). Xem tiến độ: tail -f $(pwd)/run_plan.log"
echo "Khi xong: tải file $(pwd)/results_plan.zip về máy."
