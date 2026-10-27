#!/usr/bin/env bash
# Chạy lại nhánh Prune đã sửa ở chế độ nền (tắt trình duyệt / mất kết nối vẫn chạy):
#   bash scripts/run_fix.sh
# Xem tiến độ:  tail -f run_fix.log      Máy khởi động lại: chạy lại lệnh này, tự chạy tiếp.
cd "$(dirname "$0")/.."
nohup python scripts/run_fix.py "$@" >> run_fix.log 2>&1 &
echo "Đang chạy nền (PID $!). Xem tiến độ: tail -f $(pwd)/run_fix.log"
echo "Khi xong: tải file $(pwd)/results_fix.zip về máy."
