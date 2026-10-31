"""Tạo data/imagenet-100/{train,val}/<wnid> (liên kết, không chép ảnh) từ ImageNet-100 đã tải và giải nén.

  python scripts/prep_imagenet100.py /workspace/in100            # thư mục đã giải nén
  python scripts/prep_imagenet100.py /workspace/in100 --data_root data

Nhận mọi cách sắp xếp có thư mục lớp dạng n01234567, vd bản Kaggle của Ambesh Shekhar:
train.X1 .. train.X4/<wnid>/*.JPEG và val.X/<wnid>/*.JPEG. Thư mục cha có chữ 'val' -> val, còn lại -> train.
Dừng nếu không đủ 100 lớp hoặc train và val khác tập lớp.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WNID = re.compile(r"^n\d{8}$")


def find_classes(src):
    """-> {'train': {wnid: [thư mục, ...]}, 'val': {...}}; không đi vào bên trong thư mục lớp."""
    found = {"train": {}, "val": {}}
    for d, subs, _ in os.walk(src, followlinks=True):
        keep = []
        for s in subs:
            if WNID.match(s):
                split = "val" if "val" in os.path.basename(d).lower() else "train"
                found[split].setdefault(s, []).append(os.path.join(d, s))
            else:
                keep.append(s)
        subs[:] = keep
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="thư mục ImageNet-100 đã giải nén")
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    found = find_classes(a.src)
    tr, va = found["train"], found["val"]
    print(f"thấy {len(tr)} lớp train, {len(va)} lớp val")
    if len(tr) != 100 or set(tr) != set(va):
        sys.exit("không phải ImageNet-100 chuẩn: cần 100 lớp, train và val cùng tập lớp")
    base = os.path.join(a.data_root, "imagenet-100")
    for split, classes in found.items():
        os.makedirs(os.path.join(base, split), exist_ok=True)
        for c, dirs in classes.items():
            dst = os.path.join(base, split, c)
            if os.path.lexists(dst):
                continue
            if len(dirs) == 1:
                os.symlink(os.path.abspath(dirs[0]), dst)
            else:  # một lớp nằm ở nhiều thư mục: liên kết từng ảnh
                os.makedirs(dst)
                for d in dirs:
                    for f in os.listdir(d):
                        os.symlink(os.path.abspath(os.path.join(d, f)), os.path.join(dst, f))
    n = {s: sum(len(os.listdir(os.path.join(base, s, c))) for c in found[s]) for s in found}
    print(f"xong: train {n['train']} ảnh, val {n['val']} ảnh -> {base}")


if __name__ == "__main__":
    main()
