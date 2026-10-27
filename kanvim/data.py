"""CIFAR-10 / CIFAR-100 / Tiny-ImageNet / Food-101 (auto-download), ImageNet-100 (ImageFolder you prepare).
All images are resized to 224x224 as specified in the proposal."""
import copy
import os
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms as T
from torchvision.datasets.folder import default_loader

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _tf(train, center_crop=False):
    """center_crop: eval transform for non-square images (Food-101); CIFAR / Tiny are square already."""
    if train:
        return T.Compose([T.Resize(224), T.RandomCrop(224, padding=16, padding_mode="reflect"),
                          T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize(224), *([T.CenterCrop(224)] if center_crop else []), T.ToTensor(),
                      T.Normalize(MEAN, STD)])


class _TinyVal(Dataset):
    def __init__(self, root, class_to_idx, tf):
        d = os.path.join(root, "val")
        self.items = []
        for line in open(os.path.join(d, "val_annotations.txt")):
            f, c = line.split("\t")[:2]
            self.items.append((os.path.join(d, "images", f), class_to_idx[c]))
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(default_loader(p)), y


def build_datasets(name, root):
    if name == "cifar10":
        tr = datasets.CIFAR10(root, True, _tf(True), download=True)
        te = datasets.CIFAR10(root, False, _tf(False), download=True)
        return tr, te, 10
    if name == "cifar100":
        tr = datasets.CIFAR100(root, True, _tf(True), download=True)
        te = datasets.CIFAR100(root, False, _tf(False), download=True)
        return tr, te, 100
    if name == "tiny-imagenet":
        base = os.path.join(root, "tiny-imagenet-200")
        if not os.path.isdir(base):
            os.makedirs(root, exist_ok=True)
            z = os.path.join(root, "tiny.zip")
            urllib.request.urlretrieve(TINY_URL, z)
            zipfile.ZipFile(z).extractall(root)
        tr = datasets.ImageFolder(os.path.join(base, "train"), _tf(True))
        te = _TinyVal(base, tr.class_to_idx, _tf(False))
        return tr, te, 200
    if name == "food101":  # 101 classes, 75,750 train / 25,250 test, not drawn from ImageNet (~5 GB)
        tr = datasets.Food101(root, "train", _tf(True), download=True)
        te = datasets.Food101(root, "test", _tf(False, center_crop=True), download=True)
        tr.targets = list(tr._labels)  # for the stratified train / val split
        return tr, te, 101
    if name == "imagenet-100":  # root/imagenet-100/{train,val}/<class>/*.JPEG
        base = os.path.join(root, "imagenet-100")
        t = lambda tr: T.Compose([T.RandomResizedCrop(224) if tr else T.Compose([T.Resize(256), T.CenterCrop(224)]),
                                  *( [T.RandomHorizontalFlip()] if tr else []), T.ToTensor(), T.Normalize(MEAN, STD)])
        return (datasets.ImageFolder(os.path.join(base, "train"), t(True)),
                datasets.ImageFolder(os.path.join(base, "val"), t(False)), 100)
    raise ValueError(name)


def _subset(ds, n, seed=0):
    if not n or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def build_loaders(name, root, batch, workers=2, train_subset=0, test_subset=0):
    """train_subset / test_subset > 0: fixed random subset (same images for every method -> fair)."""
    tr, te, nc = build_datasets(name, root)
    tr, te = _subset(tr, train_subset), _subset(te, test_subset)
    return (DataLoader(tr, batch, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True,
                       persistent_workers=workers > 0),
            DataLoader(te, batch * 2, shuffle=False, num_workers=workers, pin_memory=True), nc)


def _targets(ds):
    t = getattr(ds, "targets", None)
    return None if t is None else [int(v) for v in t]


def split_train_val(n, targets, val_size, seed=12345):
    """Fixed stratified split (independent of the run seed -> every method sees the same val set)."""
    g = torch.Generator().manual_seed(seed)
    if targets is None:
        p = torch.randperm(n, generator=g).tolist()
        return p[val_size:], p[:val_size]
    by_cls = {}
    for i in torch.randperm(n, generator=g).tolist():
        by_cls.setdefault(targets[i], []).append(i)
    per = max(1, val_size // len(by_cls))
    va = [i for idx in by_cls.values() for i in idx[:per]]
    tr = [i for idx in by_cls.values() for i in idx[per:]]
    return tr, va


def build_loaders_val(name, root, batch, workers=2, val_size=5000, train_subset=0, test_subset=0):
    """train / val / test loaders. val = held-out part of the official train set, with eval transforms.
    Model selection and early stopping use val only; test is touched once, at the end."""
    if name == "fake":  # CPU end-to-end test of the pipeline
        tr = datasets.FakeData(1200, (3, 224, 224), 10, _tf(True))
        te = datasets.FakeData(200, (3, 224, 224), 10, _tf(False), random_offset=10 ** 6)
        nc = 10
    else:
        tr, te, nc = build_datasets(name, root)
    ev = copy.copy(tr)
    ev.transform = te.transform if hasattr(te, "transform") else te.tf
    tr_idx, va_idx = split_train_val(len(tr), _targets(tr), val_size)
    if train_subset and train_subset < len(tr_idx):
        g = torch.Generator().manual_seed(0)
        tr_idx = [tr_idx[i] for i in torch.randperm(len(tr_idx), generator=g)[:train_subset].tolist()]
    tr, va, te = Subset(tr, tr_idx), Subset(ev, va_idx), _subset(te, test_subset)
    mk = lambda ds, b, shuf: DataLoader(ds, b, shuffle=shuf, num_workers=workers, drop_last=shuf, pin_memory=True,
                                        persistent_workers=workers > 0)
    return mk(tr, batch, True), mk(va, batch * 2, False), mk(te, batch * 2, False), nc
