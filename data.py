import os
import glob
import random

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def seed_everything(seed: int):
    """재현을 위해 모든 시드를 고정합니다."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class PadSquare(A.ImageOnlyTransform):
    """이미지를 정사각형으로 만들기 위해 양쪽에 패딩을 추가합니다."""

    def __init__(self, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.border_mode = border_mode
        self.value = value

    def apply(self, image, **params):
        h, w = image.shape[:2]
        m = max(h, w)
        top = (m - h) // 2
        bottom = m - h - top
        left = (m - w) // 2
        right = m - w - left
        return cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=self.border_mode,
            value=self.value
        )

    def get_transform_init_args_names(self):
        return ("border_mode", "value")


class CustomDataset(Dataset):
    """이미지 파일 경로와 레이블을 받아 증강을 적용하는 PyTorch Dataset입니다."""

    def __init__(self, img_paths, labels=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # BGR -> RGB 변환
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 증강(transform) 적용
        if self.transform:
            img = self.transform(image=img)["image"]

        if self.labels is not None:
            return img, self.labels[idx]
        return img


def get_dataloaders(
    data_dir: str = "./train",
    img_size: int = 224,
    batch_size: int = 32,
    seed: int = 24,
    val_split: float = 0.3,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    1) data_dir/<클래스명>/*.jpg 패턴으로 이미지 경로를 수집
    2) train/validation 분할 (stratify)
    3) LabelEncoder로 레이블 수치화
    4) train/val 증강 정의
    5) CustomDataset과 DataLoader 생성 후 반환

    Returns:
        train_loader, val_loader, label_encoder
    """
    # 시드 고정
    seed_everything(seed)

    # 이미지 경로와 레이블 수집
    img_paths = glob.glob(os.path.join(data_dir, "*", "*"))
    df = pd.DataFrame({
        "img_path": img_paths,
        "label": [os.path.basename(os.path.dirname(p)) for p in img_paths],
    })

    # 학습/검증 분할
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=seed
    )

    # 레이블 인코딩
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df["label"])
    val_labels = le.transform(val_df["label"])

    # 증강(transform) 정의
    train_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize((img_size, img_size)),
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize((img_size, img_size)),
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Dataset 생성
    train_ds = CustomDataset(
        train_df["img_path"].tolist(),
        train_labels,
        transform=train_transform
    )
    val_ds = CustomDataset(
        val_df["img_path"].tolist(),
        val_labels,
        transform=val_transform
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )

    return train_loader, val_loader, le