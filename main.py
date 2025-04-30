import torch
import argparse
import train
from model.config import get_b16_config
from data import get_dataloaders
from model.vit import Vision_Transformer
from utils import save_model

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(
        description='Vision Transformer 모델의 학습, 테스트, Attention 시각화를 수행합니다.'
    )
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'visualize'], required=True,
                        help="'train', 'test', 'visualize' 중 하나를 선택")
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='사전학습 또는 fine-tuned 가중치 파일 경로 (.pth)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='학습 에폭 수 (train 모드에서만 사용)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='학습률 (train 모드에서만 사용)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay 계수 (train 모드에서만 사용)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='라벨 스무딩 값 (train 모드에서만 사용)')
    parser.add_argument('--save_fig', action='store_true',
                        help='학습 결과(loss/accuracy) 그래프 저장 여부')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Attention 시각화할 이미지 경로(visualize 모드에서 필수)')
    parser.add_argument('--data_dir', type=str, default='./train',
                        help='학습/검증 데이터 디렉토리(train), 테스트 파일(train 모드 제외)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='입력 이미지 사이즈')
    parser.add_argument('--test_csv', type=str, default='./test.csv',
                        help='테스트 이미지 경로를 가진 CSV 파일(test 모드에서만 사용)')
    return parser.parse_args()


def visualize_attention(image_path, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((model.img_size, model.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = model.patch_embed(image_tensor)
        B = tokens.shape[0]
        cls_tokens = model.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + model.pos_embed

        attn_module = model.encoder.layers[0].attn
        attn_module.vis = True
        _, attn_probs = attn_module(tokens)

    cls_attn = attn_probs[0, 0, 0, 1:].detach().cpu().numpy()
    num_patches = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(num_patches, num_patches)
    attn_map = cv2.resize(attn_map, (model.img_size, model.img_size))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = 1 - attn_map
    attn_map = np.expand_dims(attn_map, axis=-1)

    orig_np = np.array(original_image.resize((model.img_size, model.img_size))) / 255.0
    darkened = (orig_np * attn_map).clip(0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(orig_np)
    ax[0].set_title("원본 이미지")
    ax[0].axis('off')

    ax[1].imshow(darkened)
    ax[1].set_title("ViT Attention Overlay")
    ax[1].axis('off')

    return fig


def inference(model, loader, device, label_encoder=None):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(loader):
            imgs = imgs.float().to(device)
            logits = model(imgs)
            preds += logits.argmax(1).detach().cpu().tolist()
    if label_encoder is not None:
        preds = label_encoder.inverse_transform(preds)
    return preds


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # 데이터 로더
        train_loader, val_loader, label_encoder = get_dataloaders(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=42
        )
        num_classes = len(label_encoder.classes_)

        # 모델
        config = get_b16_config()
        model = Vision_Transformer(
            config,
            img_size=args.img_size,
            num_classes=num_classes,
            in_channels=3,
            pretrained=False
        ).to(device)
        ckpt = torch.load(args.pretrained_path, map_location=device)
        model.load_from(ckpt, strict=False)

        # 학습
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing
        ).to(device)

        train.train(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_fig=args.save_fig
        )
        save_model(model, "fine_tuned_model.pth")

    elif args.mode == 'test':
        # 테스트 데이터 로드
        df_test = pd.read_csv(args.test_csv)
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        test_paths = df_test['img_path'].tolist()
        test_ds = torch.utils.data.TensorDataset(torch.arange(len(test_paths)))  # placeholder
        # 실제 CustomDataset 사용 시 아래처럼 대체하세요
        # test_ds = CustomDataset(test_paths, None, test_transform)
        test_loader = DataLoader(
            CustomDataset(test_paths, None, get_dataloaders.__globals__['A'].Compose([
                PadSquare(value=(0,0,0)),
                A.Resize(args.img_size, args.img_size),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2(),
            ])),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 모델 로드
        config = get_b16_config()
        model = Vision_Transformer(
            config,
            img_size=args.img_size,
            num_classes= len(get_dataloaders.__globals__['LabelEncoder']().classes_) ,
            in_channels=3,
            pretrained=False
        ).to(device)
        ckpt = torch.load(args.pretrained_path, map_location=device)
        model.load_from(ckpt, strict=False)

        # 예측
        preds = inference(model, test_loader, device, label_encoder)
        df_test['pred'] = preds
        df_test.to_csv('submission.csv', index=False)
        print('submission.csv 생성 완료')

    else:  # visualize
        if args.image_path is None:
            raise ValueError("--image_path를 지정해주세요.")
        config = get_b16_config()
        model = Vision_Transformer(
            config,
            img_size=args.img_size,
            num_classes=1,
            in_channels=3,
            pretrained=False
        ).to(device)
        model.encoder.layers[0].attn.vis = True
        ckpt = torch.load(args.pretrained_path, map_location=device)
        model.load_from(ckpt, strict=False)

        fig = visualize_attention(args.image_path, model, device)
        fig.savefig('attention_map.png')
        print('attention_map.png 생성 완료')