# Source extracted from the Colab experiment; dataset and GPU are required.
# ---- Cell 1 ----
 # -*- coding: utf-8 -*-

# ---- Cell 2 ----
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import tifffile
    from scipy.ndimage import map_coordinates, distance_transform_edt
    import cv2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tifffile', 'scipy', 'opencv-python-headless'])
    import tifffile
    from scipy.ndimage import map_coordinates, distance_transform_edt
    import cv2

@dataclass
class Config:
    seed: int = 42
    quick_run: bool = False       # False = 스터디 결과용 재학습
    paper_channels: bool = True   # False = 메모리 절약용 32채널 (논문 기본은 64)
    batch_size: int = 1           # 원논문
    momentum: float = 0.99        # 원논문
    lr: float = 0.001              # 실험 설정: 논문과 완전히 같은 LR schedule이라는 의미 아님
    weight_decay: float = 0.0
    elastic_std: float = 10.0     # 논문 3x3 coarse grid Gaussian std=10 px
    elastic_probability: float = 0.75
    boundary_sigma: float = 5.0
    boundary_boost: float = 2.0  # 원 논문 w0=10, d1+d2 방식과 다름
    dropout_p: float = 0.5      # 원논문은 contracting-path 끝에서 dropout 사용, p는 실험 설정
    steps_per_epoch: int = 64
    epochs: int = 40
    num_workers: int = 0
    out_dir: str = 'isbi2012_unet_paper_results'
    data_dir: str = 'isbi2012_data'

cfg = Config()
if cfg.quick_run:
    cfg.epochs = 2
    cfg.steps_per_epoch = 6
if not cfg.paper_channels:
    print('주의: 현재 base_channels=32 축소 모델이며 논문 채널 수와 다릅니다.')

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(cfg.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = Path(cfg.out_dir)
output_dir.mkdir(parents=True, exist_ok=True)
print('Device:', device, '| quick_run:', cfg.quick_run,
      '| base_channels:', 64 if cfg.paper_channels else 32)

# ---- Cell 3 ----
DATA_MIRRORS = [
    'https://raw.githubusercontent.com/zhixuhao/unet/master/data/membrane/',
    'https://raw.githubusercontent.com/hayashimasa/UNet-PyTorch/main/data/',
    'https://raw.githubusercontent.com/KevinDruciak/slicer-connectomics-toolkit/master/sample_data/isbi2012/',
]

def download_isbi(data_dir: str = 'isbi2012_data'):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ('train-volume.tif', 'train-labels.tif'):
        path = root / name
        if path.exists() and path.stat().st_size > 100_000:
            continue
        errors = []
        for base in DATA_MIRRORS:
            try:
                tmp = path.with_suffix('.part')
                request = urllib.request.Request(base + name, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(request, timeout=90) as response, open(tmp, 'wb') as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                if tmp.stat().st_size < 100_000:
                    raise RuntimeError('Downloaded file too small')
                tmp.replace(path)
                print('다운로드:', name, f'{path.stat().st_size/1e6:.1f}MB')
                break
            except Exception as e:
                errors.append(f'{base}: {str(e)[:90]}')
                if tmp.exists():
                    tmp.unlink()
        if not path.exists():
            raise RuntimeError(f'{name} 다운로드 실패. 직접 {root}에 넣어 주세요.\n' + '\n'.join(errors))
    images = tifffile.imread(root / 'train-volume.tif')
    labels = tifffile.imread(root / 'train-labels.tif')
    if images.shape != (30, 512, 512) or labels.shape != (30, 512, 512):
        raise ValueError(f'예상 데이터 크기 (30,512,512), 실제 {images.shape}, {labels.shape}')
    images = images.astype(np.float32) / 255.0
    labels = (labels > 127).astype(np.uint8)  # 1=신경 내부, 0=막/경계
    return images, labels

# ---- Cell 4 ----
class DoubleConvValid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


def center_crop_like(encoder_feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eh, ew = encoder_feature.shape[-2:]
    th, tw = target.shape[-2:]
    if eh < th or ew < tw:
        raise ValueError(f'Skip cannot crop {eh}x{ew} to {th}x{tw}')
    top, left = (eh - th) // 2, (ew - tw) // 2
    return encoder_feature[..., top:top+th, left:left+tw]


class UpValid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConvValid(out_ch * 2, out_ch)

    def forward(self, decoder, encoder):
        decoder = self.up(decoder)
        encoder = center_crop_like(encoder, decoder)
        return self.conv(torch.cat([encoder, decoder], dim=1))


class UNet2015(nn.Module):
    def __init__(self, base_channels: int = 64, num_classes: int = 2, dropout_p: float = 0.5):
        super().__init__()
        b = base_channels
        self.down1 = DoubleConvValid(1, b)
        self.down2 = DoubleConvValid(b, b*2)
        self.down3 = DoubleConvValid(b*2, b*4)
        self.down4 = DoubleConvValid(b*4, b*8)
        self.bottleneck = DoubleConvValid(b*8, b*16)
        self.dropout = nn.Dropout2d(p=dropout_p)  # 원논문 contracting path 끝 dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.up4 = UpValid(b*16, b*8)
        self.up3 = UpValid(b*8, b*4)
        self.up2 = UpValid(b*4, b*2)
        self.up1 = UpValid(b*2, b)
        self.head = nn.Conv2d(b, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        z = self.dropout(self.bottleneck(self.pool(x4)))
        z = self.up4(z, x4)
        z = self.up3(z, x3)
        z = self.up2(z, x2)
        z = self.up1(z, x1)
        return self.head(z)


@torch.no_grad()
def check_paper_geometry():
    # 경량 네트워크로 정확한 공간 크기 검증. base_channels=64도 같은 572 -> 388.
    model = UNet2015(base_channels=4).eval()
    out = model(torch.randn(1, 1, 572, 572))
    assert out.shape == (1, 2, 388, 388), out.shape
    print('Figure 1 shape test PASS:', (1, 1, 572, 572), '→', tuple(out.shape))

# ---- Cell 5 ----
INPUT_TILE = 572
OUTPUT_TILE = 388
BORDER = (INPUT_TILE - OUTPUT_TILE) // 2
assert BORDER == 92


def elastic_warp_pair(image, mask, rng, std=10.0):
    h, w = image.shape
    dx = cv2.resize(rng.normal(0, std, (3, 3)).astype(np.float32),
                    (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.normal(0, std, (3, 3)).astype(np.float32),
                    (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')
    coords = (yy + dy, xx + dx)
    out_image = map_coordinates(image, coords, order=1, mode='reflect')
    out_mask = map_coordinates(mask, coords, order=0, mode='nearest')
    return out_image, out_mask


def augmented_image_mask(image, mask, rng, elastic_p, elastic_std):
    k = int(rng.integers(4))
    image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    if rng.random() < 0.5:
        image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
    if rng.random() < elastic_p:
        image, mask = elastic_warp_pair(image, mask, rng, elastic_std)
    # 영상 밝기/대비 변화, 레이블은 그대로
    image = np.clip(image * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), 0, 1)
    return image.astype(np.float32), mask.astype(np.uint8)


def patch_from_full(image, mask, y, x):
    padded = np.pad(image, BORDER, mode='reflect')
    image_patch = padded[y:y+INPUT_TILE, x:x+INPUT_TILE]
    target_patch = mask[y:y+OUTPUT_TILE, x:x+OUTPUT_TILE]
    if image_patch.shape != (INPUT_TILE, INPUT_TILE) or target_patch.shape != (OUTPUT_TILE, OUTPUT_TILE):
        raise ValueError((image_patch.shape, target_patch.shape))
    return image_patch, target_patch


def pixel_weights(mask, class_weights, sigma, boost):
    # 완전한 Ronneberger Eq. (2) 구현이 아님.
    # d1+d2 두 개의 개별 세포 경계 대신 신경막까지의 거리로 근사.
    distance = distance_transform_edt(mask == 1)
    weights = class_weights[mask].astype(np.float32)
    weights += boost * np.exp(-(distance**2) / (2*sigma**2)).astype(np.float32)
    return weights


class ISBIPatchDataset(Dataset):
    def __init__(self, images, labels, indices, training, steps_per_epoch, config):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.training = training
        self.steps = steps_per_epoch
        self.config = config
        # 두 클래스를 원본 픽셀 빈도 기반으로 균형 있게 가중. 원논문 wc와 동일 계산은 아님.
        pixel_counts = np.bincount(labels[self.indices].reshape(-1), minlength=2).astype(np.float64)
        freq = pixel_counts / pixel_counts.sum()
        self.class_weights = (1 / np.maximum(freq, 1e-5))
        self.class_weights /= self.class_weights.mean()

    def __len__(self):
        return self.steps if self.training else len(self.indices)

    def __getitem__(self, idx):
        # PyTorch worker 설정 0이 기본. 랜덤 seed는 전역 generator 사용.
        if self.training:
            slice_id = int(np.random.choice(self.indices))
            image, mask = augmented_image_mask(
                self.images[slice_id], self.labels[slice_id], np.random.default_rng(int(np.random.randint(0, 2**31))),
                self.config.elastic_probability, self.config.elastic_std)
            y = int(np.random.randint(0, 512-OUTPUT_TILE+1))
            x = int(np.random.randint(0, 512-OUTPUT_TILE+1))
        else:
            slice_id = int(self.indices[idx])
            image, mask = self.images[slice_id], self.labels[slice_id]
            y = x = (512-OUTPUT_TILE)//2
        ip, mp = patch_from_full(image, mask, y, x)
        weight = pixel_weights(mp, self.class_weights,
                               self.config.boundary_sigma, self.config.boundary_boost)
        return (torch.from_numpy(ip[None].copy()).float(),
                torch.from_numpy(mp.copy()).long(),
                torch.from_numpy(weight.copy()).float())

# ---- Cell 6 ----
def weighted_cross_entropy(logits, target, weight):
    loss = F.cross_entropy(logits, target, reduction='none')
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def confusion_update(confusion, pred, target, n=2):
    k = target.reshape(-1).long()*n + pred.reshape(-1).long()
    confusion += torch.bincount(k, minlength=n*n).reshape(n, n).cpu().numpy()


def compute_metrics(confusion):
    tp = confusion.diagonal().astype(float)
    union = confusion.sum(0) + confusion.sum(1) - tp
    sum_dice = confusion.sum(0) + confusion.sum(1)
    iou = tp/np.maximum(union, 1.0)
    dice = (2*tp)/np.maximum(sum_dice, 1.0)
    return {
        'mIoU': float(iou.mean()),
        'mean_dice': float(dice.mean()),
        'membrane_iou': float(iou[0]),
        'membrane_dice': float(dice[0]),
        'pixel_accuracy': float(tp.sum()/max(confusion.sum(),1)),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y, weight in loader:
        x, y, weight = x.to(device), y.to(device), weight.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = weighted_cross_entropy(logits, y, weight)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_center(model, loader, device):
    model.eval()
    confusion = np.zeros((2,2), dtype=np.int64)
    loss_sum = 0.0
    for x, y, weight in loader:
        x, y, weight = x.to(device), y.to(device), weight.to(device)
        logits = model(x)
        loss_sum += weighted_cross_entropy(logits, y, weight).item()
        confusion_update(confusion, logits.argmax(1), y)
    return loss_sum/max(len(loader),1), compute_metrics(confusion)


def run_training(cfg: Config, images, labels, device):
    # 인접한 3D slice 사이의 상관관계를 줄이기 위해 연속된 마지막 6장을 검증 사용.
    train_idx = np.arange(0, 24)
    val_idx = np.arange(24, 30)
    train_dataset = ISBIPatchDataset(images, labels, train_idx, True,
                                    cfg.steps_per_epoch, cfg)
    val_dataset = ISBIPatchDataset(images, labels, val_idx, False, 0, cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    base_channels = 64 if cfg.paper_channels else 32
    model = UNet2015(base_channels=base_channels, dropout_p=cfg.dropout_p).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    history = []
    best = -1.0
    exp_dir = Path(cfg.out_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir/'best_paper_unet.pt'
    for epoch in range(1, cfg.epochs+1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        vloss, metrics = evaluate_center(model, val_loader, device)
        entry = dict(epoch=epoch, train_loss=loss, val_loss=vloss, **metrics)
        history.append(entry)
        print(f'Epoch {epoch:02d}/{cfg.epochs}: train_loss={loss:.4f} '
              f'val_center_mIoU={metrics["mIoU"]:.4f} '
              f'val_center_membrane_IoU={metrics["membrane_iou"]:.4f}')
        if metrics['mIoU'] > best:
            best = metrics['mIoU']
            torch.save({'model': model.state_dict(), 'config': asdict(cfg),
                        'best_center_val_mIoU': best}, ckpt_path)
    (exp_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    chk = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(chk['model'])
    return model, history, val_idx

# ---- Cell 7 ----
@torch.no_grad()
def predict_full_image(model, gray: np.ndarray, device: torch.device):
    model.eval()
    h, w = gray.shape
    tile_out = OUTPUT_TILE
    rows, cols = math.ceil(h/tile_out), math.ceil(w/tile_out)
    extra_y = rows*tile_out - h
    extra_x = cols*tile_out - w
    padded = np.pad(gray, ((BORDER, BORDER+extra_y), (BORDER, BORDER+extra_x)), mode='reflect')
    out = np.zeros((rows*tile_out, cols*tile_out), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            y, x = i*tile_out, j*tile_out
            tile = padded[y:y+INPUT_TILE, x:x+INPUT_TILE]
            assert tile.shape == (INPUT_TILE, INPUT_TILE), tile.shape
            inp = torch.from_numpy(tile[None,None].copy()).float().to(device)
            logits = model(inp)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            out[y:y+tile_out, x:x+tile_out] = pred
    return out[:h, :w]


def evaluate_full_images(model, images, labels, val_indices, device):
    confusion = np.zeros((2,2), dtype=np.int64)
    preds = []
    for idx in val_indices:
        pred = predict_full_image(model, images[int(idx)], device)
        preds.append(pred)
        confusion_update(confusion, torch.from_numpy(pred).unsqueeze(0),
                         torch.from_numpy(labels[int(idx)]).unsqueeze(0))
    return compute_metrics(confusion), preds


def visualize_results(images, labels, indices, preds, destination):
    import matplotlib.pyplot as plt
    n = min(3, len(indices))
    fig, axes = plt.subplots(n, 3, figsize=(10, n*3.2), squeeze=False)
    for i in range(n):
        idx = int(indices[i])
        axes[i,0].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        axes[i,1].imshow(labels[idx], cmap='gray', vmin=0, vmax=1)
        axes[i,2].imshow(preds[i], cmap='gray', vmin=0, vmax=1)
        axes[i,0].set_ylabel(f'Slice {idx}', fontsize=10)
        for j, title in enumerate(('EM input', 'GT (white=cell interior)', 'Predicted mask')):
            axes[i,j].set_title(title, fontsize=10)
            axes[i,j].axis('off')
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_history(history, destination):
    import matplotlib.pyplot as plt
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, [h['train_loss'] for h in history], label='Train loss')
    axes[0].plot(epochs, [h['val_loss'] for h in history], label='Val loss')
    axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].set_title('Weighted CE')
    axes[1].plot(epochs, [h['mIoU'] for h in history], label='Center-tile val mIoU')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mIoU'); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].set_title('Validation mIoU')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.show(); plt.close(fig)

# ---- Cell 8 ----
def main():
    check_paper_geometry()
    images, labels = download_isbi(cfg.data_dir)
    print('Images:', images.shape, 'Labels:', labels.shape, 'Classes:', np.unique(labels))
    assert images.shape == labels.shape == (30, 512, 512)
    # 텐서 및 증강 예시 확인
    tmp_ds = ISBIPatchDataset(images, labels, np.arange(24), True, 1, cfg)
    a, b, w = tmp_ds[0]
    assert a.shape == (1, 572, 572) and b.shape == (388, 388) and w.shape == b.shape
    print('Patch sample:', tuple(a.shape), tuple(b.shape), tuple(w.shape))
    model, history, val_indices = run_training(cfg, images, labels, device)
    plot_history(history, output_dir/'training_curves.png')
    metrics, preds = evaluate_full_images(model, images, labels, val_indices, device)
    visualize_results(images, labels, val_indices, preds, output_dir/'sample_predictions.png')
    result = {'evaluation': 'held-out 6 training-volume slices, full-image overlap tile',
              'quick_run': cfg.quick_run,
              'dataset': 'ISBI 2012 EM train-volume.tif (24 train / 6 val, no secret challenge test)',
              'full_validation_metrics': metrics,
              'best_center_validation_mIoU': max(h['mIoU'] for h in history),
              'config': asdict(cfg),
              'paper_differences': [
                  'Training duration / LR schedule not original',
                  'Weighted loss uses Gaussian membrane-distance approximation, not exact instance d1+d2',
                  'Rotation ensemble and original challenge metrics not implemented',
                  'Elastic deformation grid is approximated with bicubic interpolation',
                  'Dropout probability 0.5 is an experiment setting (not uniquely specified in paper)',
              ]}
    (output_dir/'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n전체 512×512 검증 결과 (공식 test 점수 아님):')
    print(json.dumps(metrics, indent=2))
    print('Saved outputs:', output_dir.resolve())
    return result

# Threshold analysis is a separate callable step; importing the module does not run training or inference.
def run_threshold_sweep():
    """Evaluate saved checkpoint thresholds on the held-out validation slices."""
    import csv

    MEMBRANE_CLASS = 0        # ISBI masks: 0=membrane (black), 1=cell interior (white)
    THRESHOLDS = (0.5, 0.6, 0.7)
    CKPT_PATH = Path(cfg.out_dir) / 'best_paper_unet.pt'
    SWEEP_DIR = Path(cfg.out_dir) / 'threshold_sweep'
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.is_file():
        raise FileNotFoundError(
            f'최고 성능 체크포인트를 찾지 못했습니다: {CKPT_PATH}\n'
            '기존 학습 런타임에서 실행하거나, best_paper_unet.pt를 해당 경로에 넣어주세요.\n'
            'Colab 런타임이 초기화된 경우 체크포인트가 삭제됐을 수 있습니다.'
        )

    # 이전 학습에서 저장된 모델의 설정을 읽어 정확한 채널 수로 복구한다.
    try:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    train_cfg = checkpoint.get('config', {})
    base_channels = 64 if train_cfg.get('paper_channels', cfg.paper_channels) else 32
    model_sweep = UNet2015(
        base_channels=base_channels,
        num_classes=2,
        dropout_p=train_cfg.get('dropout_p', cfg.dropout_p)
    ).to(device)
    model_sweep.load_state_dict(checkpoint['model'], strict=True)
    model_sweep.eval()
    print('Loaded checkpoint:', CKPT_PATH)
    print('Best training-time center validation mIoU:', checkpoint.get('best_center_val_mIoU', '(unavailable)'))

    images_sweep, labels_sweep = download_isbi(cfg.data_dir)
    validation_indices = np.arange(24, 30)  # original train 0..23, val 24..29

    @torch.no_grad()
    def predict_membrane_probability_full(model, gray, device):
        # Returns class-0 (membrane) probability for the entire image via overlap-tile.
        model.eval()
        h, w = gray.shape
        rows, cols = math.ceil(h / OUTPUT_TILE), math.ceil(w / OUTPUT_TILE)
        extra_y = rows * OUTPUT_TILE - h
        extra_x = cols * OUTPUT_TILE - w
        padded = np.pad(
            gray, ((BORDER, BORDER + extra_y), (BORDER, BORDER + extra_x)),
            mode='reflect'
        )
        probs = np.zeros((rows * OUTPUT_TILE, cols * OUTPUT_TILE), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                y, x = i * OUTPUT_TILE, j * OUTPUT_TILE
                tile = padded[y:y + INPUT_TILE, x:x + INPUT_TILE]
                assert tile.shape == (INPUT_TILE, INPUT_TILE), tile.shape
                inp = torch.from_numpy(tile[None, None].copy()).float().to(device)
                logits = model(inp)
                p_membrane = torch.softmax(logits, dim=1)[0, MEMBRANE_CLASS]
                probs[y:y+OUTPUT_TILE, x:x+OUTPUT_TILE] = p_membrane.cpu().numpy()
        return probs[:h, :w]


    def predict_mask_at_threshold(prob_membrane, threshold):
        # 0 (black) = membrane. Higher threshold => fewer pixels predicted membrane.
        return np.where(prob_membrane >= threshold, 0, 1).astype(np.uint8)

    # Run network inference ONCE per image. Sweep thresholds on cached probabilities.
    probability_maps = []
    for index in validation_indices:
        probability_maps.append(
            predict_membrane_probability_full(model_sweep, images_sweep[int(index)], device)
        )

    rows = []
    predictions_by_threshold = {}
    for t in THRESHOLDS:
        cm = np.zeros((2, 2), dtype=np.int64)
        masks = []
        for idx, prob in zip(validation_indices, probability_maps):
            prediction = predict_mask_at_threshold(prob, t)
            masks.append(prediction)
            ground_truth = labels_sweep[int(idx)]
            confusion_update(cm, torch.from_numpy(prediction), torch.from_numpy(ground_truth))
        metrics = compute_metrics(cm)
        tp = int(cm[MEMBRANE_CLASS, MEMBRANE_CLASS])
        fp = int(cm[1, MEMBRANE_CLASS])  # GT=interior, prediction=membrane => black false positives
        fn = int(cm[MEMBRANE_CLASS, 1])  # GT=membrane, prediction=interior
        metrics.update({
            'threshold': float(t),
            'membrane_precision': tp / max(tp + fp, 1),
            'membrane_recall': tp / max(tp + fn, 1),
            'membrane_fp_pixels': fp,
            'membrane_fn_pixels': fn,
        })
        rows.append(metrics)
        predictions_by_threshold[t] = masks

    print('\nWhole-image validation results (6 images, NOT official test):')
    print('Threshold | mIoU   | Membrane IoU | Membrane Precision | Membrane Recall | Black FP pixels')
    for r in rows:
        print(f"{r['threshold']:>9.2f} | {r['mIoU']:.4f} | {r['membrane_iou']:.4f}      | "
              f"{r['membrane_precision']:.4f}             | {r['membrane_recall']:.4f}          | "
              f"{r['membrane_fp_pixels']:,}")

    with (SWEEP_DIR / 'threshold_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (SWEEP_DIR / 'threshold_metrics.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    # Visual comparison on identical validation slices; black = membrane.
    import matplotlib.pyplot as plt
    n_show = min(3, len(validation_indices))
    fig, axes = plt.subplots(n_show, 2 + len(THRESHOLDS), figsize=(17, 3.5 * n_show), squeeze=False)
    for k in range(n_show):
        idx = int(validation_indices[k])
        mats = [images_sweep[idx], labels_sweep[idx]] + [
            predictions_by_threshold[t][k] for t in THRESHOLDS
        ]
        titles = ['EM input', 'Ground truth', *[f'Pred threshold={t:.1f}' for t in THRESHOLDS]]
        for j, (mat, title) in enumerate(zip(mats, titles)):
            axes[k, j].imshow(mat, cmap='gray', vmin=0, vmax=1)
            axes[k, j].set_title(title)
            axes[k, j].axis('off')
        axes[k, 0].set_ylabel(f'Validation slice {idx}', fontsize=11)
    fig.suptitle('ISBI 2012: membrane=black (class 0); higher threshold is more conservative', y=1.01)
    fig.tight_layout()
    fig.savefig(SWEEP_DIR / 'threshold_comparison.png', dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # This threshold is tuned on validation slices only, so do not call it a test score.
    best_row = max(rows, key=lambda r: r['membrane_iou'])
    print(f"\nBest validation Membrane IoU: threshold={best_row['threshold']:.2f}, "
          f"Membrane IoU={best_row['membrane_iou']:.4f}, mIoU={best_row['mIoU']:.4f}")
    print('Saved:', SWEEP_DIR.resolve())


if __name__ == '__main__':
    main()
    run_threshold_sweep()
