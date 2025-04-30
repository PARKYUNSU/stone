import torch
from tqdm.auto import tqdm


def evaluate(model, loader, device):
    """
    모델과 데이터로더를 받아 최종 정확도를 반환합니다.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating', ncols=100):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total