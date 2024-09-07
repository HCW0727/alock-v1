import torch
from tqdm import tqdm

def eval_model_regression(model, test_loader, criterion, device):
    '''
    모델의 성능을 테스트하는 코드입니다.

    Args:
        model: 테스트할 모델
        test_loader: 테스트 데이터를 불러올 loader
        device: 구동할 device
    '''
    model.eval()
    evaluation_loss = 0.0

    with torch.no_grad():
        tqdm_bar = tqdm(test_loader, desc=f"Evaluation", leave=False, colour='red', ncols=100)
        for inputs, labels in tqdm_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs,labels)

            evaluation_loss += loss.item() * inputs.size(0)

    evaluation_loss /= len(test_loader.dataset)

    print(f"Evaluation Result | loss : {evaluation_loss} ")



def eval_model_classification(model, test_loader, criterion, device):
    '''
    모델의 성능을 테스트하는 코드입니다.

    Args:
        model: 테스트할 모델
        test_loader: 테스트 데이터를 불러올 loader
        criterion: 손실 함수
        device: 구동할 device
    '''
    model.eval()

    evaluation_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        tqdm_bar = tqdm(test_loader, desc=f"Evaluation", leave=False, colour='red', ncols=100)
        for inputs, labels in tqdm_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            evaluation_loss += loss.item() * inputs.size(0)
            
            # Top-1 Accuracy
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (labels == predicted).sum().item()
            
            # Top-5 Accuracy
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            correct_top5 += (top5_pred == labels.view(-1, 1)).sum().item()

    evaluation_loss /= len(test_loader.dataset)
    accuracy_top1 = correct_top1 / len(test_loader.dataset)
    accuracy_top5 = correct_top5 / len(test_loader.dataset)

    print(f"Evaluation Result | loss: {evaluation_loss:.4f}, Top-1 Accuracy: {accuracy_top1:.4f}, Top-5 Accuracy: {accuracy_top5:.4f}")