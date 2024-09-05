import torch
from tqdm import tqdm

def train_model(model, train_loader, epochs, optimizer, criterion, device) -> None:
    '''
    모델을 학습하는 코드입니다.

    Args:
        model: 학습할 모델
        train_loader: 전처리된 데이터를 불러올 loader
        epochs: epoch 수
        optimizer: 학습에 사용할 최적화 함수
        criterion: 학습에 사용할 손실 함수
        device: 구동할 device
    '''
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, colour='green', ncols=100)
        for inputs, labels in tqdm_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} done. Running loss : {running_loss}")
            

