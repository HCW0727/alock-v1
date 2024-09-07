import torch
import torch.optim as optim
import torch.nn as nn
import os
from utils import create_img_loader, parameters_len, save_model_checkpoint
from models import CNN_Base, Unet2, VGG16
from scripts import train_model, eval_model_classification

def main():
    '''
    CNN,stl10 image classification을 진행하는 코드입니다.
    전처리, 모델, 평가 모두 모듈화시킨 상태입니다.
    '''
    # 맥북 기반으로 개발하고 있기에, mps를 사용합니다. (간단한 모델을 사용해, mps로도 충분히 성능이 나옵니다.)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # loader를 생성합니다.
    train_loader = create_img_loader(data="stl10",
                                     mode='train',
                                     batch_size=32,
                                     shuffle=True)
    
    test_loader = create_img_loader(data="stl10",
                                     mode='test',
                                     batch_size=32,
                                     shuffle=True)

    # model을 불러옵니다.
    model = VGG16(3).to(device)
    print(f"Model Parameters : {parameters_len(model)}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    # model을 train 및 eval 시킵니다.
    train_model(model=model,
                train_loader=train_loader,
                epochs=1,
                optimizer=optimizer,
                criterion=criterion,
                device=device)
    
    eval_acc = eval_model_classification(model=model,
               test_loader=test_loader,
               criterion=criterion,
               device=device)
    
    save_model_checkpoint(model=model, accuracy=eval_acc)

if __name__ == "__main__":
    main()