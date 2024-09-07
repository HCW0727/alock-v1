from datetime import datetime
import torch, os

def parameters_len(model):
    """
    PyTorch 모델의 파라미터 개수를 출력하는 함수.
    
    Args:
        model (torch.nn.Module): PyTorch 모델 인스턴스.
    
    Returns:
        int: 모델의 총 파라미터 개수.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")
    return total_params

def parameters_header(model, num_params=5) -> None:
    '''
    5개 파라미터의 값을 전달하는 함수. (디버깅용)

    Args:
        model(nn.Module): Pytorch 모델 인스턴스.
    '''
    print(f"Printing {num_params} parameters from the model:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= num_params:
            break
        print(f"Parameter {i+1} - {name} | Value: {param.data.flatten()[:5]}") 


def save_model_checkpoint(model, accuracy):
    """
    모델 체크포인트를 저장하는 함수입니다.
    
    Args:
        model(torch.nn.Module): 저장할 모델
        accuracy(float): 모델의 top1 정확도
    """
    ckpt_base_path = "./data/ckpt"
    current_time = datetime.now().strftime("%m%d")
    checkpoint_path = os.path.join(ckpt_base_path, f"{model.__class__.__name__}_{accuracy}_{current_time}.pt")
    torch.save(model, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

