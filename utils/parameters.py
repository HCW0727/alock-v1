import torch

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

def parameters_header(model, num_params=5):
    print(f"Printing {num_params} parameters from the model:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= num_params:
            break
        print(f"Parameter {i+1} - {name} | Value: {param.data.flatten()[:5]}")  # 5개의 값을 플래튼하여 출력


