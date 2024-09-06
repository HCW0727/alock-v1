import torch

# 모델의 파라미터를 5개만 출력하는 코드
def print_model_parameters(model, num_params=5):
    print(f"Printing {num_params} parameters from the model:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= num_params:
            break
        print(f"Parameter {i+1} - {name} | Value: {param.data.flatten()[:5]}")  # 5개의 값을 플래튼하여 출력

