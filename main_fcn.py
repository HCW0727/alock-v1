from models import FCN_Base
from scripts import eval_model_regression, train_model
from utils import preprocess_data, create_csv_loader

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def main():
    csv_path = "./data/text/house_price_prediction.csv"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    x_train, x_test, y_train, y_test = preprocess_data(file_path=csv_path,
                    categorical_features=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                        'airconditioning', 'prefarea', 'furnishingstatus'],
                    numerical_features=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])
    
    model = FCN_Base(13).to(device)
    
    train_loader = create_csv_loader(x_train, y_train, 32, shuffle=True)
    test_loader = create_csv_loader(x_test, y_test, 32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_model(model=model,
                train_loader=train_loader,
                epochs=50,
                optimizer=optimizer,
                criterion=criterion,
                device=device)
    
    eval_model_regression(model=model,
               test_loader=test_loader,
               criterion=criterion,
               device=device)

if __name__ == "__main__":
    main()