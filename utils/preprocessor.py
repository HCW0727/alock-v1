import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(file_path:str,
                    categorical_features:list,
                    numerical_features:list):
    main_df = pd.read_csv(file_path)

    # 추후 자동 column 분류 구현 필요
    # categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
    #                     'airconditioning', 'prefarea', 'furnishingstatus']
    # numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',numerical_transformer,numerical_features),
            ('cat',categorical_transformer,categorical_features)
        ])
    
    X = main_df.drop('price', axis=1)
    Y = main_df['price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    return X_train_processed, X_test_processed, Y_train, Y_test


if __name__ == "__main__":
    preprocess_data()