# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("./pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("./pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("./train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("./train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("./test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.encoder = nn.Sequential(
            nn.Linear(1000, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(128,32),
    
            nn.Linear(32,16)

        )
        self.decoder = nn.Sequential(
            nn.Linear(16,1)
        )


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        result = self.encoder(x)
        result = self.decoder(result)
        return result
    
    def get_features(self, x):
        return self.encoder(x)
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=False)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    
    epochs = 20
    train_data = TensorDataset(x_tr, y_tr)
    val_data = TensorDataset(x_val, y_val)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_data, batch_size=int(eval_size/epochs), shuffle=False)
    val_dataloader = iter(val_dataloader)

    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    for epoch in range(epochs):
        for X_b, y_b in train_dataloader:
            
            y_hat = model(X_b)
            loss = nn.functional.mse_loss(y_hat.squeeze(), y_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        X_test, y_test = next(val_dataloader)

        y_hat = model(X_test)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y_test)

        print(f'epoch: {epoch}, loss on validation batch: {loss}')



    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        with torch.no_grad():   
            x = torch.tensor(x, dtype=torch.float)
            result = model.get_features(x)
            result = result.squeeze()
            result = result.numpy()
        return result

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = LinearRegression()
    # model = LassoCV(max_iter=5000, eps=0.0001, n_alphas=10000)
    # model = GaussianProcessRegressor()
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    torch.manual_seed(0)
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    test_size = int(sys.argv[1])

    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    feature_maker = PretrainedFeatureClass(feature_extractor="pretrain")

    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    x_train_featurized = feature_maker.transform(x_train)
    
    if test_size > 0:
        x_t_f_1, x_t_f_2, y_1, y_2 = train_test_split(x_train_featurized, y_train, test_size=test_size, shuffle=False)
        regression_model.fit(x_t_f_1, y_1)
        y_hat = regression_model.predict(x_t_f_2)
        val_loss = mean_squared_error(y_2, y_hat)
        print(f"val loss: {val_loss}")
    else:
        regression_model.fit(x_train_featurized, y_train)
        
    x_test_featurized = feature_maker.transform(x_test.to_numpy())
    # x_test_featurized = scaler.transform(x_test_featurized)
    y_pred += regression_model.predict(x_test_featurized)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")