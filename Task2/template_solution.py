# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, Exponentiation, ExpSineSquared, DotProduct, WhiteKernel
from sklearn.metrics import r2_score, make_scorer
from sklearn import preprocessing


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    train_df = train_df.dropna(subset=["price_CHF"])
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Map Seasons to numbers
    train_df_dummies = pd.get_dummies(train_df['season'])
    train_merged = pd.concat([train_df, train_df_dummies], axis='columns')
    train_merged.drop(["season"],  inplace = True, axis = 'columns')

    test_df_dummies = pd.get_dummies(test_df['season'])
    test_merged = pd.concat([test_df, test_df_dummies], axis='columns')
    test_merged.drop(["season"], inplace = True, axis = 'columns')


    y_train = train_merged["price_CHF"].values
    X_train = train_merged.drop(columns=["price_CHF"]).values
    X_test = test_merged.values
    #impute values using KNN
    imputer =  KNNImputer(n_neighbors=2)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"

    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    """
        {
        'kernel': [DotProduct(sigma_0=s) for s in np.logspace(-0.3, 0.3, 10)],
        'alpha': [ 1],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
    },

    {
        'kernel': [RationalQuadratic(length_scale=l) for l in np.logspace(-1, 1, 10)],
        'alpha': [1e-4, 1e-2, 1e-1, 1],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
    },

         {
        'kernel': [Matern(length_scale=l, nu=n) for l in np.logspace(-4, 10, 100) for n in [1231232312.5]],
        'alpha': [1e-3],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
    }

    {
        'kernel': [Matern(length_scale=l, nu=n) for l in np.logspace(-1, 1, 10) for n in [0.5, 1.5, 2.5]],
        'alpha': [1e-4, 1e-2, 1e-1, 1],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],
    }

     'kernel': [RBF(length_scale=l) for l in np.logspace(-10, 2, 50)],
        'alpha': [1e-2,1e-1, 1],
        'optimizer': ['fmin_l_bfgs_b', 'fmin_cg'],


        {
        'kernel': [(s**2)*Matern(length_scale=l, nu=n) for l in np.linspace(0.3, 0.3, 1) for n in [0.5, 1.5, 2.5, 3] for s in np.linspace(1,1, 1)],
        'alpha': [0.2],
        'optimizer': ['fmin_l_bfgs_b'],
        'normalize_y': [False],
    },
    """

    params = [
    {'kernel': [s*Matern(length_scale=l, nu=0.5)+RBF(length_scale=l) for l in np.linspace(0.5, 0.5, 1) for s in np.logspace(1e-6, 1000, 100)],
        'alpha': [0.9*1e-1],
        'optimizer': ['fmin_l_bfgs_b'],
    }

    
   ]

    
    search = GridSearchCV(GaussianProcessRegressor(), param_grid=params, cv = 5, refit=True, n_jobs = -1, verbose= 3,scoring ="r2")

    search.fit(X_train, y_train)
    
    y_pred = search.predict(X_test)
    print("Best score: ",search.best_score_)
    print(search.best_params_)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
