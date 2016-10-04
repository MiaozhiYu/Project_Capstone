# ------------------------------------------------------------------------------
# Created by: Emma (Jielei) Zhu
# Created on: Sep 8, 2016
# ------------------------------------------------------------------------------

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
import numpy as np
import pandas as pd
# import xgboost as xgb

from bayes_opt import BayesianOptimization


alg = 'gbm'  # svm, forest, gbm, ada
features = 10
impute = 'none'  # mean, median, most_frequent, knn, interpolate, zeros, none
standardize = False
whiten = False
foldsno = 3
load_training_data = True
load_model = False
train_model = True
save_model = False
data_dir = './data/'
training_file = 'cs-training.csv'
test_file = 'cs-test.csv'
submit_file = alg + 'submission.csv'
model_file = 'model.pkl' 
y_column = 'SeriousDlqin2yrs'
replace_val = np.nan # -999 for Higgs Boson 
replace_by = -999
plot_flag = False
seed = 1234


def process_data(filename, training_flag, features, impute, standardize, whiten, y_column, replace_val, replace_by):
    """
    Reads in training data and prepares numpy arrays.
    """
    data = pd.read_csv(filename, sep=',', index_col=0)

    X = data.drop([y_column], axis=1).values

    if training_flag:
        y = data[y_column].values


    if replace_val is not None:
        print '\n'
        print 'replacing...'
        if np.isnan(replace_val):
            X[np.where(np.isnan(X))] = replace_by
        else:
            X[X == replace_val] = replace_by
        print 'Total entries of ' + str(replace_val) + ': ' + str(np.sum(X==replace_val))
        print 'Total entries of' + str(replace_by) + ': ' + str(np.sum(X==replace_by))
        print '\n'
             

    # optionally impute the -999 values
    # Bernard - added the median and most frequent code for imputting
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="mean")
        X = imp.fit_transform(X)
    elif impute == "median":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="median")
        X = imp.fit_transform(X)
    elif impute == "most_frequent":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="most_frequent")
        X = imp.fit_transform(X)
    elif impute == "knn":
        X = KNN(k = math.sqrt(len(X))).complete(X)
    #elif impute == "interpolate":
        #X[X==-999] = np.NaN
        #X = X.interpolate()
    #elif impute == 'biscaler':
    #elif impute == "nuclear":
    #elif impute == "softimpute":
    elif impute == 'zeros':
        print '\n'
        print 'Total number of zeros: ' + str(np.sum(X==0))
        print 'Imputing...'
        X[np.where(np.isnan(X))] = 0
        print 'Total number of zeros: ' + str(np.sum(X==0))
        print '\n'
    elif impute == 'none':
        pass
    else:
        print 'Error: Imputation method not found.'
        exit()

    # create a standardization transform
    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)

    # create a PCA transform
    pca = None
    if whiten:
        pca = decomposition.PCA(whiten=True)
        pca.fit(X)

    if training_flag:
        return data, X, y, scaler, pca
    else:
        return data, X, scaler, pca


data, X, y, scaler, pca = process_data('./data/cs-training.csv', True, features, impute, standardize, whiten, y_column, replace_val, replace_by)


def svccv(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, seed=2),
                           data, target, 'f1', cv=5).mean()

def rfccv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state=seed):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               max_depth=int(max_depth),
                               min_samples_split=int(min_samples_split),
                               min_samples_leaf=int(min_samples_leaf),
                               max_features=min(max_features, 0.999),
                               random_state=random_state),
                           X, y, 'roc_auc', cv=10).mean()

def gbmcv(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features='log2', seed=2):
    return cross_val_score(GBC(n_estimators=int(n_estimators),
                               max_depth=int(max_depth),
                               min_samples_split=int(min_samples_split),
                               min_samples_leaf=int(min_samples_leaf),
                               max_features=max_features,
                               random_state=seed),
                           X, y, 'roc_auc', cv=10).mean()

def xgbcv(n_estimators, eta, colsample_bytree, min_child_weight, gamma, max_depth, subsample, seed=seed, objective='binary:logistic', missing=replace_by):
    return xgb.train(list(n_estimators=int(n_estimators),
                               eta=float(eta),
                               colsample_bytree=float(colsample_bytree),
                               min_child_weight=int(min_child_weight),
                               gamma=float(gamma),
                               max_depth=int(max_depth),
                               subsample=float(subsample),
                               seed=int(seed),
                               objective=objective),
                      xgb.DMatrix(X, label=y, missing=missing), 150)

def adacv(n_estimators, learning_rate, seed = seed):
    return cross_val_score(ABC(n_estimators=int(n_estimators),
                               learning_rate=float(learning_rate),
                               random_state=int(seed)),
                           X, y, 'roc_auc', cv=10).mean()




if __name__ == "__main__":

    if alg == 'svm':
        optimizer = BayesianOptimization(svccv, {'C': (0.001, 100), 'gamma': (0.0001, 0.1)})
        svcBO.explore({'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})

    elif alg == 'forest':
        optimizer = BayesianOptimization(rfccv, 
          {"n_estimators": (1, 400),
          "max_depth": (1, 30),
          "min_samples_split": (20, 400),
          "min_samples_leaf": (1, 200),
          "max_features": (0.1, 0.99)
           })

    elif alg == 'gbm':
        optimizer = BayesianOptimization(gbmcv, 
          {"n_estimators":(1, 400), 
          "max_depth":(1, 30),
          "min_samples_split":(20, 400), 
          "min_samples_leaf":(1, 200)
          })

    elif alg == 'ada':
        optimizer = BayesianOptimization(adacv,
          {"n_estimators":(200, 1000),
          "learning_rate":(0.1, 0.5)
          })

    #!!!!! xgboost is not working
    # elif alg == 'xgboost':
    #     xgbBO = BayesianOptimization(xgbcv,
    #       {"n_estimators":(1, 400),
    #       "eta":(0.001, 1),
    #       "colsample_bytree":(0.001, 1),
    #       "min_child_weight":(1, 100),
    #       "gamma":(0.001, 1),
    #       "max_depth":(1, 100),
    #       "subsample":(0.001, 1)
    #       })
    #     print('-'*53)
    #     xgbBO.maximize()
    #     print('-'*53)
    #     print('Final Results')
    #     print(alg + ': %f' % rfcBO.res['max']['max_val'])
    else:
        print 'No such algorithm'
        exit()

    print('-'*53)
    optimizer.maximize()
    print('-'*53)
    print('Final Results')
    print(alg + ': %f' % optimizer.res['max']['max_val'])
    

