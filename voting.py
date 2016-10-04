# ------------------------------------------------------------------------------
# Created by: Emma (Jielei) Zhu
# Created on: Sep 11, 2016
# ------------------------------------------------------------------------------

import os, math, time, pickle
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint

from itertools import product
from sklearn.ensemble import VotingClassifier 
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC



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


def calculate_auc(y, y_pred_prob, plot_flag = False):
    """
    updated score function to plot ROC curve
    Create weighted signal and background sets and calculate the AMS.
    """
    #print 'within score w shape=', w.shape
    print 'within score y shape=', y.shape
    print 'within score y estimated shape=',  y_pred_prob.shape  #y_est.shape
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob, pos_label = 1)
    aucscore = auc(fpr, tpr)
    print 'AUC = ', aucscore
    print 'Thresholds = ', thresholds

    #visualize roc 
    if plot_flag:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % aucscore)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    return aucscore


def train(X, y, alg, scaler, pca, features, replace_by, foldsno, seed = 1234):
	clf1 = GradientBoostingClassifier(n_estimators=197, 
									  max_depth=5,
									  min_samples_split=319,
									  min_samples_leaf=89,
									  max_features='log2',
									  random_state=seed)
	clf2 = RandomForestClassifier(n_estimators=161,
								  criterion='gini',
								  min_samples_split=223,
								  min_samples_leaf=9,
								  max_features=1,
								  max_depth=14,
								  random_state=seed)

	eclf = VotingClassifier(estimators=[('gbm', clf1), ('rf', clf2)], voting='soft')
    
    eclf.fit(X, y)

	return eclf


def cross_validate(X, y, alg, scaler, pca, foldscount, features, missing):
    """
    Perform cross-validation on the training set and compute the AMS scores.
    """
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    scores = [0] * foldscount
    folds = cross_validation.StratifiedKFold(y, n_folds=foldscount)
    i = 0

    for i_train, i_val in folds:
        # create the training and validation sets
        X_train, X_val = X[i_train], X[i_val]
        y_train, y_val = y[i_train], y[i_val]

        # train the model
        model = train(X_train, y_train, alg, scaler, pca, features, missing, foldscount)

        # predict and score performance on the validation set
        y_val_prob = predict(X_val, model, alg, scaler, pca)
        scores[i] = score(y_val, y_val_prob)
        i += 1

    return np.mean(scores)



def predict(X, model, alg, scaler, pca):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == 'xgboost':
        xgmat = xgb.DMatrix(X, missing=-999.0)
        y_prob = model.predict(xgmat)
    else:
        y_prob = model.predict_proba(X)[:, 1]

    return y_prob

def score(y, y_est):
    return calculate_auc(y, y_est)


def create_submission(y_test_prob, submit_file):
    """
    Create a new data frame with the submission data.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    submit = pd.DataFrame({'Id': range(1, len(y_test_prob)+1), 'Probability': y_test_prob})

    # finally create the submission file
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)


def main():
    # perform some initialization
    alg = 'voting'  # bayes, logistic, svm, forest, boost, dectree, dectree, adaboost, xgboost
    features = 10
    impute = 'none'  # mean, median, most_frequent, knn, interpolate, zeros, none
    standardize = False
    whiten = False
    foldsno = 5
    load_training_data = True
    load_model = False
    train_model = True
    save_model = False
    create_visualizations = False
    create_submission_file = True
    code_dir = './'
    data_dir = './data/'
    training_file = 'cs-training.csv'
    test_file = 'cs-test.csv'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submit_file = alg + 'submission' + timestr + '.csv'
    model_file = 'model.pkl' 
    y_column = 'SeriousDlqin2yrs'
    replace_val = np.nan # -999 for Higgs Boson 
    replace_by = -999
    plot_flag = False

    os.chdir(code_dir)

    print 'Starting process...'
    print 'alg={0}, impute={1}, standardize={2}, whiten={3}, folds={4}'.format(
        alg, impute, standardize, whiten, foldsno)

    if load_training_data:
        print 'Reading in training data...'
        training_data, X, y, scaler, pca = process_data(
            data_dir + training_file, True, features, impute, standardize, whiten, y_column, replace_val, replace_by)  

    if create_visualizations:
        print 'Creating visualizations...'
        visualize(training_data, X, y, scaler, pca, features)

    if load_model:
        print 'Loading model from disk...'
        model = load(alg, data_dir + model_file)

    if train_model:
        print 'Training model on full data set...'
        model = train(X, y, alg, scaler, pca, features, replace_by, foldsno)

        print 'Calculating predictions...'
        y_prob = predict(X, model, alg, scaler, pca)

        print 'Calculating AUC...'
        auc_val = score(y, y_prob)
        print 'AUC =', auc_val

        print 'Performing cross-validation...'
        val = cross_validate(X, y, alg, scaler, pca, foldsno, features, replace_by)
        print'Cross-validation AUC =', val

    if save_model:
        print 'Saving model to disk...'
        save(alg, model, data_dir + model_file)

    if create_submission_file:
        print 'Reading in test data...'
        test_data, X_test, scaler, pca = process_data(
            data_dir + test_file, False, features, impute, standardize, whiten, y_column, replace_val, replace_by)

        print 'Predicting test data...'
        y_test_prob = predict(X_test, model, alg, scaler, pca)

        print 'Creating submission file...'
        create_submission(y_test_prob, data_dir + submit_file)

    print 'Process complete.'


if __name__ == "__main__":
    main()



