# ------------------------------------------------------------------------------
# Created by: Bernard Ong
# Created on: Sep 6, 2016
# ------------------------------------------------------------------------------

import os, math, time, pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


# Pre-Process the data sourced
def process_data(filename, training_flag, features, impute, standardize, whiten, y_column, replace_val, replace_by):

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

    # imputation section
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="mean")
        X = imp.fit_transform(X)
    elif impute == "median":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="median")
        X = imp.fit_transform(X)
    elif impute == "most_frequent":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="most_frequent")
        X = imp.fit_transform(X)
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
        quit()

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
        return data, X

# Function to create model, required for KerasClassifier
def create_deepmodel(features=10, dropout=0, seed=7):
    # set seed
    np.random.seed(seed)
	# define the model = hidden layers + nodes
    model = Sequential()
    model.add(Dense(12, input_dim=features, init='uniform', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(features, init='uniform', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
	# Compile the model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def train(X, y, alg, scaler, pca, features, seed=7):
    """
    Trains a new model using the training data.
    """
    np.random.seed(seed)

    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == "deep":
        model = KerasClassifier(build_fn=create_deepmodel, nb_epoch=2, batch_size=2, verbose=1)
    else:
        print 'No model defined for ' + alg
        exit()

    # train model on full data set
    t0 = time.time()
    kfold = StratifiedKFold(y=y, n_folds=3, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, y, cv=kfold)
    print(results.mean())

    # evaluate model - how accurate is the model
    # rating = model.evaluate(X, y, verbose=1)
    # print "   - %s: %.2f" % (model.metrics_names[1], rating[1])
    model.fit(X, y, verbose=1, batch_size=1)

    return model


def calculate_auc(y, y_pred_prob, plot_flag=False):
    """
    updated score function to plot ROC curve
    Create weighted signal and background sets and calculate the AMS.
    """
    #print 'within score w shape=', w.shape
    print 'within score y shape=', y.shape
    print 'within score y estimated shape=',  y_pred_prob.shape  #y_est.shape
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob, pos_label = 1)
    aucscore = auc(fpr, tpr)
    print 'AUC value =', aucscore
    print 'Thresholds = ', thresholds


def score(y, y_est):
    return calculate_auc(y, y_est)



def create_submission(y_test_prob, submit_file):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    submit = pd.DataFrame({'Id': range(1, len(y_test_prob)+1), 'Probability': y_test_prob})

    # finally create the submission file
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)


# ------------------------------------------------------------------------------------------
def main():

    # perform some initialization
    alg = 'deep'  # this is the only one for now - but can incorporate to mainline code later
    features = 10
    impute = 'none'  # mean, median, most_frequent, zeros, none, default for none = -999
    standardize = False
    whiten = False
    foldsno = 5
    load_training_data = True
    load_model = False
    train_model = True
    save_model = False
    create_visualizations = False
    create_submission_file = False
    code_dir = './'
    data_dir = './data/'
    training_file = 'cs-training.csv'
    test_file = 'cs-test.csv'
    submit_file = 'submission.csv'
    model_file = 'deepmodel.pkl'
    y_column = 'SeriousDlqin2yrs'
    replace_val = np.nan # -999 for Higgs Boson
    replace_by = -999
    plot_flag = False

    # change to code directory area
    os.chdir(code_dir)

    # fix random seed for reproducibility
    seed = 0
    np.random.seed(seed)

    # -------------------------------------------------------------------------------
    # Main Theano Keras Deep Learning Pipeline
    # -------------------------------------------------------------------------------
    print "~" * 80
    print 'Starting Deep Learning Process...'
    print 'alg={0}, impute={1}, standardize={2}, whiten={3}, folds={4}, features={5}'.format(
        alg, impute, standardize, whiten, foldsno, features)

    if load_training_data:
        print "~" * 80
        print 'Reading in Training Data...'
        training_data, X, y, scaler, pca = process_data(
            data_dir + training_file, True, features, impute, standardize, whiten, y_column, replace_val, replace_by)

    if train_model:
        print "~" * 80
        print 'Training Model on Full Data...'
        # model = train(X, y, alg, scaler, pca, features)
        model = train(X, y, alg, scaler, pca, features)

        print "~" * 80
        print 'Calculating Predictions based on Full Training Data...'
        y_prob = model.predict_proba(X, batch_size=1, verbose=1) #'predict_proba' returns 2 columns: 1-prob, prob
        print y_prob

        print "~" * 80
        print 'Calculating Full Training Data AUC...'
        auc_val = score(y, y_prob[:,1])
        print '   - Full Data Prediction AUC =', auc_val

        # print "~" * 80
        # print 'Performing Cross-Validation...'
        # val = cross_validate(X, y, scaler, pca, foldsno, features, seed)

        # print "~" * 80
        # print 'Net Average: Cross-Validation AUC Score =', val


    if create_submission_file:
        print "~" * 80
        print 'Reading in Test Data...'
        test_data, X_test, a, b, c = process_data(
            data_dir + test_file, False, features, impute, standardize, whiten, y_column, replace_val, replace_by)

        print "~" * 80
        print 'Predicting Test Data...'
        y_test_prob = model.predict(X_test)

        print "~" * 80
        print 'Creating Submission File...'
        create_submission(y_test_prob, data_dir + submit_file)

    print "~" * 80
    print 'Deep Learning Process Complete.'


if __name__ == "__main__":
    main()
