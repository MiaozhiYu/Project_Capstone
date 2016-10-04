import os, math, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# import xgboost as xgb
# from fancyimpute import KNN

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
        quit()

    if training_flag:
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
    
        return data, X, y, scaler, pca
    else: # return test data
        return data, X


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
    print 'AUC value =', aucscore
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


def load(alg, filename):
    """
    Load a previously training model from disk.
    """
    if alg == 'xgboost':
        model = xgb.Booster({'nthread': 16}, model_file=filename)
    elif alg == "svm":
        model = joblib.load( filename)
    else:
        model_file = open(filename, 'rb')
        model = pickle.load(model_file)
        model_file.close()

    return model


def save(alg, model, filename):
    """
    Persist a trained model to disk.
    """
    if alg == 'xgboost':
        model.save_model(filename)
    elif alg == "svm":
        joblib.dump(model, filename)
    else:
        model_file = open(filename, 'wb')
        pickle.dump(model, model_file)
        model_file.close()


def visualize(training_data, X, y, scaler, pca, features):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.

    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """

    # feature histograms
    fig1, ax1 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16):
        ax1[i % 4, i / 4].hist(X[:, i])
        ax1[i % 4, i / 4].set_title(training_data.columns[i + 1])
        ax1[i % 4, i / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16, features):
        ax2[i % 4, (i - 16) / 4].hist(X[:, i])
        ax2[i % 4, (i - 16) / 4].set_title(training_data.columns[i + 1])
        ax2[i % 4, (i - 16) / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig2.tight_layout()

    # covariance matrix
    if scaler is not None:
        X = scaler.transform(X)

    cov = np.cov(X, rowvar=0)

    fig3, ax3 = plt.subplots(figsize=(16, 10))
    p = ax3.pcolor(cov)
    fig3.colorbar(p, ax=ax3)
    ax3.set_title('Feature Covariance Matrix')

    # pca plots
    if pca is not None:
        X = pca.transform(X)

        fig4, ax4 = plt.subplots(figsize=(16, 10))
        ax4.scatter(X[:, 0], X[:, 1], c=y)
        ax4.set_title('First & Second Principal Components')

        fig5, ax5 = plt.subplots(figsize=(16, 10))
        ax5.scatter(X[:, 1], X[:, 2], c=y)
        ax5.set_title('Second & Third Principal Components')


def train(X, y, alg, scaler, pca, features):
    """
    Trains a new model using the training data.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == 'xgboost':
        # use a separate process for the xgboost library
        return train_xgb(X, y, w, scaler, pca)

    t0 = time.time()

    if alg == 'bayes':
        model = naive_bayes.GaussianNB()
    elif alg == 'logistic':
        model = linear_model.LogisticRegression()
    elif alg == 'svm':
        model = svm.SVC(kernel = 'linear', probability=True)   # kernel = linear, poly, rbf, sigmoid, precomputed
    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=7,
            min_samples_split=200, min_samples_leaf=200, max_features=features)
    elif alg == 'forest':
        model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=7,
            min_samples_split=200, min_samples_leaf=200, max_features=features)
    # ------------------------------------------------------------------------------
    # Adding more ensembles by Bernard Ong
    # ------------------------------------------------------------------------------
    elif alg == "dectree":
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=1)
    elif alg == "extratree":
        model = ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_split=1)
    elif alg == "adaboost":
        model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=1)
    else:
        print 'No model defined for ' + alg
        exit()

    model.fit(X, y)

    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)

    return model


def train_xgb(X, y, scaler, pca):
    """
    Trains a boosted trees model using the XGBoost library.
    """
    t0 = time.time()

    xgmat = xgb.DMatrix(X, label=y, missing=-999.0)

    # -------------------------------------------------------------------------------------------------------------------------
    # Optimizing the XGBoost parameters
    # -------------------------------------------------------------------------------------------------------------------------
    # https://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    # -------------------------------------------------------------------------------------------------------------------------
    param = {}
    # objective = reg:linear, reg:logistic, reg:gamma, binary:logistic, binary:logitraw, multi:softmax, multi:softprob
    # objective = count:poisson, rank:pairwise, rank:ndcg, rank:map
    param['objective'] = 'binary:logitraw'     # no change from original setting
    param['eta'] = 0.08                        # (0.01 to 0.20), original 0.08, use small shrinkage
    param['max_depth'] = 7                     # (3 to 10), original 7
    param['subsample'] = 0.5                   # (0.5 t0 1.0), original 0.8
    param['eval_metric'] = 'auc'               # auc, rmse, mae, logloss, error, merror, mlogloss, original auc
    param['silent'] = 1
    nrounds = 250

    plst = list(param.items())
    watchlist = []

    model = xgb.train(plst, xgmat, nrounds, watchlist)

    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)

    return model


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



def cross_validate(X, y, alg, scaler, pca, foldscount, features):
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
        model = train(X_train, y_train, alg, scaler, pca, features)

        # predict and score performance on the validation set
        y_val_prob = predict(X_val, model, alg, scaler, pca)
        scores[i] = score(y_val, y_val_prob)
        i += 1

    return np.mean(scores)


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
    alg = 'boost'  # bayes, logistic, svm, boost, xgboost
    features = 10
    impute = 'none'  # mean, median, most_frequent, knn, interpolate, zeros, none
    standardize = False
    whiten = False
    foldsno = 3
    load_training_data = True
    load_model = False
    train_model = True
    save_model = False
    create_visualizations = False
    create_submission_file = True
    code_dir = './'
    data_dir = './'
    training_file = 'cs-training.csv'
    test_file = 'cs-test.csv'
    submit_file = 'submission.csv'
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
        model = train(X, y, alg, scaler, pca, features)

        print 'Calculating predictions...'
        y_prob = predict(X, model, alg, scaler, pca)

        print 'Calculating AUC...'
        auc_val = score(y, y_prob)
        print 'AUC =', auc_val

        print 'Performing cross-validation...'
        val = cross_validate(X, y, alg, scaler, pca, foldsno, features)
        print'Cross-validation AUC =', val

    if save_model:
        print 'Saving model to disk...'
        save(alg, model, data_dir + model_file)

    if create_submission_file:
        print 'Reading in test data...'
        test_data, X_test = process_data(
            data_dir + test_file, False, features, impute, standardize, whiten, y_column, replace_val, replace_by)

        print 'Predicting test data...'
        y_test_prob = predict(X_test, model, alg, scaler, pca)

        print 'Creating submission file...'
        create_submission(y_test_prob, data_dir + submit_file)

    print 'Process complete.'


if __name__ == "__main__":
    main()
