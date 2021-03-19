''' Computes probabilities for HPC model '''

# downloading model for transfer learning
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from sklearn.svm import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

import pandas as pd
import numpy as np

import argparse


#################################
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
from sklearn.metrics import classification_report
from keras.models import Model

#####################################
# ----------------
META_AVG = 'avg'
META_STD = 'std'

# ----------------


def get_full_rbf_svm_clf(train_x, train_y, c_range=None, gamma_range=None):
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(cache_size=1024),
                        param_grid=param_grid, cv=cv, n_jobs=14, verbose=10)
    grid.fit(train_x, train_y)

    print("The best parameters are %s with a score of %0.2f" %
          (grid.best_params_, grid.best_score_))

    scores = grid.cv_results_['mean_test_score'].reshape(
        len(c_range), len(gamma_range))
    print("Scores:")
    print(scores)

    print("c_range:", c_range)
    print("gamma_range:", gamma_range)

    c_best = grid.best_params_['C']
    gamma_best = grid.best_params_['gamma']

    clf = SVC(C=c_best, gamma=gamma_best, verbose=True)
    return clf

# ----------------


def prep(data):
    data[feats] = (data[feats] - meta[META_AVG]) / meta[META_STD]  # normalize
    data.fillna(0, inplace=True)							# impute NaNs with mean=0

    if '_count' in data.columns:
        data.drop('_count', axis=1, inplace=True)

    data_x = data.iloc[:, 0:-1].astype('float32').values
    data_y = data.iloc[:,   -1].astype('int32').values

    return data_x, data_y


# ----------------
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True, help="dataset name")
parser.add_argument('-svmgamma', type=float, help="SVM gamma parameter")
parser.add_argument('-svmc', type=float, help="SVM C parameter")

args = parser.parse_args()

DATASET = args.dataset

DATA_FILE = '../data/' + DATASET + '-train'
VAL_FILE = '../data/' + DATASET + '-val'
TEST_FILE = '../data/' + DATASET + '-test'
META_FILE = '../data/' + DATASET + '-meta'
HPC_FILE = '../data/' + DATASET + '-hpc'

print("Using dataset", DATASET)
# ----------------

data_train = pd.read_pickle(DATA_FILE)
data_val = pd.read_pickle(VAL_FILE)
data_test = pd.read_pickle(TEST_FILE)
meta = pd.read_pickle(META_FILE)

feats = meta.index


train_x, train_y = prep(data_train)
val_x, val_y = prep(data_val)
test_x, test_y = prep(data_test)

# print("\n##########Feature extraction#################\n\n")
# Layer_Feature = 'block4_conv4'
# model_vgg19 = VGG19(include_top=True, weights=None)
# optimizer = Adam(lr=0.0001)

# arg_model = Model(inputs=model_vgg19.input,
#                 outputs=model_vgg19.get_layer(Layer_Feature).output)

# arg_model.compile(loss='categorical_crossentropy',
#                 metrics=['accuracy'])
# optimizer = optimizer,
# bottleneck_train1 = arg_model.predict(
#     preprocess_input(train_x), batch_size=5, verbose=1)
# print("\n----bottleneck-------\n")
# print(bottleneck_train1)

# print("###########################################################################\n\n")
# print(val_y, test_y)



# directory1 = "/home/nav/Downloads/data-rw"
# X_test_features = np.load(directory1+'/X_test_features_b4C4-003.npy')
# X_train_features = np.load(directory1+'/X_train_features_b4C4-005.npy')

# y_test_1dim=np.load(directory1+'/Y_test_1dim.npy')
# y_train_1dim=np.load(directory1+'/Y_train_1dim.npy')




# print(y_test_1dim.shape)
# print(y_train_1dim.shape)
# print(X_test_features.shape)
# print(X_train_features.shape)





print('*'*40)

# print('train_x : ',train_x.shape,'\t', '',train_y.shape, val_x.shape,
#     val_y.shape, test_x.shape, test_y.shape)


if args.svmgamma is not None and args.svmc is not None:
    model = SVC(C=args.svmc, gamma=args.svmgamma, cache_size=4096)
else:
    print("Searching for hyperparameters...")
    c_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-5, 1, 7)
    model = get_full_rbf_svm_clf(
        train_x, train_y, c_range=c_range, gamma_range=gamma_range)

print("Training...")
model.fit(train_x, train_y)

# ----------------
print("Trn score:  {:.4f}".format(model.score(train_x, train_y)))
print("Val score:  {:.4f}".format(model.score(val_x, val_y)))
print("Tst score:  {:.4f}".format(model.score(test_x, test_y)))

# ----------------
print("\nSaving...")
train_p = model.predict(train_x)
val_p = model.predict(val_x)
test_p = model.predict(test_x)

# --Checking accuracy with Confusion matrix-----------------------
print("\n\n ------Generating Confusion matrix-------------\n")

cm_x = metrics.confusion_matrix(test_y, test_p)
print(cm_x)
print("Accuracy = ", metrics.accuracy_score(test_p, test_y)*100)
print("\n------------***********----------------------\n\n")
# -------------------------------------------------------------------

data_p = pd.DataFrame(data=[train_p, val_p, test_p], index=[
    'train', 'validation', 'test']).transpose()
data_p.to_pickle(HPC_FILE)
