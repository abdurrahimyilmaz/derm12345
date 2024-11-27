#%%
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# system libraries

import os
from glob import glob
import uuid
from tqdm import tqdm
from datetime import datetime

# numpy and pandas
import pandas as pd
import numpy as np
from numpy import save
from numpy import load

# matplotlib and visualization libs
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import seaborn as sns
import itertools
from PIL import Image

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
#from sklearn.model_selection import cross_val_score

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow import config
from tensorflow.keras.models import model_from_json # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
#from tensorflow.keras.models import Model  # for quadrants
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical  # type: ignore # convert to one-hot-encoding
#from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K # type: ignore
#from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
#from tensorflow.keras.models import load_model
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger # type: ignore

# used for converting labels to one-hot-encoding
from collections import Counter 

np.random.seed(123)

print("Num GPUs Available: ", len(
    config.experimental.list_physical_devices('GPU')))
physical_devices = config.list_physical_devices("GPU")

config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.set_visible_devices(physical_devices[0], 'GPU')
os.getcwd()
#%%  
def plot_confusion_matrix(cm, classes,
                          model_no,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, folder_name = "model_files_folder",**kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(12,12),**kwargs)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{folder_name}/kfold_{model_no}.png")
    plt.show()
   
# %% variables

classes = {
    "acb": 0,       
    "cb":1,       
    "ccb":2,
    "mcb":3,
    "bdb":4,
    "db":5,
    "ajb":6,
    "cjb":7,
    "jb":8,
    "acd":9,
    "cd":10,
    "ccd":11,
    "ajd":12,
    "jd":13,       
    "srjd":14,
    "rd":15,
    "isl":16,
    "ls":17,
    "sl":18,
    "lk":19,
    "sk":20,
    "df":21,
    "ha":22,
    "la":23,
    "pg":24,
    "angk":25,
    "sa":26,
    "ak":27,
    "alm":28,
    "anm":29,
    "lm":30,
    "lmm":31,
    "mel":32,
    "bcc":33,
    "bd":34,
    "ch":35,
    "mpd":36,
    "scc":37,
    "dfsp":38,
    "ks":39,
}

batch_size = 64 # edit your batch size here
loss_function = "categorical_crossentropy" # edit your loss function here
input_shape = (224, 224, 3) # must match with data array shape
num_classes = 40 # number of classes
verbosity = 1
num_folds = 1
note = "40 class" # edit your note here
model_save_path = "MODEL_SAVE_PATH" # edit your model save path here
path = "DATA_PATH" # edit your data path here

# %%

X_train = load(os.path.join(path, "derm12345_train_224_X.npy"))
y_train = load(os.path.join(path, "derm12345_train_224_y.npy"))

X_test = load(os.path.join(path, "derm12345_test_224_X.npy"))
y_test = load(os.path.join(path, "derm12345_test_224_y.npy"))


# %%
# Get label column, you can edit here based on your research
y_train = np.delete(y_train, 1, 1) # column_index, column-wise
y_test = np.delete(y_test, 1, 1) # column_index, column-wise
y_train = np.delete(y_train, 1, 1) # column_index, column-wise
y_test = np.delete(y_test, 1, 1) # column_index, column-wise
y_train = np.delete(y_train, 1, 1) # column_index, column-wise
y_test = np.delete(y_test, 1, 1) # column_index, column-wise
        
y_train = y_train.ravel()
y_train = y_train.astype(int)
y_test = y_test.ravel()
y_test = y_test.astype(int)

print(f"Data shape: {X_train.shape}")
print(f"Target shape: {y_train.shape}")
print(f"Data shape: {X_test.shape}")
print(f"Target shape: {y_test.shape}")

# %%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)

# free up memory
# del x_data
# del y_data
# del X_train
# del Y_train

print(f"Train size: {len(y_train)}")
print(f"Val size: {len(y_val)}")
print(f"Test size: {len(y_test)}")
print(f"Total size: {len(y_train) + len(y_val) + len(y_test)}")

X_train = X_train.reshape(X_train.shape[0], * input_shape)
X_val = X_val.reshape(X_val.shape[0], * input_shape)
X_test = X_test.reshape(X_test.shape[0], * input_shape)

# Parse numbers as floats
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

# %%
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# for different validation methods, not required for holdout validation
val_acc_per_fold = []
val_loss_per_fold = []

test_acc_per_fold_eva = []
test_loss_per_fold_eva = []

test_acc_per_fold_pred = []
test_loss_per_fold_pred = []

# %%
train_datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=45,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.2,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.2,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

train_datagen.fit(X_train)

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0.0, amsgrad=False)

#%%

fold_name = "holdout"

from timeit import default_timer as timer
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

total_duration = 0
metrics = pd.DataFrame(columns = ['metric', "value"])
metrics.loc[len(metrics)] = ['fold_name', fold_name]

timing_callback = TimingCallback()

os.chdir(model_save_path)

base_model = keras.applications.ResNet50( # edit here to try different models
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=input_shape,
    include_top=False)

folder_name = "model_files_folder" + " - " + str(datetime.today().strftime('%Y-%m-%d %H.%M.%S'))
os.makedirs(folder_name)
   
# callbacks part
mc_path = f"{folder_name}/{base_model.name}_holdout-model.hdf5"
   
model_checkpoint = ModelCheckpoint(filepath = mc_path,
                                monitor ='val_accuracy',#'val_recall',#'val_accuracy',
                                mode = 'max',
                                verbose = 1,
                                save_best_only = True,
                                save_weights_only = True)
   
   
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',#'val_recall',#'val_accuracy',
                                        patience=5,
                                        verbose=1,
                                        factor=0.1,
                                        min_lr=0.00001)
   
log_csv = CSVLogger(f'{folder_name}/{base_model.name}_holdout-model.csv', separator = ',', append = False)


callbacks_list = [learning_rate_reduction, model_checkpoint, log_csv, timing_callback]
   
base_modelQ4 = base_model
   
base_modelQ4.trainable = False
inputs = keras.Input(shape=input_shape)

# edit your model
q4 = base_modelQ4(inputs, training=False)
q4 = keras.layers.GlobalAveragePooling2D()(q4)
q4 = keras.layers.Dropout(0.2)(q4)
q4 = keras.layers.Flatten()(q4)
q4 = keras.layers.Dense(256, activation='relu')(q4)
q4 = keras.layers.Dropout(0.2)(q4)
q4 = keras.layers.Dense(128, activation='relu')(q4)
q4 = keras.layers.Dropout(0.3)(q4)
q4 = keras.layers.Dense(64, activation='relu')(q4)
q4 = keras.layers.Dropout(0.3)(q4)
    
outputs = keras.layers.Dense(num_classes, activation='softmax')(q4)
modelQ4 = keras.Model(inputs, outputs)
   
   
modelQ4.compile(loss= loss_function,
                metrics=["accuracy"],
                #metrics= [
        #keras.metrics.Recall(name='recall')
        #keras.metrics.FalseNegatives(),
    #],
                optimizer=optimizer)
   
epochs = 1 # before fine tuning
modelQ4.fit(train_datagen.flow(X_train, y_train, batch_size = batch_size),
                        epochs=epochs,
                        validation_data = (X_val, y_val),
                        verbose=1,
                        steps_per_epoch = X_train.shape[0] // batch_size,
                        callbacks = [learning_rate_reduction, timing_callback])

#total_duration += sum(timing_callback.logs)

# Fine Tuning
base_modelQ4.trainable = True
modelQ4.compile(loss= loss_function,
                metrics=["accuracy"],
                #metrics= [
        #keras.metrics.Recall(name='recall')
        #keras.metrics.FalseNegatives(),
   #],
                optimizer = optimizer)


epochs = 1 # for fine tuning
modelQ4.fit(train_datagen.flow(X_train, y_train, batch_size = batch_size),
                        epochs = epochs,
                        validation_data = (X_val, y_val),
                        verbose=1,
                        steps_per_epoch= X_train.shape[0] // batch_size,
                        callbacks = callbacks_list)

total_duration += sum(timing_callback.logs)
    
#modelQ4.save(f'data_models/224x224_bs8_300fine_{model_no}.h5', overwrite=True)
# serialize model to JSON
classifier_json = modelQ4.to_json()
with open(f"{folder_name}/{base_model.name}_holdout-model.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
modelQ4.save_weights(f"{folder_name}/{base_model.name}_holdout-model.h5")
print("Saved weights to disk")
   
    # Generate a print
print('------------------------------------------------------------------------')
print(f'Training for holdout ...')
     
json_file = open(f'{folder_name}/{base_model.name}_holdout-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"{folder_name}/{base_model.name}_holdout-model.hdf5")
print("Loaded model from disk")

loaded_model.compile(loss= loss_function,
                metrics=["accuracy"],
                #metrics= [
        #keras.metrics.Recall(name='recall')
        #keras.metrics.FalseNegatives(),
    #],
                optimizer = optimizer)


# Generate generalization metrics
scores = loaded_model.evaluate(X_val, y_val, verbose = 0)
print(f'Score for holdout: {loaded_model.metrics_names[0]} of {scores[0]}; {loaded_model.metrics_names[1]} of {scores[1]*100}%\n\n')
val_acc_per_fold.append(scores[1])
val_loss_per_fold.append(scores[0])

metrics.loc[len(metrics)] = ['val_acc_per_fold', scores[1] * 100]
metrics.loc[len(metrics)] = ['val_loss_per_fold', scores[0]]
   
test_loss_eval, test_accuracy_eval = loaded_model.evaluate(X_test, y_test, verbose = 0)
print("Eval: accuracy = %f  ;  loss = %f" % (test_accuracy_eval, test_loss_eval))
metrics.loc[len(metrics)] = ['test_acc_eval', test_accuracy_eval]
metrics.loc[len(metrics)] = ['test_loss_eval', test_loss_eval]
        
test_acc_per_fold_eva.append(test_accuracy_eval)
test_loss_per_fold_eva.append(test_loss_eval)    


Y_pred = loaded_model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=classes, model_no = 1, folder_name = folder_name)  

test_accuracy_pred = "%.6f" % accuracy_score(Y_true, Y_pred_classes)

metrics.loc[len(metrics)] = ["test_accuracy", "%.6f" % accuracy_score(Y_true, Y_pred_classes)]
metrics.loc[len(metrics)] = ["micro_precision", "%.6f" % precision_score(Y_true, Y_pred_classes, average='micro')]
metrics.loc[len(metrics)] = ["micro_recall", "%.6f" % recall_score(Y_true, Y_pred_classes, average='micro')]
metrics.loc[len(metrics)] = ["micro_f1", "%.6f" % f1_score(Y_true, Y_pred_classes, average='micro')]
metrics.loc[len(metrics)] = ["macro_precision", "%.6f" % precision_score(Y_true, Y_pred_classes, average='macro')]
metrics.loc[len(metrics)] = ["macro_recall", "%.6f" % recall_score(Y_true, Y_pred_classes, average='macro')]
metrics.loc[len(metrics)] = ["macro_f1", "%.6f" % f1_score(Y_true, Y_pred_classes, average='macro')]
metrics.loc[len(metrics)] = ["weighted_precision", "%.6f" % precision_score(Y_true, Y_pred_classes, average='weighted')]
metrics.loc[len(metrics)] = ["weighted_recall", "%.6f" % recall_score(Y_true, Y_pred_classes, average='weighted')]
metrics.loc[len(metrics)] = ["weighted_f1", "%.6f" % f1_score(Y_true, Y_pred_classes, average='weighted')]

print(f"\nAccuracy: {metrics[metrics['metric'] == 'test_accuracy'].values[0][1]}\n")

print(f"Micro Precision: {metrics[metrics['metric'] == 'micro_precision'].values[0][1]}")
print(f"Micro Recall: {metrics[metrics['metric'] == 'micro_recall'].values[0][1]}")
print(f"Micro F1: {metrics[metrics['metric'] == 'micro_f1'].values[0][1]}")

print(f"Macro Precision: {metrics[metrics['metric'] == 'macro_precision'].values[0][1]}")
print(f"Macro Recall: {metrics[metrics['metric'] == 'macro_recall'].values[0][1]}")
print(f"Macro F1: {metrics[metrics['metric'] == 'micro_f1'].values[0][1]}")

print(f"Weighted Precision: {metrics[metrics['metric'] == 'weighted_precision'].values[0][1]}")
print(f"Weighted Recall: {metrics[metrics['metric'] == 'weighted_recall'].values[0][1]}")
print(f"Weighted F1: {metrics[metrics['metric'] == 'weighted_f1'].values[0][1]}")

test_acc_per_fold_pred.append(test_accuracy_pred)    

metrics.loc[len(metrics)] = ["total_duration_sec", int(total_duration)]
metrics.loc[len(metrics)] = ["total_duration_min", int(total_duration / 60)]

print('\nClassification Report\n')
report = classification_report(Y_true, Y_pred_classes, target_names= classes, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()
metrics = pd.concat([metrics, report_df])
print(classification_report(Y_true, Y_pred_classes, target_names= classes, digits=4))

print(f"Total duration in second: {metrics[metrics['metric'] == 'total_duration_sec'].values[0][1]}")
print(f"Total duration in minute: {metrics[metrics['metric'] == 'total_duration_min'].values[0][1]}")

metrics.to_csv(f"{folder_name}/metrics.csv", index = True)

results_df = pd.DataFrame(Y_true, columns=['Original'])
results_df.insert(1, "Pred", Y_pred_classes)

results_df.to_csv(f"{folder_name}/original_predictions.csv", index=False)

cm_df = pd.DataFrame(confusion_mtx)

cm_df.to_csv(f"{folder_name}/confusion_matrix.csv", index=True)

with open(f"{folder_name}/notes.txt", "w") as text_file:
    print(f"{note}", file=text_file)
