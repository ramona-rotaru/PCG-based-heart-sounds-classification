import os
import numpy as np
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/' 
from preprocessing_functions import *
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, GlobalAveragePooling1D,  Embedding,Dropout, Flatten, Conv2D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
import datetime
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


from tensorflow.python.platform import build_info as tf_build_info
tf_build_info.build_info

from platform import python_version
print(python_version())

print(tf.__version__)



def model_conv(input_shape, num_classes):
    # Create a Sequential model
    model = Sequential()


    # Add a 1D Convolution layer with ReLU activation
    model.add(Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(input_shape, 1)))
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=10, activation='relu'))

    
    # Add GlobalAveragePooling layer
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(rate=0.5))
    model.add(Flatten())

    # Add a Dense layer with Softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Print the model summary
    model.summary()
    return model




def get_tensorboard_callback(model_name):
    #SAVE LOG AS MODEL NAME PLUS DATE AND TIME
    log_dir = "logs/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback



set = 'a'

matrix = np.load('./dala_labels_multiclass/a/array_matrix_'+ set +'.npy')
labels = np.load('./dala_labels_multiclass/a/labels_'+ set +'.npy')

print(np.unique(labels), 'labels uniq')
num_classes = len(np.unique(labels))

print(matrix.shape, 'matrix shape')


# Instantiate the encoder
encoder = LabelEncoder()

# Fit the encoder and transform labels
labels = encoder.fit_transform(labels)

# Now you can convert to categorical
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(matrix, labels, test_size=0.2, random_state=33, shuffle=True, stratify=labels)

print(x_train.shape, 'x_train shape')
print(y_train.shape, 'y_train shape')
print(x_test.shape, 'x_test shape')
print(y_test.shape, 'y_test shape')

model = model_conv(x_train.shape[1], num_classes=num_classes)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



save_model_path = './dala_labels_multiclass/models_multiclass/conv1D_final_a.h5'

# model.fit(x_train, y_train, batch_size=100, epochs=350, verbose=2, callbacks=[get_tensorboard_callback(str(save_model_path)), tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=50, verbose=0, mode="auto", baseline=None, restore_best_weights=False), ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(x_test, y_test))


load_model_path = save_model_path 
model = tf.keras.models.load_model(load_model_path)
model.summary()
print(model.evaluate(x_test, y_test, verbose=2))

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes, normalize='true')

# Calculate precision, recall, and F1-score with 'micro' averaging
precision = precision_score(y_test_classes, y_pred_classes, average='micro')
recall = recall_score(y_test_classes, y_pred_classes, average='micro')
f1 = f1_score(y_test_classes, y_pred_classes, average='micro')

print("Precision:", precision)
print("Recall:", recall)

# Plot confusion matrix with additional metrics
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='.4f', xticklabels=encoder.classes_, yticklabels=encoder.classes_, annot_kws={"size": 14})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)

# Save the figure
# plt.savefig('confusion_matrix_plot.png', dpi=300, bbox_inches='tight')
plt.show()