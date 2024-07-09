import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models


def load_data_and_normalization():
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalization
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# Reshape the data 2d to 3d. x_train = x_train.reshape((-1, 28, 28, 1)) x_test = x_test.reshape((-1, 28, 28, 1))
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
CNN = models.Sequential([
Input(shape=(28, 28, 1)), layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), #layers.Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid'), #layers.MaxPooling2D((2, 2)), layers.Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid'), layers.MaxPooling2D((2, 2)), #layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'), #layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')
])
#Testing the model with different optimizers and loss functions
#CNN.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])#accuracy: 0.8891 -
loss: 0.4281 - val_accuracy: 0.9101 - val_loss: 0.3431
#CNN.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])#accuracy: 0.1145 - loss: 0.0906 - val_accuracy: 0.1060 - val_loss: 0.0906
#SGD is not a good optimizer for this model
#CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )#accuracy:
0.9923 - loss: 0.0259 - val_accuracy: 0.9834 - val_loss: 0.0584
CNN.compile(optimizer='adam', loss='mse', metrics=['accuracy'])#accuracy: 0.9892 - loss: 0.0018 - val_accuracy: 0.9832 - val_loss: 0.0026
#Adam is a good optimizer for this model
#CNN.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )#accuracy:
0.9892 - loss: 0.0336 - val_accuracy: 0.9858 - val_loss: 0.0505
#CNN.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'] )#accuracy: 0.9791 - loss: 0.0034 - val_accuracy: 0.9745 - val_loss: 0.0038
#RMSprop is a good optimizer for this model too but has a little bit lower accuracy than Adam
return CNN

def precision_from_confusion_matrix(Conf_mat): num_class = Conf_mat.shape[0]
precision = np.zeros(num_class)
for i in range(num_class):
true_positives = Conf_mat[i, i]
false_positives = np.sum(Conf_mat[:, i]) - true_positives
precision[i] = true_positives / (true_positives + false_positives) \
if (true_positives + false_positives) != 0 else 0
return precision

def recall_from_confusion_matrix(Conf_mat): num_classes = Conf_mat.shape[0]
recall = np.zeros(num_classes)
for i in range(num_classes):
true_positives = Conf_mat[i, i]
false_negatives = np.sum(Conf_mat[i, :]) - true_positives
recall[i] = true_positives / (true_positives + false_negatives) \
if (true_positives + false_negatives) != 0 else 0
return recall

# Calculating the value of F1 Score
def f1_score_from_confusion_matrix(Conf_mat): num_classes = Conf_mat.shape[0]
f1_scores = np.zeros(num_classes)
for i in range(num_classes):
true_positives = Conf_mat[i, i]
false_positives = np.sum(Conf_mat[:, i]) - true_positives
false_negatives = np.sum(Conf_mat[i, :]) - true_positives
precision = true_positives / (true_positives + false_positives) \
if (true_positives + false_positives) != 0 else 0
recall = true_positives / (true_positives + false_negatives) \
if (true_positives + false_negatives) != 0 else 0
f1_scores[i] = 2 * (precision * recall) / (precision + recall) \
if (precision + recall) != 0 else 0
return f1_scores

# Calculating Macroaverage of the Precision
def macro_average_precision_score(Precision): num_class = len(Precision) macro_average_P = sum(Precision) / num_class
return macro_average_P

# Calculating Macroaverage of the Recall
def recall_macro_average(Recall): num_class = len(Recall) macro_average_R = sum(Recall) / num_class
return macro_average_R

# Calculating Macroaverage of the F1_Score
def F1_macro_average(Precision, Recall):
total_precision = sum(Precision)
total_recall = sum(Recall) macro_precision = total_precision / len(Precision) macro_recall = total_recall / len(Recall) macro_average_F1= (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
return macro_average_F1


(x_train, y_train), (x_test, y_test) = load_data_and_normalization()
CNN = create_cnn_model()
CNN.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
test_loss, test_acc = CNN.evaluate(x_test, y_test, verbose=0)
y_predict = CNN.predict(x_test)
y_predict_classes = np.argmax(y_predict, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_true, y_predict_classes)
print("Accuracy:\n", accuracy)
Conf_mat = confusion_matrix(y_true, y_predict_classes)
print("Confusion Matrix:\n", Conf_mat)
Precision = precision_from_confusion_matrix(Conf_mat)
print("Precision:\n", Precision)
Recall = recall_from_confusion_matrix(Conf_mat)
print("Recall:\n", Recall)
F1_Score = f1_score_from_confusion_matrix(Conf_mat)
print("F1 Score:\n", F1_Score)
Macro_average_P = macro_average_precision_score(Precision)
print("Macroaverage_Precision:\n", Macro_average_P)
Macro_average_R = recall_macro_average(Recall)
print("Macroaverage_Recall:\n", Macro_average_R)
Macro_average_F1_score = F1_macro_average(Precision, Recall)
print("Macro_average_F1_score:\n", Macro_average_F1_score)
