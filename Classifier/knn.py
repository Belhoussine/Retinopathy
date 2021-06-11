from sklearn.model_selection import train_test_split
from skimage.io import imread_collection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import cv2
from PIL import Image

def toImage(matrix):
    return Image.fromarray(matrix.astype('uint8'))

def image_to_feature_vector(image, size=(750, 750)):
    return cv2.resize(image, size)

drPath = "../ProcessedImages/DR/*.jpg"
healthyPath = "../ProcessedImages/Healthy/*.jpg"

# drPath = "./DR/*.jpg"
# healthyPath = "./Healthy/*.jpg"

drCollection = imread_collection(drPath)
healthyCollection = imread_collection(healthyPath)

# Get image dataset + labels
dataset = np.concatenate((drCollection, healthyCollection))
labels = np.array([0 for i in range(len(drCollection))] +
                  [1 for i in range(len(healthyCollection))])

dataset = np.array(list(map(image_to_feature_vector, dataset)))
x_train, x_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.25, random_state=42)

size = x_train[0].shape[0] * x_train[0].shape[1]
# Build Input Vector
x_train = x_train.reshape(x_train.shape[0], size)
x_test = x_test.reshape(x_test.shape[0], size)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the data to help with the training
x_train /= 255
x_test /= 255

# KNN
k_range = range(1, 5)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))
    print(f"{k}: {scores[k]:.2%}")
print(scores_list)