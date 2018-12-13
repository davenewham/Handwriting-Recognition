#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

mnist = fetch_mldata('MNIST original')
X = mnist['data']
y = mnist['target']

# train the data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]
classifier = SGDClassifier(random_state=42)
classifier.fit(X_train, y_train)

some_index = 56783

print('the value of random is', classifier.predict([X[some_index]]))
# this does some pretty important stuff
y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=4)
conf_mx = confusion_matrix(y_train, y_train_pred)
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print("trained data ");

# Adjust the input to make it similar to the mnist samples
def adjust(image):
    # build a lookup table
    T = 100;  # Threshold
    table = np.arange(0, 256)
    for i in table:
        if (i > T):
            table[i] = 0
        else:
            table[i] = -255 * i * i / (T * T) + 255
    table = table.astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


while (cv2.waitKey(1) & 0xFF != ord('q')):
    th = cv2.imread('two.png', 0)
    mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, se2)
    image, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    sample = adjust(image);
    cv2.imshow("crop_img", sample)
    pred_sample = cv2.resize(sample, (28, 28))
    predction = classifier.predict(pred_sample.reshape(1, 28 * 28))
    print(np.array2string(predction))
    #cv2.imshow("bw", mask)

cv2.destroyAllWindows()
