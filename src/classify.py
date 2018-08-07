# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/example_01.jpg

import pickle

import cv2
import imutils
import numpy as np
from keras.models import load_model
# import the necessary packages
from keras.preprocessing.image import img_to_array

# imagePath = "test/1.jpg"
imagePath = "dataset/consonants/5/001_02.jpg"

# load the image
image = cv2.imread(imagePath)
output = imutils.resize(image, width=250)


# Contour detection
def bw_scale(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #  Apply threshold to get image with only black and white
    img = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img, 7)
    img = cv2.medianBlur(img, 7)
    img = cv2.adaptiveThreshold(img, 255, cv2.THRESH_BINARY, cv2.THRESH_BINARY, 31, 2)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Apply dilation and erosion to remove some noise
    return img, thresh


def edge_detect(file_name):
    (img, thresh) = bw_scale(file_name)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours


# img, con = edge_detect(imagePath)
# image = img
# cv2.drawContours(img, con, -1, (0, 255, 0), 3)
# img = cv2.resize(img, (250, 250))
# cv2.imshow("Contour Image", img)

image = cv2.resize(image, (28, 28))
# cv2.imshow("Output Image", image)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model("model.h5")
mlb = pickle.loads(open("mlb.pickle", "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
    print("{}: {:.2f}%".format(label, p * 100))

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    print("Highest Chances::: ", label)
    cv2.putText(output, label, (10, (i * 30) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
