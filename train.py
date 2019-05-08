# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from cnn_models.vgg_16_lite import VGG_16_Lite
# from cnn_models.vgg_13_lite import VGG_13_Lite
from cnn_models.vgg_11_lite import VGG_11_Lite
# from pyimagesearch.smallervggnet import SmallerVGGNet
from TimeHistory import TimeHistory
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# add graphviz to path (used to visualize model)
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (2*96, 2*96, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = VGG_11_Lite.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# print network architecture
plot_model(model, to_file='Model.png', show_shapes=True, show_layer_names=True)

# callback to record epochs time
time_callback = TimeHistory()

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1,
	callbacks=[time_callback])

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Training Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("Model Loss")

plt.figure()
plt.plot(np.arange(0, N), H.history["acc"], label="Training Accuracy")
plt.plot(np.arange(0, N), H.history["val_acc"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.savefig("Model Accuracy")


# stats
loss = np.asarray(H.history["loss"])
acc = np.asarray(H.history["acc"])
val_loss = np.asarray(H.history["val_loss"])
val_acc = np.asarray(H.history["val_acc"])
times = time_callback.times
tot_time = sum(times)

# record the epoch times
epoch_meta_file = open("Epoch_Meta.txt","w+")
epoch_meta_file.write("time, loss, val_loss, acc, val_acc\n")
for i in range(len(time_callback.times)):
	epoch_meta_file.write("%d, %f, %f, %f, %f,\n" % (times[i], loss[i], val_loss[i], acc[i], val_acc[i]))
epoch_meta_file.close()

epoch_summary_file = open("Epoch_Summary.txt","w+")
epoch_summary_file.write("Total Time = %d\n" % (tot_time))
epoch_summary_file.write("Average/Epoch = %f\n" % (tot_time/EPOCHS))
epoch_summary_file.write("Loss Error Mean = %f\n" % (np.mean(val_loss - loss)))
epoch_summary_file.write("Loss Error Variance = %f\n" % (np.var(val_loss- loss)))
epoch_summary_file.write("Accuracy Error Mean = %f\n" % (np.mean(val_acc - acc)))
epoch_summary_file.write("Accuracy Variance = %f\n" % (np.var(val_acc - acc)))
epoch_summary_file.close()