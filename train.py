import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils


def preprocess_dataset(dataset_path):
    images = []
    labels = []

    # Traverse through each folder (letter) in the dataset path
    # i=0
    for folder_name in sorted(os.listdir(dataset_path)):
        label = int(ord(folder_name) - ord('A'))  # Convert folder name to label (0 to 25)
        # label=i
        # i=i+1
        folder_path = os.path.join(dataset_path, folder_name)

        # Traverse through each image file in the current folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
            image = cv2.resize(image, (50, 50))  # Resize the image to 50x50 pixels
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Normalize the image pixel values to the range of 0-1
    images = images.astype('float32') / 255.0

    # Reshape the image data to match the input shape expected by the CNN
    images = np.reshape(images, (images.shape[0], 50, 50, 1))

    # Convert the labels to one-hot encoded format
    num_classes = 26  # We have 26 classes (letters A to Z)
    labels = np_utils.to_categorical(labels, num_classes)

    return images, labels


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(26, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(train_images,train_labels,test_images,test_labels,model):
    model.fit(train_images,train_labels, validation_split=0.2, epochs=10, batch_size=32)

    # Evaluate the model on the entire dataset
    loss, accuracy = model.evaluate(test_images,test_labels)
    print(f"Training loss: {loss:.4f}")
    print(f"Training accuracy: {accuracy:.4f}")


def main():
    train_path='Dataset/training_set/'
    test_path='Dataset/test_set/'
    train_images,train_labels=preprocess_dataset(train_path)
    test_images,test_labels=preprocess_dataset(test_path)
    model = create_model()
    train_model(train_images,train_labels,test_images,test_labels,model)
    # save the model weights
    model.save('hand_gesture_model.h5')

main()
