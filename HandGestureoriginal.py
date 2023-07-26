from enum import auto
import os
import cv2
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Convolution2D
from keras.utils import np_utils


# def preprocess_dataset(dataset_path):
#     images = []
#     labels = []

#     # Traverse through each folder (letter) in the dataset path
#     for folder_name in sorted(os.listdir(dataset_path)):
#         label = int(ord(folder_name) - ord('A'))  # Convert folder name to label (0 to 25)
#         folder_path = os.path.join(dataset_path, folder_name)

#         # Traverse through each image file in the current folder
#         for image_name in os.listdir(folder_path):
#             image_path = os.path.join(folder_path, image_name)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
#             image = cv2.resize(image, (50, 50))  # Resize the image to 50x50 pixels
#             images.append(image)
#             labels.append(label)

#     images = np.array(images)
#     labels = np.array(labels)

#     # Normalize the image pixel values to the range of 0-1
#     images = images.astype('float32') / 255.0

#     # Reshape the image data to match the input shape expected by the CNN
#     images = np.reshape(images, (images.shape[0], 50, 50, 1))

#     # Convert the labels to one-hot encoded format
#     num_classes = 26  # We have 26 classes (letters A to Z)
#     labels = np_utils.to_categorical(labels, num_classes)

#     return images, labels

def main():
    # model = create_model()
    model = load_model('hand_gesture_model.h5')  # Load pre-trained weights

    # Open the webcam
    cap = cv2.VideoCapture(0)
    # LIGHT SLIDER WITH EXAMPLE GESTURES
    im = cv2.imread("template.png")         

    def BrightnessContrast(brightness=100):
        brightness = cv2.getTrackbarPos('Lighting', 'Light-Slider')
        return brightness
    cv2.namedWindow('Light-Slider')
    cv2.createTrackbar('Lighting', 'Light-Slider', 0, 300, BrightnessContrast)
    cv2.imshow("Light-Slider", im)
    cv2.setWindowProperty(
        "Light-Slider", cv2.WND_PROP_TOPMOST, cv2.WND_PROP_VISIBLE)
    cv2.moveWindow("Light-Slider", 1100, 100)

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        frame = cv2.flip(frame, 1)
        # Preprocess the frame

        # ret, frame = cap.read()
        # frame = cv2.flip(frame,1)
        # frame=cv2.resize(frame,(321,270))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # x1,y1 x2,y2 color(0,255,0)
        img1 = cv2.rectangle(frame, (300, 0), (600, 300),
                             (0, 255, 0), thickness=2, lineType=8, shift=0)

# TRACKBAR
        brightness = 100
        brightness = BrightnessContrast(brightness)

        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([179, 255, brightness])
        imcrop = img1[2:298, 302:598]
        # y1:y2, x1:x2
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", mask)
        cv2.setWindowProperty(
            "mask", cv2.WND_PROP_TOPMOST, cv2.WND_PROP_VISIBLE)
        cv2.resizeWindow("mask", 200, 200)
        cv2.moveWindow("mask", 900, 100)
        gray = cv2.resize(mask, (50, 50))
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        input_data = np.expand_dims(gray, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)

        # Make predictions
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions)
        predicted_class = chr(predicted_class+ord('A'))
        # Display the predicted class on the frame

        cv2.putText(frame, predicted_class,
                    (302, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('Hand Gesture Recognition', frame)
        cv2.imshow("Original", img1)
        cv2.setWindowProperty(
            "Original", cv2.WND_PROP_TOPMOST, cv2.WND_PROP_VISIBLE)
        cv2.moveWindow("Original", 200, 100)

        # slider1=self.trackbar.value()
        # height1, width1, channel1 = img1.shape
        # step1 = channel1 * width1
        # create QImage from image
        # qImg1 = QImage(img1.data, width1, height1, step1, QImage.Format_RGB888)
        # show image in img_label
        # try:
        #     self.label_3.setPixmap(QPixmap.fromImage(qImg1))
        #     slider1=self.trackbar.value()
        # except:
        #     pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
