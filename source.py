# Daniel Luper 2018
# Program that loads 97.82% accuracy model
# located in the working directory
# and lets the user create its own handwritten digits to be tested

print("Loading, this may take a few minutes...")

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import os

image = np.zeros((28, 28), np.uint8)
white = 255
drawing = False
has_clicked = False

# Load data
mnist = input_data.read_data_sets("MNIST/", one_hot=False)
mnist_data = np.reshape(mnist.test.images, (-1, 28, 28))

# Load model
model = tf.keras.models.load_model('neural-network.model')
mnist_predictions = model.predict([mnist_data])

# Clear screen
clear = lambda: os.system('cls')
clear()


def predict(img):
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    print(np.argmax(prediction))
    print("")

def mouse_draw(event, x, y, flags, params):
    global drawing, has_clicked, white, image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        has_clicked = True
        cv2.rectangle(image, (x-1, y-1), (x+1,y+1), (0,0,0), -1)
        
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            image[y,x] = white

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False


def read_from_user():
    # Setup
    global image, drawing, has_clicked
    cv2.namedWindow("Draw here", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Draw here", 280, 280)
    cv2.setMouseCallback("Draw here", mouse_draw)
    cv2.imshow("Draw here", image)
    print("HELP:\n")
    print("Spacebar to reset the window")
    print("Hold left click to draw a number (0-9)")
    print("Please draw as neatly as possible for best results...\n")
    print("NB : Sometimes the number 8 has a hard time being read by the software")
    print("Try to make it a bit bigger\n")

    # Update loop
    while cv2.getWindowProperty('Draw here', 0) >= 0:
        cv2.imshow("Draw here", image)
        if has_clicked:
            predict(image)
        k = cv2.waitKey(10)
        if k == 27:
            break
        elif k == 32:
            image = np.zeros((28, 28), np.uint8)

    # Cleanup
    has_clicked = False
    image = np.zeros((28, 28), np.uint8)
    cv2.destroyAllWindows()
    clear()


def read_from_database():
    # Get input In order to select one of the 10,000 images in the database,
    print("In order to select one of the 10,000 images in the database,")
    s = input("enter a number between 0 and 9,999: ")
    n = int(s)
    if n < 0 or n > 9999:
        print("\nInvalid input. Please try again. \n\n")
        return
    
    # Render image
    image = (mnist_data[n] * 255).astype(np.uint8)
    winName = "Image " + s
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, image)
    
    print("\nThis is a " + str(np.argmax(mnist_predictions[n])) + "\n")

    while cv2.waitKey(10) != 27 and cv2.getWindowProperty(winName, 0) >= 0:
        continue
    cv2.destroyAllWindows()
    clear()

# MAIN LOOP
while True:
    print("0 -> Exit the program")
    print("1 -> Test with pre-installed MNIST image")
    print("2 -> Create your own image\n")
    command = input("Enter a number: ")
    print("\n")
    if command == "0":
        break
    elif command == "1":
        read_from_database()
    elif command == "2":
        read_from_user()
    else:
        print("Invalid input. Please try again. \n\n")

cv2.destroyAllWindows()




