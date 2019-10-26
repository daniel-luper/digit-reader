# Digit Reader
Python program that can "read" digits thanks to deep learning.

## Demo
<img src=https://media.giphy.com/media/kydGUeVcJW7aqT8sJC/giphy.gif width=576 height=324> <img src=https://media.giphy.com/media/mCJT4PppHliU6ZZ2Be/giphy.gif width=576 height=324>

## How it works
#### Deep learning in general
“Deep learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Similarly to how we learn from experience, the deep learning algorithm would perform a task repeatedly, each time tweaking it a little to improve the outcome. We refer to ‘deep learning’ because the neural networks have various (deep) layers that enable learning.” – Bernard Marr

#### My implementation
I trained my artificial neural network on the MNIST digits database made up of 7000 28x28 images. The neural network can classify the digits (0-9) into 10 categories - one for each digit. To prove that the algorithm works with almost any 28x28 image of a handwritten digit, I added the ability to create one's own handwritten digit and check if the algorithm "reads" the digit. 

#### Built with
- Python 3.6.8
- Keras 2.1.6
- OpenCV 4.0.0
- MNIST digits database

## How to install
- Install Python 3.6 and pip (as of August 2019, Keras is not compatible with Python 3.7+)
- Install libraries via pip
```
pip install numpy
pip install --upgrade tensorflow
pip install opencv
```
- Download and extract the files from GitHub
- Run 'digit_reader.bat'!
