# Emergency-Vehicle-Detection-using-Image-Processing
## Implementing Convolution Neural Networks to detect emergency vehicles on Indian roads

Hack Vista Hackathon
Team members- Kaushik S and Abhishek Raman

Problem Statement and solution- We propose a model that employs real time image processing for detection of emergency vehicles using a convolutional neural network (CNN) architecture. The signal control unit can be programmed to terminate the round robin sequence preferentially upon detection of an emergency vehicle.
The described problem is addressed by training a CNN on a dataset  of images of ambulances in the Indian context. The deep learning platform used for training was TensorFlow, which is offered as a library in Python. 

Dependencies- Keras, Tensorflow, Utils, Numpy and OpenCV

The CNN is trained using Tensorflow on a dataset of  images of ambulances and the parameters of the resultant trained model is enclosed in the frozen_inference_graph.pb file.

The objectdetection1.py extracts the model from the frozen_inference_graph.pb file to generate a bounding box around the ambulance in amb1.jpg. The image with the bounding box is stored as objectdetection1.jpg

Similarly, objectdetection2.jpg corresponds to the result obtained for amb2.jpg. 

The trained model can be validated on other images of ambulances by simply altering the path in Line 24 of objectdetection1.py.

*Please download the frozen_inference_graph.pb (~181 MB) seperately and add it to the project directory. Please do not execute the program immediately after downloading it along with the .zip file.

Youtube video link: https://youtu.be/9-y6Ta1Rw3I
