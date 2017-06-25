# AutoDetect
I created an iOS app with Xcode using a image classification model with CoreML. My goal for this project was to create an real world application that makes it easy to find out the make and model of a car. For building my model in Keras, I used transfer learning for which I took the pre-trained InceptionV3 model and extended it with my own neural network. My neural network was trained on 170k car images from about 70k cars. With 468 classes/car models I managed to get an accuracy of 90%/80% on the training/validation set.  
The mobile app was created in iOS 11 beta 2 with Xcode using Apple's recently released Machine Learning library CoreML to convert the Keras model and implement it in the app.

![AD](app.png)
