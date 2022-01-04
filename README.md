# Content
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Conclusion and Recommendations](#Conclusion-and-Recommendations)
- [Dataset](#Dataset)
- [Resources](#Resources)


# Problem Statement
ASL (American Sign Language) is a visual language that is expressed through different hand motions and positions to help those who are deaf or hard of hearing communicate. Many people who can hear are also learning this language. The goal of this project is to use machine learning and object detection to identify the different hand positions that correspond to the ASL alphabet. The model will then be deployed in an app to be used in image detection and real time detection.


# Executive Summary
To answer my problem statement, I first has to find a dataset of ASL letters images. After finding a [dataset](#Dataset) that I could use, it was time to refine one of the base Yolov5 models on the custom dataset. 

Initially this was attempted by using the [Google Colab Notebook](#Resources) that RoboFlow created. After several failed attempts, I chanced strategies to train the model on my local machine using my GPU with Yolov5's `train.py` file. This drastically decreased training time. Then I ran Yolov5's `eval.py` script to evaluate the performance of the model. The evaluation indicted that the model had a `mAP_0.5` score of about `93%` indicating the model performed really well. Finally I saved my trained model to a `.pt` file so that it could be used with other files and the `Streamlit` app.

Finally I created a Streamlit app to deploy the model. The Streamlit app was designed to detect in an uploaded image and to detect in real time with a webcam.


# Conclusion and Recommendations
Although my model had a really good `mAP_0.5` score, it had a lot of trouble detecting and predicting ASL letters live. This is due to the dataset that the model used to train itself. If we look more closely at some of the images, they consist of mostly close ups of the hand in a position for a ASL letter. Contrast that with what frames the camera is feeding into the model. The frames consist of people, faces, objects in the background and more. Basically the images being feed into the model for prediction don't match the image style that were used to train the model.

Going forward, I would want to actually create two different models. The first would be an object detection model that would detect hands in an image. Not hands in a particular position but any hands in an image. Also I would make sure I had way more images (closer to 10,000) and also make sure there are a bunch of things in the image other than hands. This way the model would be able to function better in a real world scenario where someone is using my model in front of their webcam.

Next I would want to take any image that has a hand detected in it and crop it so that the new image only consists of the hand. Then I would want to pass that through a CNN model; Pytorch has a really good one called `resNet`. Then the second model would classify why ASL letter the hand is making.

I would also want to looking into augmenting the images I have to increase my dataset. Augmenting the images means making random copies of random images in the dataset and then applying an augmentation like blur to the copy. This would increase the dataset for training but also teach the model how to handle images that aren't perfect.

That being said, my current model well especially during Image Detection when the image somewhat is similar to the training images that were used (i.e. close up on hands with not much in the background). Having a `mAP_0.5` score of about `93%` is extremely good for a model. I used the model in my `Streamlit` app, and was confident in it's image detection when uploading a single image. Like I said before, real time detection still needs more work so that it can work smoothly.


# Dataset
#### American Sign Language Letters Dataset (from RoboFlow)
The dataset can be found [here](https://public.roboflow.com/object-detection/american-sign-language-letters).

The dataset contains 1728 images of size 416 pixels square. By default the augmentations applied randomly were:

- Flip (Horizontal)
- Crop (0% min zoom - 20% max zoom)
- Rotation (between -5° to +5°)
- Shear (±5° Horizontal - ±5° Vertical)
- Greyscale (Applied to 10% of images)
- Brightness (Between -25% and +25%)
- Blur (Up to 1.25px)

For each image, 3 copies were made and the augmentations named were applied at random to the copies.


# Resources


#### Modeling and Training
- [Yolov5 Repo](https://github.com/ultralytics/yolov5)
- [Training Yolov5 Model with RoboFlow](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
- [RoboFlow's Google Colab Notebook from Training](https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ)
- [RoboFlow's Blog Tutorial on Training a Model](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
- [RoboFlow's Video Tutorial on Training a Model](https://youtu.be/MdF6x6ZmLAY)

#### Deploying and Using Model
- [YOLO, Pytorch and Python Video Tutorial on Deploying a Model](https://youtu.be/tFNJGim3FXw)
- [PyTorch Tutorial - Saving and Loading Models](https://youtu.be/9L9jEOwRrCg)
- [Deep Learning With PyTorch - Full Course](https://youtu.be/c36lUUr864M)

#### Streamlit App
- [Accessing WebCam with OpenCV in Streamlit](https://youtu.be/tkFsqTjoaVM)