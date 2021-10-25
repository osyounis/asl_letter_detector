"""
This file contains the code used to create a streamlit app. The streamlit app
is an ASL (American Sign Language) letter converter.

The model that is used to make prediction on which letter is shown in the image, 
is a yolov5s model using custom weights. This model is implemented using PyTorch.

OpenCV is also used in  the app to access a webcamera. This is for real time 
detection purposes.
===============================================================================

License: MIT License

Author: Omar Younis
"""


########################################
#              Imports                 #
########################################

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image



########################################
#              Settings                #
########################################

# Loading model from file
filepath = 'C:/Users/Omar/Documents/GitHub/asl_letter_detector/code/'
model_file = 'asl_detector_model.pt'
model = torch.load(filepath + model_file)

# List of pages in the app
pages = ['About', 'Image Prediction', 'Live Prediction']



########################################
#              GUI Code                #
########################################

# Setting up the Streamlit App main settings.
st.set_page_config()
st.title('ASL Letter Translator')
page = st.sidebar.selectbox('Page', pages)


# About Page of the App
if page == 'About':
	st.subheader('About This App!')
	st.write("""
		This app is a tool which can be used to detect and translate ASL letters.
		It can be done in real time or by uploading an image.
		\nThe app uses a `Yolov5s` model with custom weights applied.
		\nTo use this app, navigate to either the `Image Prediction` or 
		`Live Prediction` section of the app using the sidebar on the left.
		\n• Use the `Image Prediction` page if you would like to upload an image
		of an ASL letter to translate.
		\n• Use the `Live Prediction` page to access your camera and translate 
		ASL letter in real time.
		""")



# Image Prediction Page of the App
elif page == 'Image Prediction':
	st.subheader('Upload Image')
	st.write("""
		To upload an image, click on the `Browse File` button or drag and drop 
		the file in the box below.
		\nImages should be `416px X 416px` in size and mush be in a `.jpg` format.
		\nOnce the image has been uploaded, the prediction will happen automatically.
		""")
	image_file_buffer = st.file_uploader("Upload Image", type=['.jpg'])
	if image_file_buffer is not None:
		uploaded_image = Image.open(image_file_buffer)


		# Original Uploaded Image
		st.subheader("Original Upload")
		st.image(uploaded_image)

		# Image Prediction
		image_pred = model(uploaded_image)
		num_pred = int(image_pred.pred[0][0][5])
		st.subheader("Image Prediction")
		st.image(np.squeeze(image_pred.render()))


# Live Prediction Page of the App
elif page == 'Live Prediction':
	st.subheader('ASL Letter Check')
	st.write("""
		Press the `Run` checkbox to start your camera. Once the camera starts 
		you can start signing different ASL letters. As you sign, the prediction
		will appear in a box around your signing hand.
		""")
	run_camera = st.checkbox('Run')
	FRAME_WINDOW = st.image([])
	camera = cv2.VideoCapture(0)

	while run_camera:
		ret, frame = camera.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		mod_frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)
		results = model(mod_frame)

		# Make detections on the current frame
		FRAME_WINDOW.image(np.squeeze(results.render())) 
	
	else:
		st.write('Stopped')