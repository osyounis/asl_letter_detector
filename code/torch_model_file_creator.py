"""
This file contains the code used to create the final model used in predicting
ASL letters in images and video. The file is saved as a .pt (PyTorch) file so 
the Streamlit app can use it to make predictions.
===============================================================================

License: MIT License

Author: Omar Younis
"""


########################################
#              Imports                 #
########################################

import torch


########################################
#              Settings                #
########################################

weights_file = "../model_weights/asl_yolov5_model_v2.pt"
filename = "asl_detector_model.pt"


########################################
#              Main Code               #
########################################

model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, force_reload=True)
torch.save(model, filename)