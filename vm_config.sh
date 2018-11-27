#!/bin/bash                                                                     
#script to configure the vm on glcoud
# to answer prompt automatically use  the following command line to start the script
#       yes Y | sh config.sh

#git already installed on this VM 

# echo 'updating apt-get'
# sudo apt-get update
# echo 'installing git'
# sudo apt-get install git
# The following line are not used in  the case of a precompiled image (from google)
# echo 'installing pip3'
# sudo apt-get install python3-pip
# echo 'install libsm6 libxext6 libxrender-dev'
# sudo apt-get install -y libsm6 libxext6 libxrender-dev
# echo 'installing numpy tensorflow and keras'
# sudo pip3 install numpy keras tensorflow pillow matplotlib 
# sudo pip3 install opencv-python
echo 'cloning https://github.com/PierrickRauby/keras-yolo3.git'
sudo git clone https://github.com/PierrickRauby/keras-yolo3.git
echo '################################################### entering keras-yolo3/ ###################################################'
alias cd_folder='cd keras-yolo3'
cd_folder
echo 'loading weight'
sudo wget https://pjreddie.com/media/files/yolov3.weights
echo 'converting weight, destination model_data/yolo.h5'
# alias command_weight='python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5'
sudo python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
echo 'copying video'
sudo gsutil cp gs://cse-6242-project/2018_10_14_0900_2200_C4.mp4 .
echo '-----'
echo 'starting analysis'
sudo python3 yolo_video.py --input 2018_10_14_0900_2200_C4.mp4 --frame_ratio 0.1 --json_config_path C4_config.json --visual_display 0
echo 'done'