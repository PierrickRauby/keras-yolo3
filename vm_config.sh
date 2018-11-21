#!/bin/bash                                                                     
#script to configure the vm on glcoud
# to answer prompt automatically use  the following command line to start the script (not tested)
#      $> yes Y |./your_script
echo 'This code has never been tested'
echo 'installing git'
sudo apt-get install git
echo 'installing pip3'
sudo apt-get install python3-pip
echo 'install libsm6 libxext6 libxrender-dev'
sudo apt-get install -y libsm6 libxext6 libxrender-dev
echo 'installing numpy tensorflow and keras'
sudo pip3 install numpy keras tensorflow pillow matplotlib opencv-python
echo 'cloning https://github.com/PierrickRauby/keras-yolo3.git'
sudo git clone https://github.com/PierrickRauby/keras-yolo3.git
echo '################################################### entering keras-yolo3/ ###################################################'
alias proj="cd keras-yolo3/"
proj
echo 'loading weight'
wget https://pjreddie.com/media/files/yolov3.weights
echo 'converting weight, destination model_data/yolo.h5'
python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
echo 'copying video'
gsutil cp gs://cse-6242-project/2018_10_17_0900_2130_C3.mp4 .
echo '-----'
echo 'starting analysis'
python3 yolo_video.py --input 2018_10_17_0900_2130_C3.mp4 --frame_ratio 0.00006 --json_config_path C3_config.json --visual_display 0
echo 'done'