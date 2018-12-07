#How to use the code for the project 

- Once the folder is opened you can download the weight from, run this : 
`
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
`

- In order to run the yolo code for only one video : 
`
python3 yolo_video.py --input <video.mp4> --json_config_path <Config file for C3 or C4> --frame_ratio <as a float e.g. 0.1> --visual_display <0 for no diplay, 1 for display>
`
__Note__: the video should be named as `yyyy_mm_dd_hhmmStart_hhmmEnd_CameraNumber.mp4` e.g. `2018_10_04_0800_2230_C4.mp4` 

- To process a folder containing video :
  Put videos under 'video_files' and run
  `chmod +x deploy.sh
  ./deploy.sh`
