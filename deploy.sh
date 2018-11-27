#!/bin/bash
folder='videos_files/'
echo 'listing files in folder'    
ls $folder | while read x; do
    echo $x
    if [[ $x == *"C4"* ]]; then
        echo "Running for C4 config"
        echo $folder$x
        python3 yolo_video.py --input $folder$x --json_config_path C4_config.json --frame_ratio 0.1  --visual_display 0; 
    fi
    if [[ $x == *"C3"* ]]; then
        echo "Running for C3 config"
        echo $folder$x
        python3 yolo_video.py --input $folder$x --json_config_path C3_config.json --frame_ratio 0.1  --visual_display 0; 
    fi
done