#!/usr/bin/python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# box_position.py
#
# read the machine position from config file and displays in the video
#
# Created: November 1, 2018 - P Rauby -- pierrick.rauby@gmail.com
#
# Modified:
#   * November 14, 2018 - PR
#           - add option to hide/display video output
#----------------------------------------------------------------------
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer
import numpy as np
import csv,os,math
import json
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


#### MACHINE DEFINITION PART #####
class machine(object):
    def __init__(self, name="",x_machine=0,y_machine=0,distance_min=0,occupation_count=0,first_timeslot_index=0,current_timeslot_index=0,status=0):
        self.name=name
        self.x_machine=x_machine
        self.y_machine=y_machine
        self.distance_min=distance_min
        self.occupation_count=occupation_count #number of occupied frames
        self.first_timeslot_index=first_timeslot_index
        self.occupation_timeslot={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,
                    15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:0,
                    28:0,29:0,30:0,31:0,32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:0,40:0,
                    41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0} #dictionnary of timeslot for a day
        self.current_timeslot_index=first_timeslot_index
        self.status=status#/!\ if status=0 the machine is free,
                          # if status =1 the machine is occupied
    def check_availability_machine(self,classified,image): #classified as the list of all object detected
        self.status=0 #free the machine, it is going to be checked after 
        for object_classified in classified: #for all object found in the image
            if object_classified["label"]=="person": #if its a person I check the distance to the machine 
                x_person=object_classified["x_box"]
                y_person=object_classified["y_box"]
                distance_to_machine=math.sqrt((x_person-image.size[0]*self.x_machine)*(x_person-image.size[0]*self.x_machine)+(y_person-image.size[1]*self.y_machine)*(y_person-image.size[1]*self.y_machine))
                if (distance_to_machine<(self.distance_min*math.sqrt(image.size[0]*image.size[0]+image.size[1]*image.size[1])/math.sqrt(2))): #the person is too close to the machine => occupied
                    self.status=1 # => machine occupied

# Creating the machines
def create_machine_from_file(machine_file_path,video_metadata):
    # from machine.json parsing and creating the machines 
    config_file=open(machine_file_path,'r') #open the config file for reading 
    content=config_file.read() #get the content of the above openned config file 
    parsed_content=json.loads(content)
    machine_list=[]
    for _machine_ in parsed_content:
        machine_now=machine(name=parsed_content[_machine_]['machine_name'],
                            x_machine=float(parsed_content[_machine_]['x_machine']),
                            y_machine=float(parsed_content[_machine_]['y_machine']),
                            distance_min=float(parsed_content[_machine_]['distance_min']),
                            first_timeslot_index=int(video_metadata['video_start']/30)+1, #first_timeslot index
                            current_timeslot_index=int(video_metadata['video_start']/30)+1, #current_timeslot_index
                            )
        #getting the index of the first time slot of the video
        print(machine_now.name)
        print("        x is: "+str(machine_now.x_machine))
        print("        y is: "+str(machine_now.y_machine))
        print("        distance min is: "+str(machine_now.x_machine))
        print("        occupation count  is: "+str(machine_now.occupation_count))
        print("        first timeslot index is: "+ str(machine_now.first_timeslot_index))
        print("        occupation timeslot is: "+str(machine_now.occupation_timeslot))
        print('        current_timeslot_index is: '+ str(machine_now.current_timeslot_index))
        print('        machine status is: '+ str(machine_now.status))
        machine_list.append(machine_now)
    return machine_list

#### AUXILIARY FUNCTIONS 
#convert hours and minutes 
def convert_hour_to_minute(hour_minute):
    hours=int(hour_minute[0:2])
    minutes=int(hour_minute[2:4])
    return 60*hours+minutes

#title parsing function 
def parse_video_path(video_path):
    video_name_format=video_path.split("/")[-1] #get the last element of the path as the name
    if len(video_name_format.split("."))==2:
        video_name, video_format=video_name_format.split(".",1) #get the format and the name splitting on the point
        if len(video_name.split("_"))==6:
            video_year,video_month,video_day,video_start,video_end,video_camera=video_name.split("_")
            return {'name':video_name,'format':video_format,'video_year':video_year,'video_month':video_month,'video_day':video_day,
            'video_start':convert_hour_to_minute(video_start),'video_end':convert_hour_to_minute(video_end),'video_camera':video_camera}
        else:
            print("ERROR: file name illegal for timestamp and camera")
    else: 
        print("ERROR: file name illegal for video format")
        return 0 


#### YOLO CLASS ##### (contains the image inferance function )
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,machine_list,visual_display):
        start = timer()
        classified=[]
        if self.model_image_size != (None, None):
            # print('model image size: '+str(self.model_image_size))
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))

        else: #in our case the condition in the above if statement is true (the following is ignored)
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print("format: "+str(image_data.shape))

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print( 'Found {} boxes for {}'.format(len(out_boxes), 'img')) 
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        # a partir de la j'ai l'image et je peux dessiner les cerles pour mes machines (fin je pense)
        # dessin des machines sur l'image
        if (visual_display):
            for machine_to_draw in machine_list:
                draw_machine = ImageDraw.Draw(image) #draw object for the selected machine
                offset = int(machine_to_draw.distance_min*math.sqrt(image.size[0]*image.size[0]+image.size[1]*image.size[1])/math.sqrt(2))
                for i in range(thickness):
                    draw_machine.ellipse(
                        [(np.floor(0.5+image.size[0]*machine_to_draw.x_machine-offset).astype('int32')) + i,
                        (np.floor(0.5+image.size[1]*machine_to_draw.y_machine-offset).astype('int32')) + i,
                        (np.floor(0.5+image.size[0]*machine_to_draw.x_machine+offset).astype('int32')) - i, 
                        (np.floor(0.5+image.size[1]*machine_to_draw.y_machine+offset).astype('int32')) - i],
                        outline='gold')
            del draw_machine
        #itterate over all the classes found 
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            # Here is where the classes output are used. 
            top, left, bottom, right = box #recupere les valeurs du format de la box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            classified.append({"label":predicted_class,"score":score,
                                "x_box":left+int((right-left)/2),"y_box":top+int((bottom-top)/2)}) #the list of all object of the image

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]]) #prepare le label sur le mettre sur l'image
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            if (visual_display):
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle( #dessine l'image 
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                # for _machine_ in machine_list : #
            del draw

        end = timer()
        print("\tComputation time: " + str(end - start))
        return image,classified #returns the image and the list of object 

    def close_session(self):
        self.sess.close()


##### VIDEO PROCESSING FUNCTION #####
def detect_video(yolo, video_path, frame_ratio,json_path,visual_display,output_path=""): #output path will be used for the csv, default the pwd
    import cv2
    video_metadata=parse_video_path(video_path)
    #prepare the time slot list as a list of frame separators:
    timeslot=[0,1800,3600,5400,7200,9000,10800,12600,14400,16200,18000,19800,21600,
            23400,25200,27000,28800,30600,32400,34200,36000,37800,39600,41400,
            43200,45000,46800,48600,50400,52200,54000,55800,57600,59400,61200,
            63000,64800,66600,68400,70200,72000,73800,75600,77400,79200,81000,
            82800,84600]
    #removing timeslots that are not on the video from the timeslot list:
    timeslot = [time-60*video_metadata['video_start'] for time in timeslot if ((time-60*video_metadata['video_start'])>0 and (time-60*video_metadata['video_start']) <= 60*(video_metadata['video_end']-video_metadata['video_start'])) ]
    print('timeslot: '+ str(timeslot))

    if(json_path==''):
        print('Warning ! No Machine configuration provided')
        machine_list=[]
    else:
        machine_list=create_machine_from_file(json_path,video_metadata)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    ### Variable definition for the analysis ###
    video_length=60*(video_metadata['video_end']-video_metadata['video_start']) #the length of the video in second
    number_frame= int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #not sure if this  value is meaningful
    fps=number_frame/video_length #number of frame per second on this video
    dont_skip_frame=0 #count of the number of frame skiped, if egal to frame_ratio_inverted then analyze the frame else skip
    frame_ratio_inverted=int(1/float(frame_ratio)-1) #to know the number of frame to skip based on the command line ration intup
    frame_counter=0 #number of frame opened since the program start
    frame_analyzed_in_timeslot=0 #counter of the number of fram
    csv_file=os.getcwd()+'/'+video_metadata['name']+'_'+'frame'+".csv" #file to write the result of the analysis
    csv_file_timeslot=os.getcwd()+'/'+video_metadata['name']+'_'+"timeslot.csv" #file to write the timeslot results
    log_file=os.getcwd()+'/'+video_metadata['name']+'_'+"analysis_log.txt" #log file for the metadata
    ### End of variable definition ###
    #displaying metadata
    print('\n------')
    print('Video metadata:')
    print("\tstart: "+ str(60*video_metadata['video_start'])+'s')
    print("\tstart: "+ str(60*video_metadata['video_end'])+'s')
    print("\tduration: "+ str(video_length)+'s')
    print("\tduration: "+ str(video_length)+'s')
    print("\tnumber_frame:" + str(number_frame))
    print("\tfps:" + str(fps))    
    print('Considering 1 frame every ' + str(frame_ratio_inverted+1)+" frames")
    print('metadata infor stored in: "'+log_file+'"')
    print('analysis result stored in: "'+csv_file+'"')
    #writting the metadata into a log file 
    log_file_fd=open(log_file,'w')
    log_file_fd.write('Video metadata:\n')
    log_file_fd.write("\tduration: "+ str(video_length)+'s\n')
    log_file_fd.write("\tnumber_frame:" + str(number_frame)+'\n')
    log_file_fd.write("\tfps:" + str(fps)+'\n')
    log_file_fd.write('Considering 1 frame every ' + str(frame_ratio_inverted+1)+" frames\n")
    log_file_fd.write('metadata infor stored in: "'+log_file+'"\n')
    log_file_fd.write('analysis result stored in: "'+csv_file+'"')
    log_file_fd.close()

    #write the analysis results in csv in present working directory
    with open(csv_file, 'w', newline='') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['frame_number','time','machine_name','status'])
        #loop over all frames of the video 
        with open(csv_file_timeslot, 'a', newline='') as csvfile:
            file_writer_timeslot = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer_timeslot.writerow(['timeslot','machine_name','machine_status'])
        while True:
            frame_counter+=1
            return_value, frame = vid.read() #no test on return_value, maybe a way tso detect the end of the video
            if(frame_counter/fps>timeslot[0]): #si le temps ecoulé de l'analyse est superieur a mon time slot alors je reinitialise le timeslot 
                print('NEXT TIME SLOT')
                for machine in machine_list: #update the availability of all machines
                    machine.occupation_timeslot[machine.current_timeslot_index]=100*machine.occupation_count/frame_analyzed_in_timeslot # Calculation of the occupation ratio as the mean over the timeslot
                    with open(csv_file_timeslot, 'a', newline='') as csvfile:
                        file_writer_timeslot = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file_writer_timeslot.writerow([machine.current_timeslot_index,machine.name,machine.occupation_timeslot[machine.current_timeslot_index]])
                    machine.current_timeslot_index+=1 #working the next timeslot
                    machine.occupation_count=0
                timeslot.pop(0) #deleting the first element of the timeslot list. 
                frame_analyzed_in_timeslot=0
            
            if return_value:
                if (frame_ratio_inverted==dont_skip_frame): #I analyze the frame
                    frame_analyzed_in_timeslot+=1
                    print("number of frame analyzed in timeslot "+str(frame_analyzed_in_timeslot))
                    #printint details  TEMPORARY
                    print(timeslot[0])
                    print(frame_counter/fps)
                    for machine in machine_list:    
                        print(machine.name)
                        print("        x is: "+str(machine.x_machine))
                        print("        y is: "+str(machine.y_machine))
                        print("        distance min is: "+str(machine.x_machine))
                        print("        occupation count  is: "+str(machine.occupation_count))
                        print("        first timeslot index is: "+ str(machine.first_timeslot_index))
                        print("        occupation timeslot is: "+str(machine.occupation_timeslot))
                        print('        current_timeslot_index is: '+ str(machine.current_timeslot_index))
                        print('        machine status is: '+ str(machine.status))
                    #end printing details

                    print('\n------')
                    print('Frame number ' + str(frame_counter)+' at time ' + str(frame_counter/fps))
                    image = Image.fromarray(frame)
                    image,classified = yolo.detect_image(image,machine_list,visual_display) #classified is the list of object detected in the image
                    result = np.asarray(image)
                    for machine in machine_list: #update the availability of all machines
                        machine.check_availability_machine(classified,image) 
                        file_writer.writerow([frame_counter,frame_counter/fps,machine.name,machine.status])
                        status_as_string='available' if (machine.status==0) else 'occupied'
                        print('\t'+machine.name+" is "+ status_as_string)
                        machine.occupation_count+=machine.status
                    if (visual_display): #display the video if asked 
                        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                        cv2.imshow("result", result)
                        if isOutput:
                            out.write(result)
                    if cv2.waitKey(1) & 0xFF == ord('q'): #stop condition, have to check if working 
                        break
                    dont_skip_frame=0
                    print('------')
                else: # I just skip the frame
                    dont_skip_frame+=1
                    if cv2.waitKey(1) & 0xFF == ord('q'): #stop condition, have to check if working 
                        break  
            else:
                file_writer.writerow([frame_counter,frame_counter/fps,'ERROR READING FRAME','ERROR READING FRAME'])
            if (frame_counter==number_frame):
                for machine in machine_list: #update the availability of all machines
                    machine.occupation_timeslot[machine.current_timeslot_index]=100*machine.occupation_count/frame_analyzed_in_timeslot # Calculation of the occupation ratio as the mean over the timeslot
                    with open(csv_file_timeslot, 'a', newline='') as csvfile:
                        file_writer_timeslot = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        file_writer_timeslot.writerow([machine.current_timeslot_index,machine.name,machine.occupation_timeslot[machine.current_timeslot_index]])
                    machine.current_timeslot_index+=1 #working the next timeslot
                    machine.occupation_count=0
                break
    yolo.close_session()
