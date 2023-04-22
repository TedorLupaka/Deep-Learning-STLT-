import torch
import numpy as np
import cv2
from time import time
import controller as cnt


class CarDetection:
    
    """
    Class implements Yolo5 model to make inferences using Opencv2
    
    """


    def __init__(self, capture_index, capture_index1, model_name, model_name1):
        
        """
        Initializes the class with video feeds and DL model.
        :param capture_index: Index of the video feed to be used for inference.
        :param capture_index1: Index of the video feed to be used for inference.
        :param model_name: Name of the model to be used for inference.
        :param model_name1: Name of the model to be used for inference.
        
        Note: Two cameras where used for this project.
        
        """
        self.capture_index = capture_index
        self.capture_index1 = capture_index1
        self.model = self.load_model(model_name)
        self.model = self.load_model(model_name1)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object.
        """
      
        return cv2.VideoCapture(self.capture_index)
    
    def get_video_capture1(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object.
        """
      
        return cv2.VideoCapture(self.capture_index1)

    def load_model(self, model_name):
        """
        Loads our trained Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def load_model1(self, model_name1):
        """
        Loads our trained Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name1:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name1, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    
    def score_frame1(self, frame1):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame1 = [frame1]
        results1 = self.model(frame1)
        labels1, cord1 = results1.xyxyn[0][:, -1], results1.xyxyn[0][:, :-1]
        return labels1, cord1

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.0001:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                cv2.putText(frame, f'Total cars in lane: {n}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cnt.cars(n)
        return frame
        
    def plot_boxes1(self, results1, frame1):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels1, cord1 = results1 
        n1 = len(labels1)
        x_shape, y_shape = frame1.shape[1], frame1.shape[0]
        for i in range(n1):
            row = cord1[i]
            if row[4] >= 0.0001:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame1, (x1, y1), (x2, y2), bgr, 2)
                
                cv2.putText(frame1, self.class_to_label(labels1[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
                cv2.putText(frame1, f'Total cars in lane: {n1}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cnt.cars1(n1)
        return frame1    
     
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap1 = self.get_video_capture1()
        cap = self.get_video_capture()
        assert cap1.isOpened()  
        assert cap.isOpened()
          
      
        while True:
          
            ret1, frame1 = cap1.read()
            ret, frame = cap.read()
            
            assert ret
            assert ret1
            
            frame = cv2.resize(frame, (416,416))
            frame1 = cv2.resize(frame1, (416,416))
            
            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            
            start_time1 = time()
            results1 = self.score_frame1(frame1)
            frame1 = self.plot_boxes1(results1, frame1)
            
            end_time1 = time()
            fps = 1/np.round(end_time - start_time, 2)
            fps1 = 1/np.round(end_time1 - start_time1, 2)
            #print(f"Frames Per Second : {fps}")
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(frame1, f'FPS: {int(fps1)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)
            cv2.imshow('YOLOv5 Detection1', frame1)
            
            
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
      
        cap.release()
        cap1.release() 
        
        
# Create a new object and execute.
detector = CarDetection(capture_index=0, capture_index1=2, model_name='model.pt', model_name1='model.pt')
detector()