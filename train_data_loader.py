from preprocessing_data import text_filtering, hist_gauss, adaptive_filtering

import numpy as np
import cv2
import os



def train_data_loader(data_dir, labels, img_resize):
    
    """
    
        INPUT:
        data_dir - path to raw data
        labels - labels of groups
        img_resize - final size image to training
                
        OUTPUT:
        preprocessed_data - preprocessed data after text filtering, histogram equalization, gaussian blur and adaptive filtering transformation
        y_data - labels of groups
        
    """    
        
    path_collector = []
    x_data = []
    y_data = []
    data = []
    
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    
            path_collector.append(str(os.path.join(path, img)))

            resized_arr = cv2.resize(img_arr, (img_resize, img_resize)) 
            
            text_filtered = text_filtering(resized_arr)
            
            data.append([text_filtered, class_num])
                
    np_data = np.array(data, dtype=object)
    
    for feature, label in data:
        x_data.append(feature)
        y_data.append(label)
    
    hist_data = hist_gauss(x_data)
    
    preprocessed_data = adaptive_filtering(hist_data, mutiplicator=0.9)
    
    return preprocessed_data, y_data