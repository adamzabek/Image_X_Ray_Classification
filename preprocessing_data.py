import numpy as np
import cv2



def text_filtering(resized_data):
    
    """
    
        INPUT:
        resized_data - Image array uploaded and resized from source path
                
        OUTPUT:
        text_filtered - Image array after removing "text"
        
    """    
    
    original = resized_data.copy()
    blank = np.zeros(resized_data.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(resized_data, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Merge text into a single contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter using contour area and aspect ratio    
    for c in cnts:

        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w / float(h)
        if (ar > 1.4 and ar < 4) or ar < .85 and area > 10 and area < 500:
    
            # Find rotated bounding box
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(resized_data,[box],0,(36,255,12),2)
            cv2.drawContours(blank,[box],0,(255,255,255),-1)

        # Bitwise operations to isolate text
        extract = cv2.bitwise_and(thresh, blank)
        extract = cv2.bitwise_and(original, original, mask=extract) 
        gray2 = cv2.cvtColor(extract, cv2.COLOR_BGR2GRAY)
        blur2 = cv2.GaussianBlur(gray2, (5,5), 0)
        thresh2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY)[1]
        text_filtered = cv2.inpaint(original, thresh2, 7, cv2.INPAINT_TELEA)
        text_filtered = cv2.cvtColor(text_filtered, cv2.COLOR_BGR2GRAY)    

    return text_filtered



def hist_gauss(data):
    
    """
    
        INPUT:
        data - Image matrix (list or numpy array)
                    
        OUTPUT:
        processed_data - Image matrix after histogram equalization and gaussian blue transformation
        
    """
    
    processed_data_hist = np.copy(data)
    
    processed_data_temp = []
    
    """
    
        Histogram equilization transformation
   
    """
    
    for count, value in enumerate(processed_data_hist):
        processed_img = cv2.equalizeHist(processed_data_hist[count], (0,255))
        processed_data_temp.append(processed_img)

    processed_data_hist = []
    processed_data_hist = processed_data_temp
    
    """
    
        Gaussian blur transformation
   
    """
    
    processed_data_gaus = np.copy(processed_data_hist)    

    processed_data_temp = []
    
    for count, value in enumerate(processed_data_gaus):
        processed_img = cv2.GaussianBlur(processed_data_gaus[count], (3,3), cv2.BORDER_DEFAULT)
        processed_data_temp.append(processed_img)

    processed_data_gaus = []
    processed_data_gaus = processed_data_temp
    
    processed_data = []
    processed_data = processed_data_gaus
    
    return processed_data



def adaptive_filtering(img, mutiplicator):
    
    """ 
        
        INPUT:
        img - Image matrix (list or numpy array)
        mutiplicator - mian numerical value for imager artifacts removal
            ~ Suggested mutiplicator = 0.9
                
        OUTPUT:
        img_filtered - Image matrix after adaptive filtering transofmration
        
    """
    
    img_composition = []
    img_filtered = []
    
    if isinstance(img, list):
        numer_iteration = len(img)  
    
    if isinstance(img, np.ndarray):
        numer_iteration = img.shape
        numer_iteration = numer_iteration[0]
    
    if not mutiplicator:
        mutiplicator = 0.9
    
    for num in range(numer_iteration):
        temp_img=[]
        temp_img = np.copy(img[num])
        min_value = []
        max_value = []
        adpt_filtr_img = []
        num_size = temp_img.shape
        min_val = np.amin(temp_img)
        max_val = np.amax(temp_img)
        threshold = []
        threshold = min_val+(mutiplicator*(max_val - min_val))
        temp_img_wo = np.copy(temp_img)
        temp_img[temp_img<threshold] = 0
        dst = cv2.inpaint(temp_img_wo,temp_img,3,cv2.INPAINT_NS)
        img_filtered.append(dst)
    
    return np.array(img_filtered)