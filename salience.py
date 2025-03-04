import fitz
import os
import cv2
import matplotlib.pyplot as plt
from DeepRare_2019_lib import DeepRare2019
from keras.applications.vgg16 import VGG16
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import pandas as pd




#pdf to image
files = os.listdir("pdf")

for file in files:
    fname = file[0:-4]
    print(file)
    dpi = 300  # choose desired dpi here
    zoom = dpi / 72  # zoom factor, standard: 72 dpi
    magnify = fitz.Matrix(zoom, zoom)  # magnifies in x, resp. y direction
    doc = fitz.open("pdf\%s"%file)  # open document
    if not os.path.exists("input\%s"%fname):
        os.makedirs("input\%s"%fname)
    for page in doc:
        pix = page.get_pixmap(matrix=magnify)  # render page to an image
        pix.save(f"input\%s\%s_{page.number}.png"%(fname,fname))   

#saliency map calculation
sal = DeepRare2019() # instantiate class
sal.model = VGG16() # call VGG16 and send it to the class
sal.filt = 1 # make filtering
sal.face = 1 # use face (to be used only with VGG16)

items = os.listdir('input')

for item in items:

    directory = r'input\%s'%item
    #plt.figure(1)

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            go_path = os.path.join(directory, filename)

            img = cv2.imread(go_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            sal.img = img

            saliency_map, saliency_details, face = sal.compute() # here no visualization, only final sal maps but if you use the line above and uncomment the lines in the lib you can get the groups

        
            if not os.path.exists('output_raw\%s'%item):
                os.makedirs('output_raw\%s'%item)
            file_no_extension = os.path.splitext(filename)[0]
            file_out = file_no_extension + '.jpg'
            go_path_raw = os.path.join('output_raw\%s'%item, file_out)
            cv2.imwrite(go_path_raw, 255*saliency_map)

        else:
            continue

# Salient regions detection
folders = os.listdir('output_raw')

sal_area = []
sal_cnt = []
ticker = []
year = []

for folder in folders:
    
    tic = folder.split("_")[1]
    yr = folder.split("_")[2]
    print(tic)
    
    files = os.listdir('output_raw\%s'%folder)
    for file in files:
    

        image = cv2.imread('output_raw\%s\%s'%(folder,file))
        image = cv2.resize(image,(0, 0),fx=0.18, fy=0.18, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        #cv2.imshow("a1",blurred)
        #cv2.waitKey(0)

        thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 1500:
                mask = cv2.add(mask, labelMask)
                
                
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        count = -1 
        try:
            cnts = contours.sort_contours(cnts)[0]
        except ValueError:
            count = 0

        if count==0:
            sal_area.append(0)
            sal_cnt.append(0)
            ticker.append(tic)
            year.append(yr)
        else:
        # loop over the contours
            for (i, c) in enumerate(cnts):
                print(i)
                (x, y, w, h) = cv2.boundingRect(c)
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(cX), int(cY)), int(radius),
                    (0, 0, 255), 3)
                cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # show the output image
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

            sal_area.append(cv2.countNonZero(thresh)/thresh.shape[0]/thresh.shape[1]*100)
            sal_cnt.append(i+1)
            ticker.append(tic)
            year.append(yr)

contrast = np.array(sal_area).mean()/100
alpha = np.array(sal_cnt)/np.array(sal_cnt).sum()
p = np.repeat(1,len(sal_cnt))/len(sal_cnt)
q = p+alpha
kl1 = p*np.log(p/q)
kl2 = q*np.log(q/p)
surprise = (kl1.sum()+kl2.sum())/2

print("contrast-based salience:%f"%contrast)
print("surprise-based salience:%f"%surprise)