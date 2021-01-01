import cv2

thres = 0.5
# img = cv2.imread('lena.jpg')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames= []
 
calssFile = 'data\\coco.names'

with open(calssFile,'rt') as f:

    classNames = f.read().rstrip('\n').split('\n')

configPath = 'data\\ssdlite_mobilenet_v3_small_320x320_coco.pbtext'

weightsPath = 'data\\frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)

net.setInputScale(1.0/ 127.5)

net.setInputMean((127.5, 127.5, 127.5))

net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)

            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1 ,(0,255,0),2)
            
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1 ,(0,255,0),2)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
