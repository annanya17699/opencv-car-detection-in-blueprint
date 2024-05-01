import cv2 as cv
from inference_sdk import InferenceHTTPClient

def getPredictions():
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="5mhIEl9IiNAbnEpOkjUJ"
    )

    result = CLIENT.infer('./parkbp2.jpg', model_id="cars---overhead-view/3")
    return result

img = cv.imread('./parkbp2.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

recRes = getPredictions()

def getCoordinates(recRes, img):
    predictions = []
    for key in recRes:
        if(key == 'predictions'):
            predictions = recRes[key]
    arr = []
    for value in predictions:
        x,y,width,height,flag = '','','','',False
        for key in value:
            if(key == 'confidence' and float(value[key]) < 0.7):
                flag = True
                break
            if(key == 'x'):
                x = value[key]
            if(key == 'y'):
                y = value[key]
            if(key == 'width'):
                width = value[key]
            if(key == 'height'):
                height = value[key]
        if(flag==False):
            cv.rectangle(img, (round(x)-17,round(y)-30), (round(x)+round(width)-17, round(y)+round(height)-30),(0,255,0), thickness=1)
            obj = {
                "x" : x-17,
                "y" : y-30,
                "x1": x+width-17,
                "y1": y+height-30
            }
            arr.append(obj)


coordinates = getCoordinates(recRes,img)

cv.imshow('Detected cars',img)

cv.waitKey(0)