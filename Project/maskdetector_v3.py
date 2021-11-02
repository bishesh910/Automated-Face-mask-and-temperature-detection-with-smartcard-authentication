def main():
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    from imutils.video import VideoStream
    
    import numpy as np
    import argparse
    import imutils
    import time
    import cv2
    import os
    from picamera.array import PiRGBArray
    from datetime import date
    from datetime import datetime
    import mysql.connector
    import sys
    import RPi.GPIO as gpio
    from mfrc522 import SimpleMFRC522
    from smbus2 import SMBus
    from mlx90614 import MLX90614
    
    try:
        con = mysql.connector.connect(host='localhost', user='bishesh', passwd='1234',database='scard')
        cur = con.cursor()

    except Exception as err:
        print("Error while creating connection", err)
    

    CardReader = SimpleMFRC522()
    print ('[INFO] Please scan your card..')
    try:
        id, text = CardReader.read()
        a = id
        b=[]
        print('[INFO] LOADING....')
        sql_query = (f"select name,role from data where aes_decrypt(UserID,'Tab69070#') = {a} ")
        cur.execute(sql_query, a)
        result = cur.fetchone()

        for row in result:
            b.append(row)
        print(b)   
    except Exception as err:

        print("[ERROR] Invalid Card")
        cv2.destroyAllWindows()
        main()
        
    finally:
        gpio.cleanup()



    
    #For Temperature


    print("[INFO] MEASURE TEMPERATURE IN:")
    for i in range(15,0,-1):
        print(i)
        time.sleep(1)
    
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    temp = 0
    Cel = sensor.get_object_1()
    temp = ((Cel*9/5)+32)/100
    temp = temp*10000/100
    temp = round(temp,2)
    temp = temp + 7
        
    if temp >100.4:
        print("[WARNING] HIGH TEMPERATURE")
        print("[INFO] ACCESS DENIED")
        main()
    
    print (f'TEMPERATURE : {temp}')
    bus.close()
    


    #For detecting facearea mask area and prediction with confidence
    def detect_and_predict_mask(frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))


        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")


                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)


                faces.append(face)
                locs.append((startX, startY, endX, endY))


        if len(faces) > 0:

            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)


        return (locs, preds)


    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())


    print("[INFO] LOADING FACEDETECTION MODEL...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


    print("[INFO] LOADING TRAINED MODEL...")
    maskNet = load_model(args["model"])
    

    print("[INFO] WARMING UP CAMERA")
    vs = VideoStream(usePiCamera=True,framerate=36).start()
    
    
    time.sleep(2)


    #For date and time of logs
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    today = date.today()
    sec = time.time()
    now_time = time.ctime(sec)
    img_counter=0
    key = cv2.waitKey(1) & 0xFF

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        


        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            
            if mask > withoutMask:
                label = "MASK DETECTED"
                color = (0, 255, 0)
                if key%256==32:
                    frame1=vs.read()
                    frame1 = imutils.resize(frame, width=600)
                    label = "MASK DETECTED"
                    color = (0, 255, 0)
                    
                    cv2.putText(frame1,f'Name:{b[0]}',(5,20),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                    cv2.putText(frame1,f'Role:{b[1]}',(5,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                    cv2.putText(frame1,f'Temp:{temp}F',(5,80),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                   
                    img_name = f"/home/pi/mask/snap-logs/ D {today} T {current_time}.jpg".format(img_counter)
                    cv2.imwrite(img_name,frame1)
                    print("ACCESS GRANTED")
                    cv2.imshow("Preview",frame1)
                    
                    
                    cv2.destroyAllWindows()
                    vs.stop()
                    main()
                    
                    

            else:
                if key%256==32:
                    print("[INFO] CANNOT SNAPSHOT WITHOUT MASK..")
                label = "NO MASK DETECTED"
                color = (0, 0, 255)


            cv2.putText(frame, label, (startX-50, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        cv2.imshow("Face Mask Detector", frame)
        key = cv2.waitKey(1) & 0xFF


        if key ==ord("q"):
            break
            
            

    cv2.destroyAllWindows()
    vs.stop()

main()