import cv2
import os
import csv
import numpy as np
from PIL import Image

def take_images(id, name):
    cam = cv2.VideoCapture(0)
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    
    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)
    
    print(f"[{name}] ki images lena shuru kar rahe hain. Camera ki taraf dekhein...")
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1
            # Chehre ka sample save karein
            cv2.imwrite(f"TrainingImage/{name}.{id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('Face Recognition Registration', img)
            
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum >= 60: # 60 samples lenge accurate training ke liye
            break
            
    cam.release()
    cv2.destroyAllWindows()
    
    # Details CSV me save karein
    row = [id, name]
    file_exists = os.path.isfile('StudentDetails/student_details.csv')
    with open('StudentDetails/student_details.csv', 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        if not file_exists:
            writer.writerow(['Id', 'Name'])
        writer.writerow(row)
    print("Images successfully capture ho gayi hain.")

def train_images():
    print("AI Model Train ho raha hai, kripya intezaar karein...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    imagePaths = [os.path.join("TrainingImage", f) for f in os.listdir("TrainingImage")]
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
        
    os.makedirs("Training", exist_ok=True)
    recognizer.train(faces, np.array(Ids))
    recognizer.save("Training/trainer.yml")
    print("Training complete ho gayi. Model save ho gaya.")
