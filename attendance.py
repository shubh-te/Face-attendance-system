import cv2
import os
import csv
import pandas as pd
from datetime import datetime

def mark_attendance():
    if not os.path.exists("Training/trainer.yml"):
        raise Exception("Pehle koi face register karein (Trainer file nahi mili).")
        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Training/trainer.yml")
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    try:
        df = pd.read_csv("StudentDetails/student_details.csv")
    except FileNotFoundError:
        raise Exception("Student details nahi mile. Kripya pehle register karein.")
        
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    os.makedirs("Attendance", exist_ok=True)
    
    print("Camera on hai. Face detect ho raha hai. Band karne ke liye 'q' dabayein.")
    
    # Track logged IDs to avoid multiple entries in one session (optional, but good)
    logged_ids = set()
    
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            # Confidence check (Lower is better in LBPH)
            if conf < 60:
                name_row = df.loc[df['Id'] == Id]['Name'].values
                name = name_row[0] if len(name_row) > 0 else "Unknown"
                display_text = f"{Id} - {name}"
                
                # Agar aaj ki attendance list me nahi hai toh append karo
                if Id not in logged_ids:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    time_str = datetime.now().strftime("%H:%M:%S")
                    row = [Id, name, date_str, time_str]
                    
                    attendance_file = f"Attendance/Attendance_{date_str}.csv"
                    file_exists = os.path.isfile(attendance_file)
                    with open(attendance_file, 'a+', newline='') as csvFile:
                        writer = csv.writer(csvFile)
                        if not file_exists:
                            writer.writerow(['Id', 'Name', 'Date', 'Time'])
                        writer.writerow(row)
                    logged_ids.add(Id)
                    print(f"[{time_str}] {name} ki attendance lag gayi!")
            else:
                display_text = "Unknown"
                
            cv2.putText(im, str(display_text), (x, y + h + 20), font, 0.8, (255, 255, 255), 2)
            
        cv2.imshow('Face Attendance Viewer', im)
        
        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()
