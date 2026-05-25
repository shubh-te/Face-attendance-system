import streamlit as st
import cv2
import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="Smart Face Attendance", page_icon="👨‍🎓", layout="centered")

st.title("👨‍🎓 Smart Face Attendance System")

# Sidebar Menu
menu = st.sidebar.selectbox("Choose Option", ["Home", "Register User", "Mark Attendance", "View Attendance"])

if menu == "Home":
    st.write("Welcome to the Face Recognition Attendance System.")
    st.info("Kripya Sidebar se option choose karein: Naya user register karna ho ya attendance mark karna ho.")

elif menu == "Register User":
    st.header("Register New Face")
    uid = st.text_input("Enter User ID (Number form mein, jaise: 1, 2, 3)", "")
    name = st.text_input("Enter Name", "")
    
    if st.button("Start Camera & Register"):
        if uid and name:
            try:
                uid = int(uid)
                st.info("Naya camera window khuega... Camera ki taraf dekhein! (Process 10 photos lega aur turant train karega)")
                
                cam = cv2.VideoCapture(0)
                harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                detector = cv2.CascadeClassifier(harcascadePath)
                sampleNum = 0
                
                os.makedirs("TrainingImage", exist_ok=True)
                os.makedirs("StudentDetails", exist_ok=True)
                
                while True:
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        sampleNum += 1
                        cv2.imwrite(f"TrainingImage/{name}.{uid}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                        cv2.imshow('Registration - Capturing Face (Press Q to exit)', img)
                        
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    elif sampleNum >= 10: # 10 samples lekar turant train karega
                        break
                        
                cam.release()
                cv2.destroyAllWindows()
                
                # Save Details to CSV
                row = [uid, name]
                file_exists = os.path.isfile('StudentDetails/student_details.csv')
                with open('StudentDetails/student_details.csv', 'a+', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    if not file_exists:
                        writer.writerow(['Id', 'Name'])
                    writer.writerow(row)
                    
                st.success(f"Pictures captured successfully for {name}!")
                
                # Train the Model
                with st.spinner("AI Model Train ho raha hai, wahi rukiye..."):
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
                
                st.success("Registration aur Training Complete ho gayi!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error aayi hai: {e}")
        else:
            st.warning("Kripya valid User ID aur Name daalein.")

elif menu == "Mark Attendance":
    st.header("Mark Attendance")
    if st.button("Open Camera for Attendance"):
        if not os.path.exists("Training/trainer.yml"):
            st.error("AI Model nahi mila. Pehle kisi ko register karein.")
        else:
            st.info("Scanner khul raha hai! Window band karne ke liye 'q' press karein.")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("Training/trainer.yml")
            harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(harcascadePath)
            
            try:
                df = pd.read_csv("StudentDetails/student_details.csv")
                
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                os.makedirs("Attendance", exist_ok=True)
                
                logged_ids = set()
                attendance_marked = False
                
                while True:
                    ret, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        
                        # Confidence (Distance) kam hogi utna better match hai.
                        # Reduced to 45 for maximum accuracy to avoid wrong attendance.
                        if conf < 45:
                            name_row = df.loc[df['Id'] == Id]['Name'].values
                            name = name_row[0] if len(name_row) > 0 else "Unknown"
                            display_text = f"{Id} - {name}"
                            
                            # Agar user ne abhi tak attendance nahi lagwai is session me
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
                                attendance_marked = True
                        else:
                            display_text = "Unknown"
                            
                        cv2.putText(im, str(display_text), (x, y + h + 20), font, 0.8, (255, 255, 255), 2)
                        
                    cv2.imshow('Face Attendance Viewer (Press Q to exit)', im)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                cam.release()
                cv2.destroyAllWindows()
                if attendance_marked:
                    st.success("Attendance lag gayi! Aap 'View Attendance' me check kar sakte hain.")
                else:
                    st.warning("Koi naya face detect nahi hua.")
                    
            except FileNotFoundError:
                st.error("Student Details file nahi mili. Kripya pehle kisi ko register karein.")
            except Exception as e:
                st.error(f"Error aaya: {e}")

elif menu == "View Attendance":
    st.header("Today's Attendance List")
    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"Attendance/Attendance_{date_str}.csv"
    
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        if len(df) > 0:
            st.success("Aaj ki attendance list:")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Aaj ki file hai but koi attendance nahi hai.")
    else:
        st.info("Aaj ka koi attendance record nahi mila.")
