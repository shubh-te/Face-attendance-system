from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_key" # Required for flash messages

# Recognition confidence threshold (lower is more confident). Increase to allow
# slightly higher values to be accepted if your model is noisy.
# Reduced to 45 for maximum accuracy to avoid wrong attendance.
RECOG_CONF_THRESHOLD = 45


def train_from_images():
    """Train LBPH recognizer from images in TrainingImage/ and save to Training/trainer.yml.
    Returns (success: bool, message: str)."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists('TrainingImage'):
        return False, 'TrainingImage folder not found.'

    imageFiles = [f for f in os.listdir('TrainingImage') if f.lower().endswith(valid_ext)]
    faces = []
    Ids = []
    for fname in imageFiles:
        path = os.path.join('TrainingImage', fname)
        try:
            pilImage = Image.open(path).convert('L')
            imageNp = np.array(pilImage, 'uint8')
        except Exception:
            continue

        uid = None
        parts = fname.split('.')
        for p in parts:
            if p.isdigit():
                uid = int(p)
                break
        if uid is None:
            continue

        faces.append(imageNp)
        Ids.append(uid)

    if len(faces) == 0:
        return False, 'No valid training images found.'

    try:
        os.makedirs('Training', exist_ok=True)
        recognizer.train(faces, np.array(Ids, dtype='int32'))
        recognizer.save('Training/trainer.yml')
        return True, 'Model trained and saved.'
    except Exception as e:
        return False, f'Training error: {e}'

@app.route('/')
def index():
    month = request.args.get('month')
    date_val = request.args.get('date')
    records = []
    
    # Get all available months from the Attendance folder
    attendance_dir = "Attendance"
    available_months = set()
    if os.path.exists(attendance_dir):
        for file in os.listdir(attendance_dir):
            if file.startswith("Attendance_") and file.endswith(".csv"):
                try:
                    date_part = file.split('_')[1].split('.')[0] # YYYY-MM-DD
                    month_val = date_part[:7] # YYYY-MM
                    available_months.add(month_val)
                except Exception:
                    pass
                    
    available_months = sorted(list(available_months), reverse=True)
    month_options = []
    for m in available_months:
        try:
            lbl = datetime.strptime(m, "%Y-%m").strftime("%B %Y")
            month_options.append({"value": m, "label": lbl})
        except:
            month_options.append({"value": m, "label": m})
    
    if date_val:
        # Fetch specific date data
        table_title = f"Attendance Log for {date_val}"
        attendance_file = f"Attendance/Attendance_{date_val}.csv"
        if os.path.exists(attendance_file):
            try:
                df = pd.read_csv(attendance_file)
                if 'Time' in df.columns:
                    df = df.sort_values(by=['Time'], ascending=False)
                records = df.to_dict('records')
            except Exception:
                pass
    elif month:
        # Fetch monthly data
        attendance_dir = "Attendance"
        try:
            display_month = datetime.strptime(month, "%Y-%m").strftime("%B %Y")
        except:
            display_month = month
        table_title = f"Attendance Log for {display_month}"
        if os.path.exists(attendance_dir):
            all_dfs = []
            for file in os.listdir(attendance_dir):
                if file.startswith(f"Attendance_{month}") and file.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(attendance_dir, file))
                        all_dfs.append(df)
                    except Exception:
                        pass
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                # Sort by Date and Time so the data appears date-wise
                if 'Date' in combined_df.columns:
                    if 'Time' in combined_df.columns:
                        combined_df = combined_df.sort_values(by=['Date', 'Time'], ascending=[False, False])
                    else:
                        combined_df = combined_df.sort_values(by=['Date'], ascending=False)
                records = combined_df.to_dict('records')
    else:
        # Default: today's data
        date_str = datetime.now().strftime("%Y-%m-%d")
        table_title = "Today's Attendance Log"
        attendance_file = f"Attendance/Attendance_{date_str}.csv"
        
        if os.path.exists(attendance_file):
            try:
                df = pd.read_csv(attendance_file)
                records = df.to_dict('records')
            except Exception:
                pass
                
    for r in records:
        if str(r.get('Date', '')) != 'nan' and r.get('Date'):
            try:
                r['Month'] = datetime.strptime(str(r['Date']), "%Y-%m-%d").strftime("%B")
            except:
                r['Month'] = ''
        else:
            r['Month'] = ''
        
    return render_template('index.html', records=records, table_title=table_title, selected_month=month, selected_date=date_val, available_months=month_options)

# Details feature removed per user request. Routes and templates for details,
# image serving and CSV download have been removed.

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    
    if name:
        try:
            os.makedirs("StudentDetails", exist_ok=True)
            db_path = 'StudentDetails/student_details.csv'
            
            uid = 1
            if os.path.isfile(db_path):
                try:
                    df_students = pd.read_csv(db_path)
                    if not df_students.empty and 'Id' in df_students.columns:
                        uid = int(df_students['Id'].max()) + 1
                except Exception:
                    pass
                    
            # CAP_DSHOW windows p jaldi camera kholne me help karta hai
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
                elif sampleNum >= 10: # Ab 10 photo capture hogi aur automatic train ho jayegi
                    break
                    
            cam.release()
            cv2.destroyAllWindows()
            
            # Save student info
            row = [uid, name]
            file_exists = os.path.isfile('StudentDetails/student_details.csv')
            with open('StudentDetails/student_details.csv', 'a+', newline='') as csvFile:
                writer = csv.writer(csvFile)
                if not file_exists:
                    writer.writerow(['Id', 'Name'])
                writer.writerow(row)
            
            # Train the Model
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            # Only include common image extensions
            valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
            imageFiles = [f for f in os.listdir("TrainingImage") if f.lower().endswith(valid_ext)]
            faces = []
            Ids = []

            for fname in imageFiles:
                imagePath = os.path.join("TrainingImage", fname)
                try:
                    pilImage = Image.open(imagePath).convert('L')
                    imageNp = np.array(pilImage, 'uint8')
                except Exception:
                    # skip unreadable files
                    continue

                # Extract id from filename. Expecting pattern like Name.Id.sample.jpg
                uid = None
                parts = fname.split('.')
                # try common positions for numeric id
                for p in parts:
                    if p.isdigit():
                        uid = int(p)
                        break

                if uid is None:
                    # skip files without a parsable id
                    continue

                faces.append(imageNp)
                Ids.append(uid)

            if len(faces) == 0 or len(Ids) == 0:
                flash('Training failed: no valid training images found. Register at least one student image.', 'error')
            else:
                os.makedirs("Training", exist_ok=True)
                try:
                    recognizer.train(faces, np.array(Ids, dtype='int32'))
                    recognizer.save("Training/trainer.yml")
                except Exception as e:
                    flash(f"Training error: {e}", 'error')
            
            flash(f"Successfully Registered and Trained for {name}! (Auto-assigned ID: {uid})", "success")
        except Exception as e:
            flash(f"Registration Error: {e}", "error")
    else:
        flash("Please enter the student Name.", "error")
        
    return redirect(url_for('index'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if not os.path.exists("Training/trainer.yml"):
        flash("AI Model not found. Please register a face first.", "error")
        return redirect(url_for('index'))
        
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Try reading existing trainer; if it's corrupt/unreadable, attempt to rebuild
        try:
            recognizer.read("Training/trainer.yml")
        except Exception:
            ok, msg = train_from_images()
            if not ok:
                flash(f"Model load error and retrain failed: {msg}", 'error')
                return redirect(url_for('index'))
            try:
                recognizer.read("Training/trainer.yml")
            except Exception as e:
                flash(f"Failed to load trainer after retrain: {e}", 'error')
                return redirect(url_for('index'))
        harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        # Load student details; coerce Id to int when possible
        try:
            df = pd.read_csv("StudentDetails/student_details.csv")
            if 'Id' in df.columns:
                df['Id'] = df['Id'].astype(int)
        except Exception:
            df = pd.DataFrame(columns=['Id', 'Name'])

        # Build id->name map for robust lookup
        id_name_map = {}
        if not df.empty and 'Id' in df.columns and 'Name' in df.columns:
            try:
                id_name_map = {int(r['Id']): r['Name'] for _, r in df.iterrows()}
            except Exception:
                id_name_map = {str(r['Id']): r['Name'] for _, r in df.iterrows()}
        
        # CAP_DSHOW taaki bina loading latency ke webcam pop up ho jaye
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        os.makedirs("Attendance", exist_ok=True)
        
        date_today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"Attendance/Attendance_{date_today}.csv"
        
        logged_ids = set()
        if os.path.isfile(attendance_file):
            try:
                df_attendance = pd.read_csv(attendance_file)
                if not df_attendance.empty and 'Id' in df_attendance.columns:
                    try:
                        logged_ids = set(df_attendance['Id'].astype(int).values)
                    except Exception:
                        logged_ids = set(df_attendance['Id'].values)
            except Exception:
                pass
                
        attendance_marked = False
        
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

                # Normalize predicted id to int when possible
                try:
                    pred_id = int(Id)
                except Exception:
                    pred_id = Id

                # Lower `conf` means better match for LBPH. Use configurable threshold.
                if conf < RECOG_CONF_THRESHOLD:
                    # Map predicted Id to student name robustly using id_name_map
                    name = None
                    try:
                        name = id_name_map.get(pred_id)
                    except Exception:
                        name = None

                    if not name:
                        try:
                            # try direct dataframe lookup as fallback
                            name_row = df.loc[df['Id'] == pred_id]['Name'].values
                            name = name_row[0] if len(name_row) > 0 else None
                        except Exception:
                            name = None

                    name = name if name else "Unknown"

                    if pred_id not in logged_ids:
                        display_text = f"{pred_id} - {name}"
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        time_str = datetime.now().strftime("%H:%M:%S")
                        month_str = datetime.now().strftime("%B")
                        row = [pred_id, name, date_str, time_str, month_str]

                        attendance_file = f"Attendance/Attendance_{date_str}.csv"
                        file_exists = os.path.isfile(attendance_file)
                        with open(attendance_file, 'a+', newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            if not file_exists:
                                writer.writerow(['Id', 'Name', 'Date', 'Time', 'Month'])
                            writer.writerow(row)
                        try:
                            logged_ids.add(int(pred_id))
                        except Exception:
                            logged_ids.add(pred_id)
                        attendance_marked = True
                    else:
                        # Agar pehle hi attendance lag gayi hai toh (Marked) dikhayega
                        display_text = f"{Id} - {name} (Marked)"
                        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2) # Color halka cyan ho jayega

                else:
                    display_text = "Unknown"
                    
                cv2.putText(im, str(display_text), (x, y + h + 20), font, 0.8, (255, 255, 255), 2)
                
            cv2.imshow('Face Attendance Viewer (Press Q to exit)', im)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cam.release()
        cv2.destroyAllWindows()
        
        if attendance_marked:
            flash("Attendance marked successfully!", "success")
        else:
            flash("No new face detected or attendance already recorded.", "success")
            
    except Exception as e:
        flash(f"Error: {e}", "error")
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
