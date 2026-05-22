import tkinter as tk
from tkinter import messagebox
import register
import attendance

# Main Application Window Create karna
root = tk.Tk()
root.title("Smart Face Attendance System")
root.geometry("500x350")
root.configure(bg="#f0f0f0")

def handle_register():
    uid = id_entry.get().strip()
    name = name_entry.get().strip()
    
    if uid and name:
        try:
            uid = int(uid)
            messagebox.showinfo("Wait", "Camera khulega, kripya thoda wait karein. Register hone de.")
            # 1. Image Capture karo
            register.take_images(uid, name)
            # 2. Model ko train karo
            register.train_images()
            
            messagebox.showinfo("Success", f"{name} ko successfully register aur train kar liya gaya!")
            
            # Input clear kar do
            id_entry.delete(0, 'end')
            name_entry.delete(0, 'end')
        except ValueError:
            messagebox.showerror("Error", "Kripya ID sirf number (int) mein enter karein.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showwarning("Warning", "Kripya User ID aur Name dono fields bharein.")

def handle_attendance():
    try:
        messagebox.showinfo("Started", "Scanner shuru ho gaya hai. Camera close karne ke liye 'q' dabayein.")
        attendance.mark_attendance()
        messagebox.showinfo("Done", "Attendance Mark ho gayi, file Attendance/ folder mein check karein CSV.")
    except Exception as e:
        messagebox.showerror("Error", f"Kuch gadbad hai ya phir model train nahi hua:\n\n{str(e)}")


# ------------------ UI ELEMENTS ------------------ #
title_label = tk.Label(root, text="Face Recognition Attendance", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=15)

# Input frame
input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(pady=10)

tk.Label(input_frame, text="User ID (eg: 1, 2, 3):", bg="#f0f0f0", font=("Helvetica", 11)).grid(row=0, column=0, padx=10, pady=5)
id_entry = tk.Entry(input_frame, font=("Helvetica", 11))
id_entry.grid(row=0, column=1)

tk.Label(input_frame, text="Name:", bg="#f0f0f0", font=("Helvetica", 11)).grid(row=1, column=0, padx=10, pady=5)
name_entry = tk.Entry(input_frame, font=("Helvetica", 11))
name_entry.grid(row=1, column=1)

# Buttons frame
btn_frame = tk.Frame(root, bg="#f0f0f0")
btn_frame.pack(pady=20)

reg_btn = tk.Button(btn_frame, text="1. Register New Face", command=handle_register, 
                    bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
reg_btn.grid(row=0, column=0, padx=10)

att_btn = tk.Button(btn_frame, text="2. Mark Attendance", command=handle_attendance, 
                    bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), width=20, height=2)
att_btn.grid(row=0, column=1, padx=10)

footer_label = tk.Label(root, text="Designed for College Project", font=("Helvetica", 9, "italic"), bg="#f0f0f0", fg="gray")
footer_label.pack(side="bottom", pady=10)

if __name__ == "__main__":
    root.mainloop()
