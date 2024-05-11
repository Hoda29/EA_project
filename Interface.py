import tkinter as tk
import subprocess

def open_algo():
    root.destroy()
    subprocess.run(["python", "C:/Users/Mohamed/Desktop/EA/EA.py"])
    
def quit():
    root.destroy()


root = tk.Tk()
root.title("Genetic Algorithm & Differential Evolution")

root.geometry("1920x1080")


root.configure(bg="#130a2e")

main_menu_frame = tk.Frame(root, width=910, height=780, padx=20, pady=20, bg="#130a2e")
main_menu_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


title_label = tk.Label(main_menu_frame, text="Genetic Algorithm\n & \nDifferential Evolution", font=("Helvetica", 60, "bold"), bg="#130a2e", fg="white")
title_label.grid(row=0, column=0, pady=(10, 100))

play_button = tk.Button(main_menu_frame, width=8, height=1, text="Start", font=("Helvetica", 30, "bold"), bg="#562fd0", fg="white", command=open_algo,  relief=tk.GROOVE, borderwidth=10)
play_button.grid(row=1, column=0, pady=30)

quit_button = tk.Button(main_menu_frame, width=8, height=1, text="Quit", font=("Helvetica", 30, "bold"), bg="#562fd0", fg="white", command=quit,  relief=tk.GROOVE, borderwidth=10)
quit_button.grid(row=2, column=0, pady=10)


root.mainloop()
