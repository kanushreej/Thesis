import json
import os
import pandas as pd
import shutil
import tkinter as tk
from tkinter import messagebox

original_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Labelling data/ClimateChangeUK_data_0.1%.csv' # PLEASE CHANGE FILENAME IF NEEDED
copy_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Adam/ClimateChangeUK_labelled_Adam.csv' # PLEASE CHANGE FILENAME IF NEEDED
progress_file = '/Users/adamzulficar/Documents/year3/Bachelor Project/Thesis/Annotation/UK/Adam/ClimateChangeUK_progress.json'

if not os.path.exists(copy_file):
    shutil.copy(original_file, copy_file)
    print("File copied successfully.")

df = pd.read_csv(copy_file)
pd.set_option('display.max_colwidth', None)

new_columns = [
    'pro_brexit','anti_brexit',
    'pro_climateAction','anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',    
    'increase_tax', 'decrease_tax',  
    'neutral', 'irrelevant'  
]

for column in new_columns:
    if column not in df.columns:
        df[column] = float('nan')

# Sorting data by subreddit and then keyword 
df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(copy_file, index=False)

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)["last_index"]
    return 0

last_index = load_progress()
start_index = last_index if last_index < len(df) else 0

def save_progress(last_index):
    with open(progress_file, 'w') as f:
        json.dump({"last_index": last_index}, f)

class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Labeling Tool")
        self.root.geometry('1200x800')  # Set window size to 1200x800 pixels
        self.root.configure(bg='gray')
        self.df = df
        self.new_columns = new_columns
        self.start_index = start_index
        self.current_index = self.start_index

        self.selected_labels = {col: None for col in self.new_columns}
        self.create_widgets()
        self.display_current_data()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, bg='gray')
        self.scroll_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)

        self.frame = tk.Frame(self.canvas, bg='gray')
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')
        self.frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll_y.grid(row=0, column=1, sticky="ns")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.labels = {}

        for i, column in enumerate(self.df.columns):
            if column not in self.new_columns:
                label = tk.Label(self.frame, text=column, bg='gray', fg='white', wraplength=1000, justify='left', anchor='w')
                label.grid(row=i, column=0, columnspan=2, sticky="w")
                self.labels[column] = label

        self.buttons_frame = tk.Frame(self.frame, bg='gray')
        self.buttons_frame.grid(row=0, column=2, rowspan=len(self.df.columns), sticky="ns")

        self.label_buttons = {}
        for i, column in enumerate(self.new_columns):
            label_frame = tk.Frame(self.buttons_frame, bg='gray')
            label_frame.grid(row=i, column=0, sticky="ew", pady=5)

            label = tk.Label(label_frame, text=column, bg='gray', fg='white', anchor='w')
            label.pack(side=tk.LEFT)

            relevant_button = tk.Button(label_frame, text="Yes", command=lambda col=column: self.set_label(col, 1))
            relevant_button.pack(side=tk.LEFT, padx=5)

            irrelevant_button = tk.Button(label_frame, text="No", command=lambda col=column: self.set_label(col, 0))
            irrelevant_button.pack(side=tk.LEFT, padx=5)

            self.label_buttons[column] = (relevant_button, irrelevant_button)

        self.done_button = tk.Button(self.buttons_frame, text="Done", command=self.mark_labels)
        self.done_button.grid(row=len(self.new_columns), column=0, pady=20, sticky="ew")

        self.prev_button = tk.Button(self.buttons_frame, text="← Previous", command=self.prev_data)
        self.prev_button.grid(row=len(self.new_columns) + 1, column=0, pady=10, sticky="ew")

        self.next_button = tk.Button(self.buttons_frame, text="Next →", command=self.next_data)
        self.next_button.grid(row=len(self.new_columns) + 2, column=0, pady=10, sticky="ew")

        self.counter_label = tk.Label(self.frame, text=f"Data Point: {self.current_index + 1}/{len(self.df)}", bg='gray', fg='white')
        self.counter_label.grid(row=len(self.df.columns), column=0, columnspan=3, pady=10)

    def display_current_data(self):
        row = self.df.iloc[self.current_index]
        for column, value in row.items():
            if column in self.labels:
                self.labels[column].config(text=f"{column}: {value}")
        self.reset_label_buttons()
        self.counter_label.config(text=f"Data Point: {self.current_index + 1}/{len(self.df)}")

    def reset_label_buttons(self):
        for column in self.new_columns:
            value = self.df.at[self.current_index, column]
            self.selected_labels[column] = value
            relevant_button, irrelevant_button = self.label_buttons[column]
            if value == 1:
                relevant_button.config(relief=tk.SUNKEN, bg='light green')
                irrelevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')
            elif value == 0:
                relevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')
                irrelevant_button.config(relief=tk.SUNKEN, bg='light green')
            else:
                relevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')
                irrelevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')

    def set_label(self, column, value):
        self.selected_labels[column] = value
        relevant_button, irrelevant_button = self.label_buttons[column]
        if value == 1:
            relevant_button.config(relief=tk.SUNKEN, bg='light green')
            irrelevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')
        else:
            relevant_button.config(relief=tk.RAISED, bg='SystemButtonFace')
            irrelevant_button.config(relief=tk.SUNKEN, bg='light green')

    def mark_labels(self):
        for column, value in self.selected_labels.items():
            if value is not None:
                self.df.at[self.current_index, column] = value
        self.df.to_csv(copy_file, index=False)
        self.next_data()

    def prev_data(self):
        self.current_index = (self.current_index - 1) % len(self.df)
        self.display_current_data()

    def next_data(self):
        self.current_index = (self.current_index + 1) % len(self.df)
        self.display_current_data()

    def quit(self):
        save_progress(self.current_index)
        messagebox.showinfo("Progress Saved", "Progress saved successfully!")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
