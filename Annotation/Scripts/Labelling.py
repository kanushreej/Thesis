import json
import os
import pandas as pd
import shutil
import tkinter as tk
from tkinter import messagebox

# File paths
original_file = 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Annotation/US/Labelling data/ClimateChangeUS_data_0.1%.csv' # PLEASE CHANGE FILENAME IF NEEDED
copy_file = 'C:/Users/rapha/Documents/CS_VU/Thesis/Thesis/Annotation/US/Raphael/ClimateChangeLabelled.csv' # PLEASE CHANGE FILENAME IF NEEDED
progress_file = 'progress.json'

# Copy file if not exists
if not os.path.exists(copy_file):
    shutil.copy(original_file, copy_file)
    print("File copied successfully.")

# Load CSV into DataFrame
df = pd.read_csv(copy_file)
pd.set_option('display.max_colwidth', None)

# Define new columns
new_columns = [
    'pro_climateAction', 'anti_climateAction',
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',
    'increase_tax', 'decrease_tax',
    'neutral', 'irrelevant'
]

# Add new columns if not present
for column in new_columns:
    if column not in df.columns:
        df[column] = float('nan')

# Sorting data by subreddit and then keyword 
df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Function to load progress
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)["last_index"]
    return 0

last_index = load_progress()
start_index = last_index if last_index < len(df) else 0

# Function to save progress
def save_progress(last_index):
    with open(progress_file, 'w') as f:
        json.dump({"last_index": last_index}, f)

# Tkinter application class
class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Labeling Tool")
        self.root.geometry('1200x800')  # Set window size to 1200x800 pixels
        self.root.configure(bg='#2d2d2d')  # Dark background color
        self.df = df
        self.new_columns = new_columns
        self.start_index = start_index
        self.current_index = self.start_index

        self.selected_labels = {col: None for col in self.new_columns}
        self.parent_post_index = self.find_parent_post_index()
        self.create_widgets()
        self.display_current_data()

    def create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg='#2d2d2d')
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg='#2d2d2d')
        self.scroll_y = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.scroll_frame = tk.Frame(self.canvas, bg='#2d2d2d')
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.labels = {}
        for i, column in enumerate(self.df.columns):
            if column not in self.new_columns:
                label = tk.Label(self.scroll_frame, text=column, bg='#2d2d2d', fg='#ffffff', wraplength=1000, justify='left', anchor='w')
                label.grid(row=i, column=0, columnspan=2, sticky="w")
                self.labels[column] = label

        self.parent_label_frame = tk.Frame(self.scroll_frame, bg='#1e1e1e')
        self.parent_label_frame.grid(row=len(self.df.columns), column=0, columnspan=2, sticky="ew", pady=5)
        self.parent_label = tk.Label(self.parent_label_frame, text="", bg='#1e1e1e', fg='#ffffff', wraplength=1000, justify='left', anchor='w', font=('Arial', 12, 'bold'))
        self.parent_label.pack(fill='x', padx=5, pady=5)

        self.comment_label_frame = tk.Frame(self.scroll_frame, bg='#2d2d2d')
        self.comment_label_frame.grid(row=len(self.df.columns) + 1, column=0, columnspan=2, sticky="ew", pady=5)
        self.comment_label = tk.Label(self.comment_label_frame, text="", bg='#2d2d2d', fg='#ffffff', wraplength=1000, justify='left', anchor='w', font=('Arial', 12))
        self.comment_label.pack(fill='x', padx=5, pady=5)

        self.buttons_frame = tk.Frame(self.main_frame, bg='#2d2d2d')
        self.buttons_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.label_buttons = {}
        for i, column in enumerate(self.new_columns):
            label_frame = tk.Frame(self.buttons_frame, bg='#2d2d2d')
            label_frame.grid(row=i, column=0, sticky="ew", pady=5)

            label = tk.Label(label_frame, text=column, bg='#2d2d2d', fg='#ffffff', anchor='w')
            label.pack(side=tk.LEFT)

            relevant_button = tk.Button(label_frame, text="Yes", bg='#007acc', fg='#ffffff', command=lambda col=column: self.set_label(col, 1))
            relevant_button.pack(side=tk.LEFT, padx=5)

            irrelevant_button = tk.Button(label_frame, text="No", bg='#d9534f', fg='#ffffff', command=lambda col=column: self.set_label(col, 0))
            irrelevant_button.pack(side=tk.LEFT, padx=5)

            self.label_buttons[column] = (relevant_button, irrelevant_button)

        self.done_button = tk.Button(self.buttons_frame, text="Done", bg='#5cb85c', fg='#ffffff', command=self.mark_labels)
        self.done_button.grid(row=len(self.new_columns), column=0, pady=20, sticky="ew")

        self.prev_button = tk.Button(self.buttons_frame, text="← Previous", bg='#f0ad4e', fg='#ffffff', command=self.prev_data)
        self.prev_button.grid(row=len(self.new_columns) + 1, column=0, pady=10, sticky="ew")

        self.next_button = tk.Button(self.buttons_frame, text="Next →", bg='#5bc0de', fg='#ffffff', command=self.next_data)
        self.next_button.grid(row=len(self.new_columns) + 2, column=0, pady=10, sticky="ew")

        self.counter_label = tk.Label(self.main_frame, text=f"Data Point: {self.current_index + 1}/{len(self.df)}", bg='#2d2d2d', fg='#ffffff')
        self.counter_label.pack(side=tk.BOTTOM, pady=10)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def display_current_data(self):
        row = self.df.iloc[self.current_index]
        if row['type'] == 'post':
            self.parent_post_index = self.current_index

        if self.parent_post_index is not None:
            parent_post = self.df.iloc[self.parent_post_index]
            parent_text = f"Parent Post - {parent_post['author']}: {parent_post['body']}"
            self.parent_label.config(text=parent_text)
        else:
            self.parent_label.config(text="")

        comment_text = f"Comment - {row['author']}: {row['body']}" if row['type'] == 'comment' else parent_text
        self.comment_label.config(text=comment_text)

        specific_columns = ['subreddit', 'keyword', 'id', 'author', 'created_utc']
        for column in specific_columns:
            value = row[column]
            if column in self.labels:
                self.labels[column].config(text=f"{column}: {value}")

        self.reset_label_buttons()
        self.counter_label.config(text=f"Data Point: {self.current_index + 1}/{len(self.df)}")


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
        if self.df.iloc[self.current_index]['type'] == 'post':
            self.parent_post_index = self.current_index
        self.display_current_data()

    def next_data(self):
        self.current_index = (self.current_index + 1) % len(self.df)
        if self.df.iloc[self.current_index]['type'] == 'post':
            self.parent_post_index = self.current_index
        self.display_current_data()

    def find_parent_post_index(self):
        for idx in range(self.current_index, -1, -1):
            if self.df.iloc[idx]['type'] == 'post':
                return idx
        return None

    def quit(self):
        save_progress(self.current_index)
        messagebox.showinfo("Progress Saved", "Progress saved successfully!")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()
