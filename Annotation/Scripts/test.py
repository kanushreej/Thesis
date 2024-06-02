import json
import os
import pandas as pd
import shutil
import tkinter as tk
from tkinter import messagebox

# File paths
original_data_file = '/mnt/e/Clone/Thesis6011/Thesis/Subreddit Data/US/TaxationUS_data.csv' 
sample_data_file = '/mnt/e/Clone/Thesis6011/Thesis/Annotation/US/Trials/TaxationUS_trial_sample.csv' 
copy_file = '/mnt/e/Clone/Thesis6011/Thesis/Annotation/US/Zijing/TaxationUS_labelled_trial.csv' 
progress_file = 'progress.json'

if not os.path.exists(copy_file):
    shutil.copy(sample_data_file, copy_file)
    print("File copied successfully.")

df = pd.read_csv(copy_file)
pd.set_option('display.max_colwidth', None)

new_columns = [
    'pro_climateAction', 'anti_climateAction', 
    'public_healthcare', 'private_healthcare',
    'pro_israel', 'pro_palestine',
    'increase_tax', 'decrease_tax',
    'neutral', 'irrelevant'
]

for column in new_columns:
    if column not in df.columns:
        df[column] = float('nan')

df.sort_values(by=['subreddit', 'keyword', 'created_utc'], inplace=True)
df.reset_index(drop=True, inplace=True)

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

original_df = pd.read_csv(original_data_file, dtype={'id': str, 'parent_id': str}, encoding='utf-8')
parent_content_dict = original_df.set_index('id')['body'].to_dict()

def get_full_thread(comment_id, parent_content_dict, original_df):
    thread = []
    current_id = comment_id
    
    while current_id:
        if current_id not in parent_content_dict:
            thread.insert(0, f"Post with ID {current_id} not found")
            break
        
        comment = parent_content_dict.get(current_id, "Content not available")
        thread.insert(0, f"ID: {current_id}\n{str(comment)}") 
        
        row = original_df.loc[original_df['id'] == current_id]
        if row.empty:
            break
        parent_id = row.iloc[0]['parent_id']
        
        if pd.isna(parent_id) or parent_id is None:
            break
        
        current_id = str(parent_id[3:]) 
    
    return thread

class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Labeling Tool")
        self.root.geometry('1500x900') 
        self.root.configure(bg='gray')
        self.df = df
        self.new_columns = new_columns
        self.start_index = start_index
        self.current_index = self.start_index
        self.parent_content_dict = parent_content_dict

        self.selected_labels = {col: None for col in self.new_columns}
        self.create_widgets()
        self.display_current_data()

    def create_widgets(self):
        self.left_frame = tk.Frame(self.root, bg='gray')
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.scroll_y = tk.Scrollbar(self.left_frame, orient="vertical")
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_canvas = tk.Canvas(self.left_frame, bg='gray', yscrollcommand=self.scroll_y.set)
        self.text_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.config(command=self.text_canvas.yview)

        self.text_frame = tk.Frame(self.text_canvas, bg='gray')
        self.text_canvas.create_window((0, 0), window=self.text_frame, anchor='nw')

        self.text_frame.bind("<Configure>", lambda e: self.text_canvas.configure(scrollregion=self.text_canvas.bbox("all")))

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.labels = {}

        for i, column in enumerate(self.df.columns):
            if column not in self.new_columns:
                label = tk.Label(self.text_frame, text=column, bg='gray', fg='white', wraplength=1000, justify='left', anchor='w')
                label.grid(row=i, column=0, columnspan=2, sticky="w", pady=5)
                self.labels[column] = label

        # Add label for full thread
        self.full_thread_label = tk.Label(self.text_frame, text="Full Thread", bg='gray', fg='white', wraplength=1000, justify='left', anchor='w')
        self.full_thread_label.grid(row=len(self.df.columns), column=0, columnspan=2, sticky="w", pady=5)

        # Create a new frame for buttons and fix its position
        self.buttons_frame = tk.Frame(self.root, bg='gray')
        self.buttons_frame.grid(row=0, column=1, sticky="ns")

        self.label_buttons = {}
        for i, column in enumerate(self.new_columns):
            label_frame = tk.Frame(self.buttons_frame, bg='gray')
            label_frame.grid(row=i, column=0, sticky="ew", pady=10) 

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

        self.counter_label = tk.Label(self.text_frame, text=f"Data Point: {self.current_index + 1}/{len(self.df)}", bg='gray', fg='white')
        self.counter_label.grid(row=len(self.df.columns) + 1, column=0, columnspan=3, pady=10)

    def display_current_data(self):
        row = self.df.iloc[self.current_index]
        comment_id = row['id']
        full_thread = get_full_thread(comment_id, self.parent_content_dict, self.df)

        for column, value in row.items():
            if column in self.labels:
                self.labels[column].config(text=f"{column}: {value}")

        for widget in self.text_frame.grid_slaves():
            widget.grid_forget()

        for i, column in enumerate(self.df.columns):
            if column in self.labels:
                self.labels[column].grid(row=i, column=0, columnspan=2, sticky="w", pady=5)

        self.full_thread_label.grid(row=len(self.df.columns), column=0, columnspan=2, sticky="w", pady=5)

        if isinstance(full_thread[0], str) and full_thread[0].startswith("Post with ID"):
            label = tk.Label(self.text_frame, text=full_thread[0], bg='gray', fg='white', wraplength=1000, justify='left', anchor='w')
            label.grid(row=len(self.df.columns) + 1, column=0, columnspan=2, sticky="w", pady=5)
            full_thread = full_thread[1:]

        for i, comment in enumerate(full_thread):
            label = tk.Label(self.text_frame, text=comment, bg='gray', fg='white', wraplength=1000, justify='left', anchor='w')
            label.grid(row=len(self.df.columns) + i + 2, column=0, columnspan=2, sticky="w", pady=5)

        self.counter_label.config(text=f"Data Point: {self.current_index + 1}/{len(self.df)}")
        self.counter_label.grid(row=len(self.df.columns) + len(full_thread) + 2, column=0, columnspan=3, pady=10)

        self.reset_label_buttons()

    def reset_label_buttons(self):
        for column in self.new_columns:
            value = self.df.at[self.current_index, column]
            self.selected_labels[column] = value
            relevant_button, irrelevant_button = self.label_buttons[column]
            if value == 1:
                relevant_button.config(relief=tk.SUNKEN, bg='light green')
                irrelevant_button.config(relief=tk.RAISED, bg='light gray')
            elif value == 0:
                relevant_button.config(relief=tk.RAISED, bg='light gray')
                irrelevant_button.config(relief=tk.SUNKEN, bg='light green')
            else:
                relevant_button.config(relief=tk.RAISED, bg='light gray')
                irrelevant_button.config(relief=tk.RAISED, bg='light gray')

    def set_label(self, column, value):
        self.selected_labels[column] = value
        relevant_button, irrelevant_button = self.label_buttons[column]
        if value == 1:
            relevant_button.config(relief=tk.SUNKEN, bg='light green')
            irrelevant_button.config(relief=tk.RAISED, bg='light gray')
        else:
            relevant_button.config(relief=tk.RAISED, bg='light gray')
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
