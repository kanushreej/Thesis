import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import json
import os
import numpy as np

# Set paths and directories
base_directory = "/Users/kanushreejaiswal/Desktop/Thesis"
moderator_name = "Kanu"
issue = "IsraelPalestineUK"
original_data_path = f"{base_directory}/Subreddit Data/UK/{issue}_data.csv"
base_labeling_data_path = f"{base_directory}/Annotation/UK/Labelling data/{issue}_sample.csv"
moderator_labeling_data_path = f"{base_directory}/Annotation/UK/{moderator_name}/{issue}_labelled.csv"
progress_file = f"{base_directory}/Annotation/UK/{moderator_name}/Progress/{issue}.json"

os.makedirs(os.path.dirname(moderator_labeling_data_path), exist_ok=True)
os.makedirs(os.path.dirname(progress_file), exist_ok=True)

# Load Data
def load_data(filepath, basepath):
    if not os.path.exists(filepath):
        df = pd.read_csv(basepath)
        df.to_csv(filepath, index=False)
    df = pd.read_csv(filepath, dtype={'parent_id': str, 'body': str, 'title': str, 'id': str})
    df['parent_id'] = df['parent_id'].fillna('')
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')
    return df

# Load and Save Progress
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_progress(last_index):
    with open(progress_file, 'w') as f:
        json.dump({"last_index": last_index}, f)

# GUI Application
class LabelingApp:
    def __init__(self, master):
        self.master = master
        master.title("Reddit Comment Labeling")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', background='#0078D7', foreground='white', padding=5)
        style.map('TButton', background=[('active', '#0063B1'), ('disabled', '#f2f2f2')])

        self.label_data = load_data(moderator_labeling_data_path, base_labeling_data_path)
        self.original_data = load_data(original_data_path, None)
        self.current_index = load_progress()

        self.setup_gui()
        self.display_data()

    def setup_gui(self):
        self.index_label = ttk.Label(self.master, text=f"Index: {self.current_index + 1}/{len(self.label_data)}", font=('Helvetica', 12))
        self.index_label.pack(fill=tk.X, padx=10, pady=5)

        self.nav_frame = ttk.Frame(self.master)
        self.nav_frame.pack(fill=tk.X, expand=True, pady=10)

        self.prev_button = ttk.Button(self.nav_frame, text="Previous", command=self.previous_data)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.next_data)
        self.next_button.pack(side=tk.RIGHT)

        self.save_button = ttk.Button(self.nav_frame, text="Save", command=self.save_data)
        self.save_button.pack(side=tk.RIGHT, padx=10)

        self.setup_text_and_checkbuttons()

    def setup_text_and_checkbuttons(self):
        self.text_frame = tk.Frame(self.master)
        self.text_frame.pack(fill=tk.BOTH, expand=True)
        self.text_display = tk.Text(self.text_frame, height=10, width=80)
        self.text_scroll = tk.Scrollbar(self.text_frame, command=self.text_display.yview)
        self.text_display.configure(yscrollcommand=self.text_scroll.set)
        self.text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.checkbuttons_frame = ttk.Frame(self.master)
        self.checkbuttons_frame.pack(fill=tk.X, expand=True, pady=10)
        self.check_vars = {}
        labels = [
            'pro_brexit', 'anti_brexit', 'pro_climateAction', 'anti_climateAction',
            'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine',
            'increase_tax', 'decrease_tax', 'neutral', 'irrelevant'
        ]
        for label in labels:
            var = tk.IntVar(value=0)
            chk = ttk.Checkbutton(self.checkbuttons_frame, text=label, variable=var)
            chk.pack(anchor=tk.W)
            self.check_vars[label] = var

    def display_data(self):
        self.index_label.configure(text=f"Index: {self.current_index + 1}/{len(self.label_data)}")
        if not len(self.label_data):
            messagebox.showerror("Error", "No data to display.")
            self.master.quit()
            return

        current_data = self.label_data.iloc[self.current_index]
        title_text = f"Title: {current_data['title'] if current_data['title'] else 'Empty'}"
        body_text = f"Body: {current_data['body'] if current_data['body'] else 'Empty'}"
        thread = self.build_thread(current_data['id'])

        self.text_display.delete(1.0, tk.END)
        if thread.strip():
            self.text_display.insert(tk.END, thread + "\n")
            self.text_display.insert(tk.END, "----------------------------------------\n")
        self.text_display.insert(tk.END, title_text + "\n" + body_text + "\n")

        if thread.strip():
            self.text_display.tag_add("highlight", f"{int(self.text_display.index('end').split('.')[0]) - 2}.0", "end")
            self.text_display.tag_config("highlight", background="red", foreground="white")

        # Update checkbutton states
        self.update_checkbuttons()

    def build_thread(self, comment_id):
        # Build the thread by collecting parent comments first
        thread = self.build_parent_thread(comment_id)
        # Add the current comment
        current_comment = self.original_data[self.original_data['id'] == comment_id].iloc[0]
        thread += f"ID: {current_comment['id']}\nBody: {current_comment['body']}\n\n"
        return thread

    def build_parent_thread(self, comment_id):
        thread = ""
        parent_id = self.original_data.loc[self.original_data['id'] == comment_id, 'parent_id'].values[0]
        while parent_id:
            if parent_id.startswith('t1_'):
                parent_comment = self.original_data[self.original_data['id'] == parent_id[3:]].iloc[0]
                thread = f"ID: {parent_comment['id']}\nBody: {parent_comment['body']}\n\n" + thread
                parent_id = parent_comment['parent_id']
            elif parent_id.startswith('t3_'):
                parent_post = self.original_data[self.original_data['id'] == parent_id[3:]].iloc[0]
                thread = f"ORIGINAL POST\nID: {parent_post['id']}\nTitle: {parent_post['title']}\nBody: {parent_post['body']}\n\n" + thread
                break
        return thread

    def next_data(self):
        if self.current_index < len(self.label_data) - 1:
            self.save_data()  # Save the current options
            self.reset_checkbuttons()  # Reset checkbuttons before moving to the next data
            self.current_index += 1
            save_progress(self.current_index)
            self.display_data()

    def previous_data(self):
        if self.current_index > 0:
            self.reset_checkbuttons()  # Reset checkbuttons before moving to the previous data
            self.current_index -= 1
            save_progress(self.current_index)
            self.display_data()

    def update_checkbuttons(self):
        # Update the checkbuttons based on the current data
        current_data = self.label_data.iloc[self.current_index]
        for label, var in self.check_vars.items():
            value = current_data.get(label, 0)
            if pd.isna(value):
                value = 0
            var.set(int(value))

    def reset_checkbuttons(self):
        for var in self.check_vars.values():
            var.set(0)

    def on_closing(self):
        self.save_data()
        save_progress(self.current_index)  # Save position without prompting
        self.master.destroy()

    def save_data(self):
        for label, var in self.check_vars.items():
            self.label_data.at[self.current_index, label] = var.get()
        self.label_data.to_csv(moderator_labeling_data_path, index=False)
        messagebox.showinfo("Save", "Data has been saved successfully.")

# Main function
def main():
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
