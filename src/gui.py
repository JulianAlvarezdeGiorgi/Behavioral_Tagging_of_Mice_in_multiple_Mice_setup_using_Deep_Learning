import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataloader import DLCDataLoader  # Replace with your actual import
from tqdm import tqdm
import threading
import time
import sys

class TextRedirector(object):
    """Class to redirect print statements to Tkinter Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end

    def flush(self):
        pass  # Required for Python 3 compatibility


class DLCApp:
    def __init__(self, master):
        self.master = master
        master.title("DLC Application")

        # Create a container for the frames
        self.container = tk.Frame(master)
        self.container.pack(fill="both", expand=True)

        # Create a dictionary to hold references to the frames
        self.frames = {}

        # Initialize frames
        for F in (DataLoaderFrame, TrainingFrame):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(DataLoaderFrame)

    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        frame.tkraise()

class DataLoaderFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Create widgets for DataLoaderFrame
        tk.Label(self, text="DLC DataLoader", font=("Helvetica", 16)).pack(pady=10)

        # Root directory input
        self.root_label = tk.Label(self, text="Root Directory:")
        self.root_label.pack()
        self.root_entry = tk.Entry(self)
        self.root_entry.pack()

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_directory)
        self.browse_button.pack()

        # Load dataset option (True/False)
        tk.Label(self, text="Load Dataset?", font=("Helvetica", 12)).pack(pady=5)
        self.load_dataset_var = tk.BooleanVar(value=False)

        self.load_yes_button = tk.Radiobutton(self, text="Yes", variable=self.load_dataset_var, value=True)
        self.load_no_button = tk.Radiobutton(self, text="No", variable=self.load_dataset_var, value=False)
        self.load_yes_button.pack()
        self.load_no_button.pack()

        # Build graph option (True/False)
        tk.Label(self, text="Build Graph?", font=("Helvetica", 12)).pack(pady=5)
        self.build_graph_var = tk.BooleanVar(value=False)

        self.graph_yes_button = tk.Radiobutton(self, text="Yes", variable=self.build_graph_var, value=True)
        self.graph_no_button = tk.Radiobutton(self, text="No", variable=self.build_graph_var, value=False)
        self.graph_yes_button.pack()
        self.graph_no_button.pack()

        # Window size and stride (optional)
        self.window_size_label = tk.Label(self, text="Window Size (Optional):")
        self.window_size_label.pack()
        self.window_size_entry = tk.Entry(self)
        self.window_size_entry.pack()

        self.stride_label = tk.Label(self, text="Stride (Optional):")
        self.stride_label.pack()
        self.stride_entry = tk.Entry(self)
        self.stride_entry.pack()

        # Behaviour (optional)
        self.behaviour_label = tk.Label(self, text="Behaviour (Optional):")
        self.behaviour_label.pack()
        self.behaviour_entry = tk.Entry(self)
        self.behaviour_entry.pack()

        # Progress bar for loader creation
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.submit_button = tk.Button(self, text="Create DataLoader", command=self.create_loader_thread)
        self.submit_button.pack(pady=10)

        self.train_button = tk.Button(self, text="Go to Training", command=lambda: controller.show_frame(TrainingFrame))
        self.train_button.pack()

        # Log display for progress and messages
        self.log_text = tk.Text(self, height=10, width=60)
        self.log_text.pack(pady=10)

        # Redirect print statements to log text widget
        sys.stdout = TextRedirector(self.log_text)

    def browse_directory(self):
        """Open a file dialog to select a directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.root_entry.delete(0, tk.END)  # Clear the entry
            self.root_entry.insert(0, directory)  # Insert the selected directory

    def create_loader(self):
        """Create the data loader with progress indication."""
        try:
            root = self.root_entry.get()
            load_dataset = self.load_dataset_var.get()
            build_graph = self.build_graph_var.get()
            window_size = int(self.window_size_entry.get()) if self.window_size_entry.get() else None
            stride = int(self.stride_entry.get()) if self.stride_entry.get() else None
            behaviour = self.behaviour_entry.get() or None

            if load_dataset or build_graph:
                print("Starting DataLoader creation...")

                # Callback function to update the progress bar
                def update_progress_bar(current, total):
                    progress = (current / total) * 100
                    self.progress_bar['value'] = progress  # Update progress bar
                    self.update_idletasks()  # Force update of the GUI

                # Create the DLCDataLoader with progress callback
                dlc_loader = DLCDataLoader(
                    root=root,
                    load_dataset=load_dataset,
                    window_size=window_size,
                    stride=stride,
                    build_graph=build_graph,
                    behaviour=behaviour,
                    progress_callback=update_progress_bar  # Pass the progress callback
                )

                # Run the data loader creation in a separate thread
                threading.Thread(target=dlc_loader.save_dataset).start()

            else:
                messagebox.showinfo("Info", "No dataset loaded, and no graph created.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_loader_thread(self):
        """Run the create_loader method in a separate thread to prevent freezing."""
        threading.Thread(target=self.create_loader).start()

class TrainingFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Create widgets for TrainingFrame
        tk.Label(self, text="Training Model", font=("Helvetica", 16)).pack(pady=10)
        self.train_button = tk.Button(self, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.back_button = tk.Button(self, text="Back to DataLoader", command=lambda: controller.show_frame(DataLoaderFrame))
        self.back_button.pack()

    def train_model(self):
        # Implement your training logic here
        messagebox.showinfo("Training", "Training logic goes here!")


if __name__ == "__main__":
    root = tk.Tk()
    app = DLCApp(root)
    root.mainloop()
