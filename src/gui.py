import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataloader import DLCDataLoader  # Replace with your actual import
from tqdm import tqdm
import threading
import time
import sys
import analyze
from mice_annotation_gui.move_annotations_gui import MovieViewer
import pandas as pd



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
        master.title("Behaviour Analysis App")

        # Create a container for the frames
        self.container = tk.Frame(master)
        self.container.pack(fill="both", expand=True)

        # Create a dictionary to hold references to the frames
        self.frames = {}

        # Initialize frames
        for F in (DataLoaderFrame, InferenceFrame):#, TrainingFrame):
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
        tk.Label(self, text="Dataset Loader/Creation", font=("Helvetica", 16)).pack(pady=10)

        # Root directory input
        self.root_label = tk.Label(self, text="Root Directory:")
        self.root_label.pack()
        self.root_entry = tk.Entry(self)
        self.root_entry.pack()

        self.browse_button = tk.Button(self, text="Browse", command=self.browse_directory)
        self.browse_button.pack()
         # Save dataset path input
        self.save_path_label = tk.Label(self, text="Save Dataset As:")
        self.save_path_label.pack()
        self.save_path_entry = tk.Entry(self)
        self.save_path_entry.pack()

        self.save_browse_button = tk.Button(self, text="Save As", command=self.browse_save_as)
        self.save_browse_button.pack()

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

        self.submit_button = tk.Button(self, text="Get Dataset", command=self.create_loader_thread)
        self.submit_button.pack(pady=10)

        self.train_button = tk.Button(self, text="Go to Inference", command=lambda: controller.show_frame(InferenceFrame))
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
    def browse_save_as(self):
        """Open a file dialog to select where to save the dataset."""
        save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if save_path:
            self.save_path_entry.delete(0, tk.END)  # Clear the entry
            self.save_path_entry.insert(0, save_path)  # Insert the selected file path

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
                threading.Thread(target=dlc_loader.save_dataset(self.save_path_entry.get())).start()

            else:
                messagebox.showinfo("Info", "No dataset loaded, and no graph created.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_loader_thread(self):
        """Run the create_loader method in a separate thread to prevent freezing."""
        threading.Thread(target=self.create_loader).start()

class InferenceFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Create widgets for InferenceFrame
        tk.Label(self, text="Do Inference", font=("Helvetica", 16)).pack(pady=10)

        # Behavior selection dropdown
        self.behaviour_label = tk.Label(self, text="Select Behaviour:")
        self.behaviour_label.pack()
        self.behaviour_var = tk.StringVar()
        self.behaviour_dropdown = ttk.Combobox(self, textvariable=self.behaviour_var)
        self.behaviour_dropdown['values'] = list(analyze.MODELS.keys()) + ['All']
        self.behaviour_dropdown.pack()

        # Dataset path input
        self.dataset_label = tk.Label(self, text="Select Dataset:")
        self.dataset_label.pack()
        self.dataset_entry = tk.Entry(self)
        self.dataset_entry.pack()
        self.dataset_button = tk.Button(self, text="Browse", command=self.browse_dataset)
        self.dataset_button.pack()

        # Save path input
        self.save_label = tk.Label(self, text="Save Output As:")
        self.save_label.pack()
        self.save_entry = tk.Entry(self)
        self.save_entry.pack()
        self.save_button = tk.Button(self, text="Save As", command=self.browse_save_as)
        self.save_button.pack()

        # Progress bar for inference
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=300, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.train_button = tk.Button(self, text="Run Inference", command=self.run_inference_thread)
        self.train_button.pack(pady=10)

        # Button to launch analysis (disabled initially)
        self.analysis_button = tk.Button(self, text="Analyze Results", command=self.launch_analysis_gui)
        self.analysis_button.pack(pady=10)

        self.analysis_button = tk.Button(self, text="Get Statistics", command=self.get_statistics)
        self.analysis_button.pack(pady=10)

        self.back_button = tk.Button(self, text="Back to DataLoader", command=lambda: controller.show_frame(DataLoaderFrame))
        self.back_button.pack()

        # Log display for progress and messages
        self.log_text = tk.Text(self, height=10, width=60)
        self.log_text.pack(pady=10)

        # Redirect print statements to log text widget
        sys.stdout = TextRedirector(self.log_text)

    def browse_dataset(self):
        """Open a file dialog to select the dataset file."""
        dataset_path = filedialog.askopenfilename(filetypes=[("Torch files", "*.pkl")])
        if dataset_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, dataset_path)

    def browse_save_as(self):
        """Open a file dialog to select where to save the output."""
        # Must be a directory, of where to save the files
        save_path = filedialog.askdirectory()
        if save_path:
            self.save_entry.delete(0, tk.END)
            self.save_entry.insert(0, save_path)

    def run_inference(self):
        """Run inference using the selected behavior and dataset."""
        try:
            behaviour = self.behaviour_var.get()
            path_to_data = self.dataset_entry.get()
            path_to_save = self.save_entry.get()

            if not behaviour or not path_to_data or not path_to_save:
                messagebox.showerror("Error", "Please select all the necessary inputs.")
                return

            print(f"Running inference for {behaviour}...")

            # Run the inference
            if behaviour == 'All':
                analyze.inference_all_behaviors(path_to_data, path_to_save)
            else:
                analyze.inference(behaviour, path_to_data, save=True, path_to_save=path_to_save)

            print(f"Inference completed. Results saved at {path_to_save}.")

        except Exception as e:
            messagebox.showerror("Error", str(e))


    def display_statistics_treeview(self, statistics_df):
        """Display statistics in a table format using Treeview."""
        # Create a new window to display the statistics
        stats_window = tk.Toplevel(self.master)
        stats_window.title("Inference Statistics")

        # Create a Treeview widget
        tree = ttk.Treeview(stats_window)
        tree.pack(expand=True, fill="both")

        # Add columns to the Treeview based on DataFrame columns
        tree["columns"] = list(statistics_df.columns)
        tree["show"] = "headings"  # Hide the first empty column

        # Add headings for each column
        for col in statistics_df.columns:
            tree.heading(col, text=col)

        # Insert data rows into the Treeview
        for index, row in statistics_df.iterrows():
            tree.insert("", "end", values=list(row))

    def get_statistics(self):
        """Get statistics for the inference results."""
        try:
            path_to_files = self.save_entry.get()

            if not path_to_files:
                messagebox.showerror("Error", "Please select the path to the inference results.")
                return

            print("Getting statistics for inference results...")

            # Get the statistics from analyze module (returns a DataFrame)
            statistics_df = analyze.get_statistics(path_to_files)

            # Call the display function to show statistics in Treeview
            self.display_statistics_treeview(statistics_df)

            print(f"Statistics displayed for inference results from {path_to_files}.")

        except Exception as e:
             messagebox.showerror("Error", str(e))
    
    
    def launch_analysis_gui(self):
        """Launch the results analysis GUI."""
        # Create a new window for analysis
        viewer = MovieViewer()
        viewer.show()
    
    def run_inference_thread(self):
        """Run the inference in a separate thread to prevent freezing."""
        threading.Thread(target=self.run_inference).start()



if __name__ == "__main__":
    root = tk.Tk()
    app = DLCApp(root)
    root.mainloop()
