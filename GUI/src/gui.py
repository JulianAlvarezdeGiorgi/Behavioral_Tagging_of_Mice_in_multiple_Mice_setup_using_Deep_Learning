import tkinter as tk
from tkinter import messagebox
from dataloader import DLCDataLoader  # Replace 'your_module' with the actual module name

class DLCDataLoaderGUI:
    def __init__(self, master):
        self.master = master
        master.title("DLC DataLoader")

        # Labels and Entry widgets for each parameter
        self.root_label = tk.Label(master, text="Root Directory:")
        self.root_label.grid(row=0, column=0)
        self.root_entry = tk.Entry(master)
        self.root_entry.grid(row=0, column=1)

        self.load_dataset_label = tk.Label(master, text="Load Dataset (True/False):")
        self.load_dataset_label.grid(row=1, column=0)
        self.load_dataset_entry = tk.Entry(master)
        self.load_dataset_entry.grid(row=1, column=1)

        self.batch_size_label = tk.Label(master, text="Batch Size:")
        self.batch_size_label.grid(row=2, column=0)
        self.batch_size_entry = tk.Entry(master)
        self.batch_size_entry.grid(row=2, column=1)

        self.num_workers_label = tk.Label(master, text="Number of Workers:")
        self.num_workers_label.grid(row=3, column=0)
        self.num_workers_entry = tk.Entry(master)
        self.num_workers_entry.grid(row=3, column=1)

        self.device_label = tk.Label(master, text="Device (cpu/gpu):")
        self.device_label.grid(row=4, column=0)
        self.device_entry = tk.Entry(master)
        self.device_entry.grid(row=4, column=1)

        self.window_size_label = tk.Label(master, text="Window Size:")
        self.window_size_label.grid(row=5, column=0)
        self.window_size_entry = tk.Entry(master)
        self.window_size_entry.grid(row=5, column=1)

        self.stride_label = tk.Label(master, text="Stride:")
        self.stride_label.grid(row=6, column=0)
        self.stride_entry = tk.Entry(master)
        self.stride_entry.grid(row=6, column=1)

        self.build_graph_label = tk.Label(master, text="Build Graph (True/False):")
        self.build_graph_label.grid(row=7, column=0)
        self.build_graph_entry = tk.Entry(master)
        self.build_graph_entry.grid(row=7, column=1)

        self.behaviour_label = tk.Label(master, text="Behaviour:")
        self.behaviour_label.grid(row=8, column=0)
        self.behaviour_entry = tk.Entry(master)
        self.behaviour_entry.grid(row=8, column=1)

        self.submit_button = tk.Button(master, text="Submit", command=self.create_loader)
        self.submit_button.grid(row=9, column=0, columnspan=2)

    def create_loader(self):
        # Gather the inputs
        root = self.root_entry.get()
        load_dataset = self.load_dataset_entry.get() == 'True'
        batch_size = int(self.batch_size_entry.get())
        num_workers = int(self.num_workers_entry.get())
        device = self.device_entry.get()
        window_size = int(self.window_size_entry.get()) if self.window_size_entry.get() else None
        stride = int(self.stride_entry.get()) if self.stride_entry.get() else None
        build_graph = self.build_graph_entry.get() == 'True'
        behaviour = self.behaviour_entry.get() or None

        # Create the DLCDataLoader instance
        try:
            data_loader = DLCDataLoader(
                root=root,
                load_dataset=load_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                window_size=window_size,
                stride=stride,
                build_graph=build_graph,
                behaviour=behaviour
            )
            messagebox.showinfo("Success", "DLCDataLoader created successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = DLCDataLoaderGUI(root)
    root.mainloop()
