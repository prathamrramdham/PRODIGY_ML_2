import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CustomerSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation App")

        # Set the background color
        self.root.configure(bg="#e0e0e0")

        # Creating and configuring the main frame
        self.main_frame = ttk.Frame(root, padding="20", style="Main.TFrame")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Adding style for the main frame
        ttk.Style().configure("Main.TFrame", background="#e0e0e0")

        # Adding widgets
        self.create_widgets()

        # Load the dataset
        self.dataset = pd.read_csv("E:/Prodigy_ML/Mall_Customers.csv")

    def create_widgets(self):
        # Adding labels, entry fields, and buttons
        ttk.Label(self.main_frame, text="Enter Customer Information", font=("Helvetica", 16), style="Title.TLabel").grid(row=0, column=0, columnspan=3, pady=10)

        # Adding style for the title label
        ttk.Style().configure("Title.TLabel", foreground="blue", background="#e0e0e0")

        ttk.Label(self.main_frame, text="Gender:", style="SubTitle.TLabel").grid(row=1, column=0, sticky=tk.E, pady=5)
        ttk.Style().configure("SubTitle.TLabel", font=("Helvetica", 12), background="#e0e0e0")

        # Using radio buttons to limit gender choices
        self.gender_var = tk.StringVar()
        male_radio = ttk.Radiobutton(self.main_frame, text="Male", variable=self.gender_var, value="Male")
        female_radio = ttk.Radiobutton(self.main_frame, text="Female", variable=self.gender_var, value="Female")

        male_radio.grid(row=1, column=1, pady=5)
        female_radio.grid(row=1, column=2, pady=5)

        ttk.Label(self.main_frame, text="Age:", style="SubTitle.TLabel").grid(row=2, column=0, sticky=tk.E, pady=5)
        self.age_entry = ttk.Entry(self.main_frame)
        self.age_entry.grid(row=2, column=1, pady=5)

        ttk.Label(self.main_frame, text="Annual Income (k$):", style="SubTitle.TLabel").grid(row=3, column=0, sticky=tk.E, pady=5)
        self.income_entry = ttk.Entry(self.main_frame)
        self.income_entry.grid(row=3, column=1, pady=5)

        ttk.Label(self.main_frame, text="Spending Score (1-100):", style="SubTitle.TLabel").grid(row=4, column=0, sticky=tk.E, pady=5)
        self.score_entry = ttk.Entry(self.main_frame)
        self.score_entry.grid(row=4, column=1, pady=5)

        segmentation_button = ttk.Button(self.main_frame, text="Perform Segmentation", command=self.perform_segmentation)
        segmentation_button.grid(row=5, column=0, columnspan=3, pady=10)

        # Adding a matplotlib figure for displaying the clustering results
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Adding a label for displaying cluster names
        self.cluster_label = ttk.Label(self.root, text="", font=("Helvetica", 10), wraplength=150)
        self.cluster_label.grid(row=0, column=2, padx=10, pady=10, sticky=tk.N)

    def perform_segmentation(self):
        # Extract entered values
        gender = self.gender_var.get()
        age = int(self.age_entry.get())
        income = int(self.income_entry.get())
        score = int(self.score_entry.get())

        # Prepare data for clustering
        new_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [score]})
        data_for_clustering = pd.concat([self.dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], new_data], ignore_index=True)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        data_for_clustering['Cluster'] = kmeans.fit_predict(data_for_clustering[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

        # Display clustered results
        self.display_clusters(data_for_clustering)

    def display_clusters(self, data):
        # Clear previous clusters
        self.ax.clear()

        # Plot the clustered data
        unique_clusters = data['Cluster'].unique()
        cluster_names = {cluster: f'Cluster {cluster}' for cluster in unique_clusters}

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for cluster in unique_clusters:
            cluster_data = data[data['Cluster'] == cluster]
            self.ax.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                            color=colors[cluster], label=cluster_names[cluster])

        self.ax.set_xlabel('Annual Income (k$)')
        self.ax.set_ylabel('Spending Score (1-100)')
        self.ax.legend()

        # Draw the plot on the canvas
        self.canvas.draw()

        # Update the cluster label with the cluster names
        cluster_label_text = "\n".join([f"{cluster_names[cluster]}" for cluster in unique_clusters])
        self.cluster_label.config(text=cluster_label_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerSegmentationApp(root)
    root.mainloop()





