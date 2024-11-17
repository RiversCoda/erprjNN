import sys
import os
import glob
import numpy as np
from scipy.io import loadmat

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QGridLayout, QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DataViewerApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize state variables
        # self.base_dir = 'collect_data\\device3\\p4-test'
        self.base_dir = 'addNoise_data\\test\\raw'
        self.folders = self.get_scg_folders()
        self.current_folder_index = 0
        self.current_folder_name = ''
        self.files_in_current_folder = []
        self.next_file_index = 0
        self.total_files_in_current_folder = 0
        self.deleted_files_in_current_folder = 0
        self.A = 0  # Number of files read in the current folder
        self.B = 0  # Remaining files in current folder
        self.deleted_files = set()

        # Initialize UI components
        self.initUI()

        # Setup current folder
        self.setup_current_folder()

        # Load the first set of data
        self.load_next_data()
        
    def get_scg_folders(self):
        # Get list of scg folders under base_dir
        pattern = os.path.join(self.base_dir, '*', 'scg')
        scg_folders = glob.glob(pattern)
        return scg_folders

    def initUI(self):
        # Set up the UI components
        self.setWindowTitle('Data Viewer')

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Text at the top
        self.info_label = QLabel('')
        self.main_layout.addWidget(self.info_label)

        # Grid layout for plots and delete buttons
        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)

        # List to hold plot widgets and delete buttons
        self.plot_canvases = []
        self.delete_buttons = []
        self.file_paths = []

        # Create placeholders for 10 plots and buttons
        for i in range(10):
            figure = Figure(figsize=(12, 3))
            canvas = FigureCanvas(figure)
            delete_button = QPushButton('Delete')
            delete_button.clicked.connect(self.make_delete_function(i))
            self.grid_layout.addWidget(canvas, i // 2, (i % 2) * 2)
            self.grid_layout.addWidget(delete_button, i // 2, (i % 2) * 2 + 1)
            self.plot_canvases.append(canvas)
            self.delete_buttons.append(delete_button)
            self.file_paths.append(None)  # Placeholder

        # Next data button
        self.next_button = QPushButton('nextData')
        self.next_button.clicked.connect(self.load_next_data)
        self.main_layout.addWidget(self.next_button)

        # Adjust window size
        self.resize(1920, 1080)

    def make_delete_function(self, index):
        def delete_function():
            self.delete_file(index)
        return delete_function

    def update_info_label(self):
        self.info_label.setText(f'Current folder: {self.current_folder_name}, Progress: {self.A}/{self.B}')

    def load_next_data(self):
        # Load up to 10 next files from the current folder
        num_files_loaded = 0
        files_to_load = []

        while num_files_loaded < 10:
            if self.next_file_index < len(self.files_in_current_folder):
                file_path = self.files_in_current_folder[self.next_file_index]
                self.next_file_index += 1
                if file_path not in self.deleted_files:
                    files_to_load.append(file_path)
                    num_files_loaded += 1
            else:
                # No more files in the current folder
                break

        # Update plots
        for i in range(10):
            if i < len(files_to_load):
                self.load_and_plot_data(i, files_to_load[i])
                self.delete_buttons[i].setEnabled(True)
            else:
                # Clear the plot and disable delete button
                self.clear_plot(i)
                self.delete_buttons[i].setEnabled(False)

        # Update A (number of files read in the current folder)
        num_files_loaded = len(files_to_load)
        self.A += num_files_loaded
        self.update_info_label()

        # Check if the current folder is exhausted
        if self.next_file_index >= len(self.files_in_current_folder):
            self.current_folder_exhausted = True
        else:
            self.current_folder_exhausted = False

    def setup_current_folder(self):
        # Setup the counts for the new folder
        current_folder_path = self.folders[self.current_folder_index]
        self.current_folder_name = os.path.basename(os.path.dirname(current_folder_path))
        self.files_in_current_folder = glob.glob(os.path.join(current_folder_path, '*.mat'))
        self.files_in_current_folder.sort()
        self.next_file_index = 0
        self.total_files_in_current_folder = len(self.files_in_current_folder)
        self.deleted_files_in_current_folder = 0
        self.deleted_files = set()
        # Reset A and B for the new folder
        self.A = 0  # Reset number of files read to 0
        self.B = self.total_files_in_current_folder  # Remaining files in current folder
        self.update_info_label()
        self.current_folder_exhausted = False  # Flag to indicate if the current folder is exhausted

    def load_and_plot_data(self, index, file_path):
        # Load data from .mat file and plot it
        try:
            mat_data = loadmat(file_path)
            if 'accresult' not in mat_data:
                if 'scg_data' in mat_data:
                    accresult = mat_data['scg_data']
            else:
                accresult = mat_data['accresult']
            data = 0-accresult[1, :]  # Index 1 corresponds to the second row
            # Plot the data
            canvas = self.plot_canvases[index]
            figure = canvas.figure
            figure.clear()
            ax = figure.add_subplot(111)
            ax.plot(data)
            ax.set_title(os.path.basename(file_path))
            canvas.draw()
            # Store the file path
            self.file_paths[index] = file_path
        except Exception as e:
            print(f'Error loading {file_path}: {e}')

    def delete_file(self, index):
        # Delete the .mat file and update B
        file_path = self.file_paths[index]
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.deleted_files.add(file_path)
                self.deleted_files_in_current_folder += 1
                # Update B (number of remaining files in current folder)
                self.B = self.total_files_in_current_folder - self.deleted_files_in_current_folder
                self.update_info_label()
                # Clear the plot and disable delete button
                self.clear_plot(index)
                self.delete_buttons[index].setEnabled(False)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

    def clear_plot(self, index):
        # Clear the plot
        canvas = self.plot_canvases[index]
        figure = canvas.figure
        figure.clear()
        canvas.draw()
        # Clear file path
        self.file_paths[index] = None

    def nextData(self):
        # This function is called when nextData button is clicked
        if self.current_folder_exhausted:
            # If the current folder is exhausted, move to the next folder
            if self.current_folder_index + 1 < len(self.folders):
                self.current_folder_index += 1
                self.setup_current_folder()
                self.load_next_data()
            else:
                # No more folders
                print("All folders have been processed.")
                # Optionally, disable the nextData button
                self.next_button.setEnabled(False)
        else:
            # Load the next set of data from the current folder
            self.load_next_data()

    def connect_next_button(self):
        # Connect the nextData button to the correct function
        self.next_button.clicked.disconnect()
        self.next_button.clicked.connect(self.nextData)

    def start(self):
        # Start the application
        self.connect_next_button()
        self.load_next_data()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataViewerApp()
    ex.start()
    ex.show()
    sys.exit(app.exec_())
