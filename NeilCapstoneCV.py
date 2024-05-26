import cv2
import time
import numpy as np
import sys
from apriltag import apriltag
from pymongo import MongoClient
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QIcon 
from datetime import datetime, timedelta

selected_team_name = None  # the current team selection to track
timer_started = False
start_time = None
# Global variables for scoring
initial_time_score = 600  # Initial Time Score (ITS)
score_reduction_rate = 12.5  # Score Reduction Rate (SRR)
grace_period = timedelta(minutes=1)  # Grace period as a timedelta object
score_reduction_interval = timedelta(seconds=5)  # Interval for score reduction
max_cells = 64  # Maximum number of cells in an 8x8 maze
first_run = True

def init_camera_and_detector():
    cap = cv2.VideoCapture(0)
    detector = apriltag("tagStandard52h13") #- GOOD DETECTION SEEMES TO BE WORKING REALLY WELl

    return cap, detector

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Team Selection and Timer Control')
        self.setGeometry(300, 300, 500, 400)
        self.setWindowIcon(QIcon('path_to_your_icon'))  # Optional: Set  app icon here
        global final_time_score, exploration_score
        final_time_score = initial_time_score  # Initialize final_time_score
        exploration_score = 0
        self.is_drawing = False  # Flag to indicate if we're currently drawing a box
        self.is_moving = False  # Flag to indicate if we're moving the box
        self.box_start = None  # Starting point (x, y) of the box
        self.box_end = None  # Ending point (x, y) of the box
        self.box_position = None  # Current position of the box, used for moving
        self.current_team_index = 0  # Track the index of the current team
        first_run = True
        self.teams = []
        self.initUI()
        self.setupRefreshTimer()
        self.loadTeams()
        #self.startProcessing()

    def initUI(self):
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout(self.centralWidget)

        self.label = QLabel("Teams are being processed automatically based on upload timestamp:")
        self.label.setFont(QFont('Arial', 14))
        layout.addWidget(self.label)

        self.refreshButton = QPushButton("Refresh Teams")
        self.refreshButton.setFont(QFont('Arial', 12))
        self.refreshButton.clicked.connect(self.loadTeams)
        layout.addWidget(self.refreshButton)

        self.startTimerButton = QPushButton("Start Timer")
        self.startTimerButton.setFont(QFont('Arial', 12))
        self.startTimerButton.clicked.connect(self.manualStartTimer)
        layout.addWidget(self.startTimerButton)

        

        self.applyStyle()
    
    def setupRefreshTimer(self):
        self.refreshTimer = QTimer(self)
        self.refreshTimer.timeout.connect(self.loadTeams)
        self.refreshTimer.start(20000)  # Refresh every 20 seconds
    
    def loadTeams(self):
        new_teams = get_teams_sorted_by_timestamp()
        # Always include the buffer team first
        if new_teams != self.teams: #checking if list of newly retrived teams is the same as the list of old teams 
            self.teams = new_teams  # Prepend buffer team to the loaded team list
            print("Loaded Teams:")
            for team, timestamp in self.teams:
                print(f"{team} - {timestamp}")
            if not timer_started:  # Only start processing if the timer is not currently started
                self.startProcessing()
    
    def startProcessing(self):
        global timer_started, selected_team_name, first_run
        if self.teams and not timer_started:
            if first_run:
                first_run = False  # initial setup phase is complete 
                team_name, _ = self.teams.pop(0) #pops the first team from the list without processing it
                self.bufferStartTimer() #buffer start timer is called
                
                return  # Skip processing on the first run
            team_name, _ = self.teams.pop(0)  #pop the teams to start processing it.
            selected_team_name = team_name
            QMessageBox.information(self, "Team Selected", f"Selected Team: {selected_team_name}")
            self.manualStartTimer() #start time function is then called. 

    @pyqtSlot()
    def selectTeam(self):
        if self.current_team_index < len(self.teams): #ensure that the team index does not attempt to find a team that does not exist 
            team_name, _ = self.teams[self.current_team_index]
            self.current_team_index += 1 # Increment the current team index after the team is selected 
            self.teamListWidget.setCurrentRow(self.current_team_index - 1)
            selected_team_name = team_name
            QMessageBox.information(self, "Team Selected", f"Selected Team: {selected_team_name}")

    @pyqtSlot()
    def bufferStartTimer(self): #buffer function used to handle the team that is popped from the list always at the beginning. 
        global timer_started, start_time
        if not timer_started:
            timer_started = True
            start_time = time.time()
            QMessageBox.information(self, "Timer", "Timer started!")
            cap, detector = init_camera_and_detector()
            self.process_detections_buffer(cap, detector) 
            cap.release()
            cv2.destroyAllWindows()

    @pyqtSlot()
    def manualStartTimer(self):
        global timer_started, start_time
        if not timer_started: #if the time hase not started, then set it to true and start the timer.
            timer_started = True
            start_time = time.time()
            QMessageBox.information(self, "Timer", "Timer started!")
            cap, detector = init_camera_and_detector() #initialize the camera capture and detection
            self.process_detections(cap, detector) #call the function used to process camera data. 
            cap.release()
            cv2.destroyAllWindows()

    def process_detections_buffer(self, cap, detector):
        #buffer function for the team that is always popped first. 
        global timer_started, start_time
        global exploration_score, final_time_score
        total_time_seconds = 10  # Total time for the activity (5 minutes)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return
        roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        cell_size_x = roi[2] // 8
        cell_size_y = roi[3] // 8
        visited_cells = set()
        start_datetime = datetime.now()
        last_score_reduction_time = start_datetime
        while timer_started:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            current_time = datetime.now()
            elapsed_time = (current_time - start_datetime).total_seconds()
            time_left = max(0, total_time_seconds - elapsed_time)
            if time_left <= 0:
                QMessageBox.information(self, "Time's Up", "Time's up for all robots!")
                timer_started = False
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            cv2.putText(frame, f"Time left: {int(time_left)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            cv2.imshow('AprilTag Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def process_detections(self, cap, detector):
        global timer_started, start_time
        global exploration_score, final_time_score
        total_time_seconds = 300  # Total time for the activity (5 minutes)

        ret, first_frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return

        roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False) #open a window for the user to manually select the ROI and then destroy the capture window. 
        cv2.destroyWindow("Select ROI")
        cell_size_x = roi[2] // 8 #calculating the cell size on the maze divided by 8 
        cell_size_y = roi[3] // 8
        visited_cells = set()

        start_datetime = datetime.now()
        last_score_reduction_time = start_datetime

        while timer_started:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            current_time = datetime.now()
            elapsed_time = (current_time - start_datetime).total_seconds()
            time_left = max(0, total_time_seconds - elapsed_time)

            if time_left <= 0:
                QMessageBox.information(self, "Time's Up", "Time's up for all robots!")
                timer_started = False
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale which is needed for detection.
            detections = detector.detect(gray) #detecting apriltag on greyscale
            self.update_frame(frame, roi, cell_size_x, cell_size_y, visited_cells, detections) #updating the frame with detection information
            exploration_score = min(len(visited_cells) * 2, max_cells * 2)

            if elapsed_time > 60:  # Only start reducing score after 60 seconds
                if (current_time - last_score_reduction_time >= score_reduction_interval and
                    final_time_score > 0):
                    final_time_score = max(final_time_score - score_reduction_rate, 0)
                    last_score_reduction_time = current_time

            cv2.putText(frame, f"Time left: {int(time_left)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, f"Exploration Score: {exploration_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Score: {final_time_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('AprilTag Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.displayScoreAndProceed()

    def update_frame(self, frame, roi, cell_size_x, cell_size_y, visited_cells, detections):
        # Draw the grid and visited cells
        for i in range(9):  # Includes lines for all edges
            # Vertical lines
            cv2.line(frame, (roi[0] + i * cell_size_x, roi[1]), (roi[0] + i * cell_size_x, roi[1] + roi[3]), (68, 222, 253), 2)
            # Horizontal lines
            cv2.line(frame, (roi[0], roi[1] + i * cell_size_y), (roi[0] + roi[2], roi[1] + i * cell_size_y), (68, 222, 253), 2)

        # Loop through each cell marked as visited to visually update the frame
        for cell_x, cell_y in visited_cells:
            # Calculate the top left corner of the rectangle for the visited cell
            top_left_x = roi[0] + cell_x * cell_size_x
            top_left_y = roi[1] + cell_y * cell_size_y
            # Calculate the bottom right corner of the rectangle
            bottom_right_x = top_left_x + cell_size_x
            bottom_right_y = top_left_y + cell_size_y
            #fill in the visited cell in green 
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)  # Fill the cell with green

        for detection in detections:
            #extract the id and center of the apriltag
            tag_id = detection['id']
            center = detection['center']

            # Calculate the cell indices by comparing the center of the detection with the ROI and cell size
            cell_x = int((center[0] - roi[0]) / cell_size_x)
            cell_y = int((center[1] - roi[1]) / cell_size_y)
            # Check if the calculated cell coordinates are within the bounds of the 8x8 grid
            if 0 <= cell_x < 8 and 0 <= cell_y < 8:
                visited_cell = (cell_x, cell_y)
                if visited_cell not in visited_cells:
                    visited_cells.add(visited_cell)
                    print(f"Visited cell: {visited_cell}")

            # Define the points for a rectangle around the detected tag's center
            pt1 = (int(center[0]) - 10, int(center[1]) - 10)
            pt2 = (int(center[0]) + 10, int(center[1]) + 10)

            # Draw a red rectangle around the detected tag
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    def displayScoreAndProceed(self):
        global exploration_score, final_time_score, selected_team_name, timer_started, start_time

        final_score = exploration_score + final_time_score  # Assuming you sum these for the final hardware score
        QMessageBox.information(self, "Scores", f"Exploration Score: {exploration_score} - Time Score: {final_time_score} - Final Score: {final_score}")

        # Update the hardware score in the database
        self.update_hardware_score(selected_team_name, final_score)

        if self.current_team_index < len(self.teams):
            # Load next team
            team_name, _ = self.teams[self.current_team_index]
            self.current_team_index += 1
            selected_team_name = team_name
            QMessageBox.information(self, "Next Team", f"Next Team: {selected_team_name}")
            # Reset scores and timer
            timer_started = False
            start_time = None
            exploration_score = 0
            final_time_score = initial_time_score
            self.startTimerButton.setEnabled(True)
        else:
            QMessageBox.information(self, "End of Queue", "All teams have been processed.")
            self.startTimerButton.setEnabled(False)
            
    def update_hardware_score(self, team_name, score):
        uri = "mongodb+srv://mazecomphero:AgdHtQsZAmsW2wQu@cluster0.oqpibir.mongodb.net/MazeCompStor"
        client = MongoClient(uri)
        db = client['MazeCompStor']
        users = db['users']

        # Update the hardware score for the given team name
        users.update_one({"teamName": team_name}, {"$set": {"hardwareScore": score}})

        print(f"Updated hardwareScore for {team_name} to {score}")



    def applyStyle(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2C3E50;
                color: #ECF0F1;
            }
            QLabel, QPushButton {
                font-weight: bold;
                margin: 10px;
            }
            QListWidget {
                border: 1px solid #34495E;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }
            QPushButton {
                background-color: #3498DB;
                border-radius: 5px;
                padding: 10px;
                color: #ECF0F1;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)



def get_teams_sorted_by_timestamp():
    mongo_uri = "mongodb+srv://mazecomphero:AgdHtQsZAmsW2wQu@cluster0.oqpibir.mongodb.net/MazeCompStor?retryWrites=true&w=majority"
    client = MongoClient(mongo_uri)
    db = client['MazeCompStor']
    collection = db['users']
    while True:  # Loop until valid data is found
        users = collection.find({}).sort("lastUploadToBotTimestamp", 1)  # 1 for ascending order
        teams = []
        for user in users:
            if user['lastUploadToBotTimestamp'] is not None:
                teams.append((user['teamName'], user['lastUploadToBotTimestamp'].isoformat()))
        if teams:  # Only break the loop if we have valid data
            return teams
        print("Waiting for new teams to appear...")
        time.sleep(10)  # Wait for 10 seconds before trying again


def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()