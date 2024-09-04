import cv2
import time
import numpy as np
import apriltag
from pymongo import MongoClient
import tkinter as tk
from tkinter import StringVar, Tk, Label, Button, messagebox
from tkinter.font import Font
from datetime import datetime, timedelta


class DatabaseManager:
    def __init__(self, uri):
        self.client = MongoClient(uri)
        self.db = self.client['MazeCompStor']
        self.users = self.db['users']

    def get_teams_sorted_by_timestamp(self):
        users = self.users.find({}).sort("lastUploadToBotTimestamp", 1)
        teams = []
        for user in users:
            if user['lastUploadToBotTimestamp'] is not None:
                teams.append((user['teamName'], user['lastUploadToBotTimestamp'].isoformat()))
        return teams

    def update_hardware_score(self, team_name, score):
        self.users.update_one({"teamName": team_name}, {"$set": {"hardwareScore": score}})
        print(f"Updated hardwareScore for {team_name} to {score}")


class CameraProcessor:
    def __init__(self):
        self.cap = None
        self.detector = None

    def init_camera_and_detector(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            raise Exception("Failed to capture frame")
            return None
        return frame
    
    def detect_apriltags(self) -> tuple[cv2.typing.MatLike, list[apriltag.Detection]]:
        frame = self.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return frame, detections
    
    def cleanup(self):
        self.cap.release()


 
class MainWindow:
    max_time = 300
    init_time_score = 600
    score_reduction_rate = 12.5
    grace_period = timedelta(minutes=1)
    score_reduction_interval = timedelta(seconds=5)
    max_cells = 64

    def __init__(self, root: Tk, database_manager: DatabaseManager, camera_processor: CameraProcessor):
        self.root = root
        self.db = database_manager
        self.camera = camera_processor

        self.root.title("Team Selection and Timer Control")
        self.root.geometry("600x400")

        self.selected_team_name = None
        self.timer_started = False
        self.current_team_index = 0
        self.first_run = True
        self.teams = []

        self.roi: cv2.typing.Rect = None

        self.init_ui()
        self.init_components()
 

    def cleanup_nontk(self):
        self.camera.cleanup()
     
    def init_ui(self):
        label_font = Font(family="Arial", size=14)
        button_font = Font(family="Arial", size=12)

        self.label = Label(self.root, text="Teams are being processed automatically based on upload timestamp:", font=label_font)
        self.label.pack(pady=10)

        self.roi_button = Button(self.root, text="Select ROI", font=button_font, command=self.select_roi)
        self.roi_button.pack(pady=10)

        self.refresh_button = Button(self.root, text="Refresh Teams", font=button_font, command=self.load_teams)
        self.refresh_button.pack(pady=10)

        self.start_timer_button = Button(self.root, text="Start Timer", font=button_font, command=self.start_team_timer)
        self.start_timer_button.pack(pady=10)

        self.update_start_timer_button_text()

    def increment_selected_team_and_reset(self):
        self.timer_started = False
        self.start_time = None
    
        if self.current_team_index < len(self.teams) - 1:
            self.current_team_index += 1
            self.selected_team_name = self.teams[self.current_team_index][0]
   
            self.update_start_timer_button_text()
        else:
            self.start_timer_button.config(text=f"All teams have been processed!", state=tk.DISABLED)
            self.selected_team_name = None
            self.current_team_index = len(self.teams)
            messagebox.showinfo("End of Queue", "All teams have been processed.")
     
    
    def update_start_timer_button_text(self):
        """Update the text of the start timer button to reflect the selected team name."""

        if self.selected_team_name:
            self.start_timer_button.config(text=f"Start Timer for {self.selected_team_name}", state=tk.NORMAL)
      
            

    def init_components(self):
        self.camera.init_camera_and_detector()
        self.load_teams()

    def select_roi(self):
        frame = self.camera.get_frame()
        self.roi = cv2.selectROI("Select ROI", frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")
     

    def load_teams(self):
        new_teams = self.db.get_teams_sorted_by_timestamp()
        if new_teams != self.teams:
            self.teams = new_teams
            print("Loaded Teams:")
            for team, timestamp in self.teams:
                print(f"{team} - {timestamp}")

            self.current_team_index = -1
            self.increment_selected_team_and_reset()

        elif self.current_team_index == len(self.teams):
            self.current_team_index = -1
            self.increment_selected_team_and_reset()

    def start_team_timer(self):
        if self.timer_started:
            messagebox.showinfo("Timer", "Timer already started!")
            return
        

        if self.roi is None:
            messagebox.showinfo("ROI", "Please select an ROI first!")
            return

        self.timer_started = True
        self.start_time = time.time()
        messagebox.showinfo("Timer", "Timer started!")
        self.process_detections(self.selected_team_name, self.roi)
        self.timer_started = False


    def display_score_and_proceed(self, exploration_score, final_time_score):
        team_name, _ = self.teams[self.current_team_index]
        final_score = exploration_score + final_time_score
        messagebox.showinfo(f"Scores for {team_name}", f"Exploration Score: {exploration_score}\nTime Score: {final_time_score}\nFinal Score: {final_score}")
        self.db.update_hardware_score(self.selected_team_name, final_score)

        if self.current_team_index < len(self.teams):
            self.increment_selected_team_and_reset()
        else:
            messagebox.showinfo("End of Queue", "All teams have been processed.")


    def halt_until_detected_moves(self):
        """
        Not the most complete code, as if the detection is spotty this may have false positives, but good enough for now.
        """

        def dist(a: apriltag.Detection, b: apriltag.Detection):
            return a.center[0] - b.center[0] ** 2  + a.center[1] - b.center[1] ** 2

        starting_detections: list[apriltag.Detection] = []
        last_detections: list[apriltag.Detection] = []
        while self.timer_started:
            frame, detections = self.camera.detect_apriltags()
            cv2.imshow('AprilTag Movement Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('AprilTag Movement Detection')
                messagebox.showinfo("Detection skipped", "Detection skipped.")
                return

            if not detections:
                continue
            
            if not last_detections:
                starting_detections = detections # assign the starting detections if it's the first frame
                last_detections = detections
                continue
            elif len(detections) > len(last_detections): # assign starting detections if we suddenly detect a new bot and perform check again.
                starting_detections = detections
                continue

            if len(last_detections) != len(detections):
                last_detections = detections
                continue # don't bother checking on an unreliable frame.

            for detection in detections:
                # find the starting detection that is closest
                start_detection = min(starting_detections, key=lambda x: dist(x, detection))
        
                center = detection.center
                last_center = start_detection.center
    
                # find the difference in x and y coordinates
                dx = center[0] - last_center[0]
                dy = center[1] - last_center[1]

                # find size of starting apriltag
                w0 = abs(start_detection.corners[1][0] - start_detection.corners[0][0])
                h0 = abs(start_detection.corners[2][1] - start_detection.corners[1][1])

                if abs(dx) > w0 // 2 or abs(dy) > h0 // 2:
                    cv2.destroyWindow('AprilTag Movement Detection')

                    return
                
            last_detections = detections

        cv2.destroyWindow('AprilTag Movement Detection')
        raise Exception("Timer stopped unexpectedly")

            


    def process_detections(self, team_name: str, roi=None, total_time_seconds=None):
        if roi is None:
            roi = self.roi

        if total_time_seconds is None:
            total_time_seconds = self.max_time
       
        cell_size_x = roi[2] // 8
        cell_size_y = roi[3] // 8
        visited_cells = set()

        self.halt_until_detected_moves()

        start_datetime = datetime.now()
        exploration_score = 0
        final_time_score = MainWindow.init_time_score

        while self.timer_started:
 
            current_time = datetime.now()
            elapsed_time = (current_time - start_datetime).total_seconds()
            time_left = max(0, total_time_seconds - elapsed_time)

            if time_left <= 0:
                cv2.destroyWindow('AprilTag Detection')
                messagebox.showinfo("Time's Up", f"Time's up for team {team_name}!")
                self.display_score_and_proceed(exploration_score, final_time_score)
                return

            frame, detections = self.camera.detect_apriltags()

            self.update_frame(frame, roi, cell_size_x, cell_size_y, visited_cells, detections)
            exploration_score = min(len(visited_cells) * 2, self.max_cells * 2)

            if elapsed_time > 60:
                if (current_time - last_score_reduction_time >= MainWindow.score_reduction_interval and final_time_score > 0):
                    final_time_score = max(final_time_score - MainWindow.score_reduction_rate, 0)
                    last_score_reduction_time = current_time

            cv2.putText(frame, f"Time left: {int(time_left)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, f"Exploration Score: {exploration_score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Score: {final_time_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('AprilTag Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('AprilTag Detection')
        raise Exception("Timer stopped unexpectedly")


    def update_frame(self, frame, roi, cell_size_x, cell_size_y, visited_cells, detections):
        for i in range(9):
            cv2.line(frame, (roi[0] + i * cell_size_x, roi[1]), (roi[0] + i * cell_size_x, roi[1] + roi[3]), (68, 222, 253), 2)
            cv2.line(frame, (roi[0], roi[1] + i * cell_size_y), (roi[0] + roi[2], roi[1] + i * cell_size_y), (68, 222, 253), 2)

        for cell_x, cell_y in visited_cells:
            top_left_x = roi[0] + cell_x * cell_size_x
            top_left_y = roi[1] + cell_y * cell_size_y
            bottom_right_x = top_left_x + cell_size_x
            bottom_right_y = top_left_y + cell_size_y
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), -1)

        for detection in detections:
            tag_id = detection.tag_id
            center = detection.center
            cell_x = int((center[0] - roi[0]) / cell_size_x)
            cell_y = int((center[1] - roi[1]) / cell_size_y)
            if 0 <= cell_x < 8 and 0 <= cell_y < 8:
                visited_cell = (cell_x, cell_y)
                if visited_cell not in visited_cells:
                    visited_cells.add(visited_cell)
                    print(f"Visited cell: {visited_cell}")

            pt1 = (int(center[0]) - 10, int(center[1]) - 10)
            pt2 = (int(center[0]) + 10, int(center[1]) + 10)
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)



def main():
    root = Tk()
    mongo_uri = "mongodb+srv://mazecomphero:AgdHtQsZAmsW2wQu@cluster0.oqpibir.mongodb.net/NewMazeCompetition?retryWrites=true&w=majority"
    db_manager = DatabaseManager(mongo_uri)
    camera_processor = CameraProcessor()
    main_window = MainWindow(root, db_manager, camera_processor)
    root.mainloop()
    main_window.cleanup_nontk()


if __name__ == "__main__":
    main()
