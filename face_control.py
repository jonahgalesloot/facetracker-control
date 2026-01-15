import subprocess
import sys
import atexit
import time
import cv2
import zmq
import json
import numpy as np
from collections import deque
import keyboard
import pyautogui
import math

# --- Automatically start facetracker_module.py as a subprocess ---
facetracker_proc = subprocess.Popen([sys.executable, "facetracker_module.py"])
atexit.register(lambda: facetracker_proc.terminate())

# --- Set up ZeroMQ subscriber ---
context = zmq.Context()
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:5555")
socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

head_tracking_enabled = False
cursor_control_active = False  # For toggling cursor control separately if desired


# Calibration variables: use the first 30 frames to determine neutral values
calibration_frames = 30
yaw_calibration = []
pitch_calibration = []
neutral_yaw = None
neutral_pitch = None

# Sensitivity and thresholds
pitch_sensitivity = 1  # Adjust this value to change cursor movement speed
yaw_sensitivity = 1    # Adjust this value to change cursor movement speed
yaw_threshold = 10
pitch_threshold = 8

blink_frames = 0
BLINK_THRESHOLD = 10  # Number of consecutive blink frames to trigger a click


def recalibrate():
    global frame_count
    frame_count = -5
    print("Recalibration initiated. Please look straight ahead.")


def toggle_head_tracking():
    global head_tracking_enabled
    head_tracking_enabled = not head_tracking_enabled
    print(f"Head tracking {'enabled' if head_tracking_enabled else 'disabled'}")
    
    
def shutdown():
    cv2.destroyAllWindows()
    socket_sub.close()
    context.term()
    facetracker_proc.terminate()
    exit()

# Register a hotkey to toggle head tracking (Ctrl + Shift + K)
keyboard.add_hotkey("ctrl+alt+0", toggle_head_tracking)
keyboard.add_hotkey("ctrl+alt+9", recalibrate)
keyboard.add_hotkey("ctrl+alt+8", shutdown)


def main():
    global neutral_yaw, neutral_pitch, cursor_control_active, frame_count, yaw_calibration, pitch_calibration
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 30  # Initial FPS estimate
    fps_update_interval = 1.0  # Update FPS every second
    frame_count = 0
    cursor_update_interval = math.ceil(fps / 5)  # Update cursor every 5th of a second

    # For calculating average AI processing time
    ai_times = deque(maxlen=30)  # Store last 30 AI processing times

    # Get screen size for cursor boundaries
    screen_width, screen_height = pyautogui.size()

    try:
        while True:

            # Receive multipart message (JSON payload and JPEG frame)
            parts = socket_sub.recv_multipart()
            if len(parts) != 2:
                continue

            # Decode the JSON payload.
            json_payload = parts[0].decode('utf-8')
            try:
                face_info = json.loads(json_payload)
            except json.JSONDecodeError:
                print("Error decoding JSON")
                continue

            # Decode the JPEG frame.
            jpg_bytes = parts[1]
            np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if frame is None:
                print("Error decoding image frame.")
                continue

            frame_count += 1

            # Extract AI processing time from the face_info
            ai_time_str = face_info.get("ai_time", "0 ms")
            try:
                ai_time = float(ai_time_str.split()[0])
            except:
                ai_time = 0.0
            ai_times.append(ai_time)
            avg_ai_time = sum(ai_times) / len(ai_times)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time > fps_update_interval:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                cursor_update_interval = math.ceil(fps / 5)

            # For each detected face, draw landmarks and head pose info.
            for face in face_info.get("faces", []):
                # Draw landmarks from the JSON data.
                for lm in face.get("landmarks", []):
                    x = int(lm["x"])
                    y = int(lm["y"])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Determine head pose direction using Euler angles.
                head_direction = ""
                nod_direction = ""
                euler = face.get("euler")
                if euler:
                    pitch, yaw, roll = euler

                    # Calibration phase: for the first calibration_frames, record yaw and pitch
                    if frame_count <= calibration_frames:
                        if frame_count < 0:
                                yaw_calibration = []
                                pitch_calibration = []
                                neutral_yaw = None
                                neutral_pitch = None
                        yaw_calibration.append(yaw)
                        pitch_calibration.append(pitch)
                        cv2.putText(frame, "Calibrating...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                        if frame_count == calibration_frames:
                            neutral_yaw = np.mean(yaw_calibration)
                            neutral_pitch = np.mean(pitch_calibration)
                            print(f"Calibration complete: Neutral Yaw = {neutral_yaw:.2f}, Neutral Pitch = {neutral_pitch:.2f}")
                        # Skip further processing during calibration
                        continue

                    # After calibration, adjust yaw and pitch relative to the neutral values.
                    adjusted_yaw = yaw - neutral_yaw
                    if adjusted_yaw < -180:
                        adjusted_yaw += 360
                    if adjusted_yaw > 180:
                        adjusted_yaw -= 360
                    adjusted_pitch = pitch - neutral_pitch
                    if adjusted_pitch < -180:
                        adjusted_pitch += 360
                    if adjusted_pitch > 180:
                        adjusted_pitch -= 360

                    # Determine turning direction based on adjusted yaw.
                    horizontal_move = 0
                    vertical_move = 0

                    if adjusted_yaw > yaw_threshold:
                        horizontal_move = adjusted_yaw * yaw_sensitivity
                    elif adjusted_yaw < -yaw_threshold:
                        horizontal_move = adjusted_yaw * yaw_sensitivity

                    if adjusted_pitch > pitch_threshold:
                        vertical_move = adjusted_pitch * pitch_sensitivity
                    elif adjusted_pitch < -pitch_threshold:
                        vertical_move = adjusted_pitch * pitch_sensitivity

                    if horizontal_move > 0:
                        head_direction = "Horizontal: Left"
                    elif horizontal_move < 0:
                        head_direction = "Horizontal: Right"
                    else:
                        head_direction = "Horizontal: Neutral"
                    
                    if vertical_move > 0:
                        nod_direction = "Vertical: Down"
                    elif vertical_move < 0:
                        nod_direction = "Vertical: Up"
                    else:
                        nod_direction = "Vertical: Neutral"

                    if head_tracking_enabled and frame_count % cursor_update_interval == 0:
                        # Calculate new cursor position
                        current_x, current_y = pyautogui.position()

                        # Calculate horizontal movement based on yaw (left/right)
                        new_x = current_x - horizontal_move

                        # Calculate vertical movement based on pitch (up/down)
                        new_y = current_y + vertical_move
                        
                        # Clamp the new position to screen boundaries
                        new_x = max(0, min(new_x, screen_width))
                        new_y = max(0, min(new_y, screen_height))

                        # Move cursor to new position
                        pyautogui.moveTo(new_x, new_y)

                    # Optionally, display numerical values.
                    cv2.putText(frame, f"Yaw: {adjusted_yaw:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Pitch: {adjusted_pitch:.1f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # Draw eye state information.
                
                global blink_frames
                if face.get('right_eye') == 'closed' and face.get('left_eye') == 'closed':
                    blink_frames += 1
                    if head_tracking_enabled and blink_frames >= BLINK_THRESHOLD:
                        pyautogui.click()
                        print("Mouse click triggered by blink")
                        blink_frames = -10  # Reset blink counter
                elif blink_frames > 0:
                    blink_frames = 0  # Reset if eyes are open
                if blink_frames < 0:
                    blink_frames += 1
                    
                    
                # Draw blink counter
                blink_text = f"Blink frames: {blink_frames}"
                cv2.putText(frame, blink_text, (10, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                eye_text = f"RE: {face.get('right_eye','?')}, LE: {face.get('left_eye','?')}"
                cv2.putText(frame, eye_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw head turning and nodding directions.
                cv2.putText(frame, head_direction, (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, nod_direction, (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw FPS and AI processing time
                fps_text = f"FPS: {fps:.2f}"
                ai_text = f"AI: {avg_ai_time:.2f} ms"
                height_frame, width_frame = frame.shape[:2]
                fps_pos = (width_frame - 150, height_frame - 40)
                ai_pos = (width_frame - 150, height_frame - 20)
                cv2.putText(frame, fps_text, fps_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(frame, ai_text, ai_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    # Draw cursor control indicator circle
                if head_tracking_enabled:
                    cv2.circle(frame, (width_frame - 30, 30), 15, (0, 0, 255), -1)  # Red filled circle when active
                    # Draw head turning and nodding directions.
                    cv2.putText(frame, "ctrl+alt+0 : Cursor control off", (10, height_frame-50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.circle(frame, (width_frame - 30, 30), 15, (0, 0, 255), 2)  # White outline when inactive
                    cv2.putText(frame, "ctrl+alt+0 : Cursor control on", (10, height_frame-60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "ctrl+alt+9 : Recalibrate position", (10, height_frame-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "ctrl+alt+8 : Terminate Process", (10, height_frame-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Shutting down control.")


if __name__ == '__main__':
    main()
