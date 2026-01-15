#!/usr/bin/env python
import os
import sys
import argparse
import time
import traceback
import gc
import copy
import json
import cv2
import zmq
import numpy as np
import socket
import struct

# Import your custom modules (ensure these are in your PYTHONPATH)
from input_reader import InputReader, VideoReader
from tracker import Tracker



# --- Command-line arguments (simplified subset) ---
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=480)
parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=24)
parser.add_argument("-M", "--mirror-input", action="store_true", help="Process a mirror image of the input video")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces", default=1)
parser.add_argument("--model", type=int, help="Select tracking model", default=3, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="Path to the directory containing the .onnx model files", default=None)
parser.add_argument("--zmq-port", type=int, help="Port for ZeroMQ PUB socket", default=5555)
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

# --- Set up ZeroMQ publisher ---
context = zmq.Context()
socket_pub = context.socket(zmq.PUB)
socket_pub.bind(f"tcp://*:{args.zmq_port}")


input_reader = InputReader(args.capture, 0, args.width, args.height, args.fps)
if isinstance(input_reader.reader, VideoReader):
    fps = 0
else:
    fps = args.fps

def main():
    tracker = None
    first_frame = True
    frame_count = 0

    try:
        while input_reader.is_open():
            if not input_reader.is_ready():
                time.sleep(0.02)
                continue

            ret, frame = input_reader.read()
            if ret and args.mirror_input:
                frame = cv2.flip(frame, 1)
            if not ret:
                break

            frame_count += 1
            now = time.time()

            # Initialize tracker on first valid frame
            if first_frame:
                first_frame = False
                height, width, channels = frame.shape
                tracker = Tracker(width, height,
                                  threshold=args.threshold,
                                  max_threads=args.max_threads,
                                  max_faces=args.faces,
                                  detection_threshold=args.detection_threshold,
                                  model_type=args.model,
                                  model_dir=args.model_dir)
            
            # Run face tracking analysis
            # Inside the main loop, just before running the face tracking analysis
            start_time = time.time()

            # Run face tracking analysis
            faces = tracker.predict(frame)

            

            
            # Build JSON payload for each detected face.
            face_data = []
            for face in faces:
                # Ensure eye_blink exists; default to open if missing.
                if not hasattr(face, "eye_blink") or face.eye_blink is None:
                    face.eye_blink = [0.0, 0.0]  # [right, left]
                # Adjust threshold as needed (here 0.6 worked better with your webcam)
                eye_threshold = 0.7
                right_state = "open" if face.eye_blink[0] > eye_threshold else "closed"
                left_state  = "open" if face.eye_blink[1] > eye_threshold else "closed"

                # Prepare landmarks list.
                landmarks = []
                if hasattr(face, "lms") and face.lms is not None:
                    for pt in face.lms:
                        # Swap x and y if needed so that drawing is correct.
                        landmarks.append({
                            "x": float(pt[1]),
                            "y": float(pt[0]),
                            "confidence": float(pt[2])
                        })
                
                # Include Euler angles if available (assumed order: [pitch, yaw, roll])
                euler_angles = None
                if hasattr(face, "euler") and face.euler is not None:
                    euler_angles = [float(face.euler[0]), float(face.euler[1]), float(face.euler[2])]

                face_data.append({
                    "id": int(face.id),
                    "confidence": float(face.conf),
                    "pnp_error": float(face.pnp_error),
                    "right_eye": right_state,
                    "left_eye": left_state,
                    "landmarks": landmarks,
                    "euler": euler_angles
                })

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # When creating the payload, add the processing time
            payload = {
                "timestamp": now,
                "frame_count": frame_count,
                "faces": face_data,
                "ai_time": f"{processing_time:.2f} ms"
            }
            json_payload = json.dumps(payload)

            # Encode the frame as JPEG for efficient transfer.
            ret_enc, buffer = cv2.imencode('.jpg', frame)
            if not ret_enc:
                print("Warning: Failed to encode frame.")
                continue
            jpg_bytes = buffer.tobytes()

            # Send a multipart message: JSON payload and JPEG frame.
            socket_pub.send_multipart([json_payload.encode('utf-8'), jpg_bytes])
            time.sleep(0.03)
            gc.collect()

    except KeyboardInterrupt:
        print("Shutting down facetracker_module.")
    except Exception as e:
        traceback.print_exc()
    finally:
        input_reader.close()
        socket_pub.close()
        context.term()

if __name__ == '__main__':
    main()
