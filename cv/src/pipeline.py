import cv2
import mediapipe as mp
import time
import sys
import os
import urllib.request
import argparse
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model():
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    filename = "face_landmarker.task"
    if not os.path.exists(filename):
        print(f"Downloading model to {filename}...")
        urllib.request.urlretrieve(url, filename)
    return filename

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    # Returns pitch, yaw, roll in radians
    return np.array([x, y, z])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='Camera index or path to mp4 file')
    parser.add_argument('--debug', action='store_true', help='Show video window with landmarks for debugging')
    args = parser.parse_args()

    model_path = download_model()

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO 
    )

    is_video_file = not args.source.isdigit()
    source_val = args.source if is_video_file else int(args.source)

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Cannot open source {args.source}")
        return

    frames_processed = 0
    total_time = 0.0

    print(f"Starting pipeline on source: {args.source} | Debug mode: {args.debug}")
    if args.debug:
        print("Press SPACE to PAUSE/RESUME, 'q' to QUIT.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        is_paused = False
        while True:
            start_time = time.time()

            if not is_paused:
                success, frame = cap.read()
                if not success:
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                        continue
                    else:
                        break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Mediapipe requires strictly increasing timestamps for VIDEO mode.
                fake_timestamp_ms = int(time.time() * 1000)

                result = landmarker.detect_for_video(mp_image, fake_timestamp_ms)
                
                if result.face_blendshapes and result.facial_transformation_matrixes:
                    # 1. 52 ARKit blendshapes
                    shapes = result.face_blendshapes[0]
                    bs_map = { cat.category_name: cat.score for cat in shapes }
                    
                    # 2. 478 3D Landmarks (can be accessed via result.face_landmarks[0])
                    landmarks = result.face_landmarks[0]
                    
                    # 3. Head Rotation (Pitch, Yaw, Roll)
                    transform_matrix = result.facial_transformation_matrixes[0]
                    # Extract 3x3 rotation matrix
                    rotation_matrix = transform_matrix[:3, :3]
                    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)
                    pitch_deg, yaw_deg, roll_deg = map(math.degrees, [pitch, yaw, roll])

                    if args.debug:
                        h, w, _ = frame.shape
                        # Draw a few key landmarks (e.g. nose tip, eye corners)
                        for idx in [1, 33, 263, 61, 291]: 
                            lm = landmarks[idx]
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                        
                        y_pos = 30
                        # Rotation
                        cv2.putText(frame, f"Pitch: {pitch_deg:.1f} Yaw: {yaw_deg:.1f} Roll: {roll_deg:.1f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # All 52 Blendshapes in multiple columns
                        col_width = 180
                        row_height = 15
                        start_y = 60
                        
                        for i, (shape_name, val) in enumerate(bs_map.items()):
                            col = i // 25
                            row = i % 25
                            x_pos = 20 + (col * col_width)
                            y_pos = start_y + (row * row_height)
                            
                            color = (0, 255, 0) if val > 0.1 else (200, 200, 200) # Highlight active shapes
                            cv2.putText(frame, f"{shape_name}: {val:.2f}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            if args.debug:
                display_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if is_paused:
                    cv2.putText(display_frame, "PAUSED", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Debug - Headless Pipeline", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    is_paused = not is_paused

            if not is_paused:
                process_time = (time.time() - start_time) * 1000
                total_time += process_time
                frames_processed += 1

                if frames_processed % 60 == 0:
                    avg_time = total_time / 60
                    fps = 1000.0 / avg_time if avg_time > 0 else 0
                    if not args.debug:
                        print(f"[Stats] Frames: {frames_processed} | Avg processing time: {avg_time:.2f}ms | FPS: {fps:.1f}")
                        if result.face_blendshapes:
                            print(f"   -> Head Rotation (deg): Pitch={pitch_deg:.1f}, Yaw={yaw_deg:.1f}, Roll={roll_deg:.1f}")
                            print(f"   -> jawOpen: {bs_map.get('jawOpen', 0.0):.3f}")
                    total_time = 0.0
                
        if args.debug:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()