import os
# Suppress OpenCV Qt and Wayland warnings on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import mediapipe as mp
import time
import urllib.request
import argparse
import numpy as np
import math
import socket
import json
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

def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0', help='Camera index or path to mp4 file')
    parser.add_argument('--debug', action='store_true', help='Show video window with landmarks for debugging')
    parser.add_argument('--port', type=int, default=5555, help='UDP port to send data to')
    parser.add_argument('--fps', type=float, default=0.0, help='Force source FPS (overrides video metadata). Use when the container reports the wrong rate, e.g. 60 for a 30fps clip.')
    args = parser.parse_args()

    # UDP Setup
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target_addr = ("127.0.0.1", args.port)

    # AUTO-LOAD CALIBRATION
    cal_file = "calibration_result.json"
    if os.path.exists(cal_file):
        try:
            with open(cal_file, "r") as f:
                cal_data = json.load(f)
                udp_socket.sendto(json.dumps(cal_data).encode('utf-8'), target_addr)
                print(f"Applied saved calibration from {cal_file}")
        except Exception as e:
            print(f"Warning: Failed to load calibration: {e}")

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

    # Pobieranie Hz z metadanych pliku wideo
    source_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 0.0
    if is_video_file and args.fps > 0:
        # Manual override: trust the user over the (often wrong) container metadata.
        print(f"Override: forcing source FPS to {args.fps:.3f} (metadata reported {source_fps:.3f}).")
        source_fps = args.fps
    elif is_video_file and (not source_fps or source_fps <= 0 or math.isnan(source_fps)):
        source_fps = 30.0
        print(f"Warning: source FPS missing from metadata, assuming {source_fps:.0f} fps.")
    
    frame_interval = (1.0 / source_fps) if (is_video_file and source_fps > 0) else 0.0
    if is_video_file:
        print(f"Source frame rate: {source_fps:.3f} fps (frame interval {frame_interval * 1000:.2f} ms).")
    
    video_frame_index = 0
    mp_timestamp_index = 0
    frames_processed = 0
    total_time = 0.0
    stats_wall_start = time.time()

    # Calibration State
    is_calibrating = False
    calibration_frames = 0
    calibration_max_frames = 60 # ~1 second at 60fps
    accumulated_ratios = {}
    accumulated_color = np.array([0.0, 0.0, 0.0])
    final_ratios = None
    final_color = None

    print(f"Starting pipeline on source: {args.source} | Target UDP: 127.0.0.1:{args.port} | Debug: {args.debug}")
    if args.debug:
        print("Press SPACE to PAUSE/RESUME, 'c' to START CALIBRATION, 'q' to QUIT.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        is_paused = False
        while True:
            # Zaczynamy mierzyć czas DOKŁADNIE na początku klatki
            start_time = time.time()

            if not is_paused:
                success, frame = cap.read()
                if not success:
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                        video_frame_index = 0
                        continue
                    else:
                        break

                # Prawidłowy timestamp oparty na klatkach (wymagane przez MediaPipe).
                # MediaPipe wymaga MONOTONICZNIE rosnącego timestampu, więc używamy osobnego licznika,
                # który NIE resetuje się przy zapętleniu wideo (video_frame_index resetuje do 0 i cofałby czas).
                video_time_s = (video_frame_index * frame_interval) if frame_interval > 0 else 0.0

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                if frame_interval > 0:
                    fake_timestamp_ms = int(mp_timestamp_index * frame_interval * 1000)
                else:
                    fake_timestamp_ms = int(time.time() * 1000)

                video_frame_index += 1
                mp_timestamp_index += 1

                result = landmarker.detect_for_video(mp_image, fake_timestamp_ms)
                
                if result.face_blendshapes and result.facial_transformation_matrixes:
                    shapes = result.face_blendshapes[0]
                    bs_map = { cat.category_name: cat.score for cat in shapes }
                    
                    landmarks = result.face_landmarks[0]
                    
                    transform_matrix = result.facial_transformation_matrixes[0]
                    rotation_matrix = transform_matrix[:3, :3]
                    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)
                    pitch_deg, yaw_deg, roll_deg = map(math.degrees, [pitch, yaw, roll])

                    # UDP BROADCAST
                    if not is_calibrating:
                        anim_payload = {
                            "type": "animation",
                            "timestamp": fake_timestamp_ms,
                            "rotation": {"pitch": pitch_deg, "yaw": yaw_deg, "roll": roll_deg},
                            "blendshapes": bs_map
                        }
                        udp_socket.sendto(json.dumps(anim_payload).encode('utf-8'), target_addr)

                    # CALIBRATION LOGIC
                    if is_calibrating:
                        def dist(i1, i2): return calculate_distance(landmarks[i1], landmarks[i2])
                        
                        # Use face height as anchor (Forehead 10 to Chin 152)
                        face_height = dist(10, 152)
                        
                        if face_height > 0.001:
                            # Raw ratios relative to face height
                            raw = {
                                'headWidth': dist(234, 454) / face_height,
                                'neckThickness': dist(132, 361) / face_height,
                                'noseWidth': dist(129, 358) / face_height,
                                'noseSize': (dist(168, 1) * dist(129, 358)) / (face_height**2),
                                'nosePosition': dist(168, 1) / face_height,
                                'nosePronounced': (landmarks[168].z - landmarks[1].z),
                                'noseFlatten': (landmarks[1].z - landmarks[168].z),
                                'chinSize': dist(17, 152) / face_height,
                                'chinPronounced': (landmarks[1].z - landmarks[152].z),
                                'chinPosition': dist(1, 152) / face_height,
                                'mandibleSize': dist(132, 361) / face_height,
                                'jawsSize': dist(172, 397) / face_height,
                                'jawsPosition': dist(1, 152) / face_height,
                                'cheekSize': dist(116, 345) / face_height,
                                'cheekPosition': dist(168, 116) / face_height,
                                'lowCheekPronounced': (landmarks[168].z - landmarks[116].z),
                                'lowCheekPosition': dist(1, 116) / face_height,
                                'foreheadSize': dist(10, 168) / face_height,
                                'lipsSize': dist(0, 17) / face_height,
                                'mouthSize': dist(61, 291) / face_height,
                                'eyeSize': (dist(159, 145) + dist(386, 374)) / (2 * face_height),
                                'eyeSpacing': dist(133, 362) / face_height,
                            }

                            # Baselines for neutral 0.5
                            baselines = {
                                'headWidth': 0.75,
                                'neckThickness': 0.55,
                                'noseWidth': 0.18,
                                'noseSize': 0.05,
                                'nosePosition': 0.25,
                                'nosePronounced': 0.05,
                                'noseFlatten': -0.05,
                                'chinSize': 0.12,
                                'chinPronounced': 0.02,
                                'chinPosition': 0.35,
                                'mandibleSize': 0.55,
                                'jawsSize': 0.65,
                                'jawsPosition': 0.35,
                                'cheekSize': 0.70,
                                'cheekPosition': 0.20,
                                'lowCheekPronounced': 0.02,
                                'lowCheekPosition': 0.15,
                                'foreheadSize': 0.25,
                                'lipsSize': 0.08,
                                'mouthSize': 0.35,
                                'eyeSize': 0.06,
                                'eyeSpacing': 0.28,
                            }

                            dna_params = {}
                            # SENSITIVITY: 1.0 = full mapping, 0.2 = very subtle changes.
                            sensitivity = 0.4 
                            
                            # Standard parameters
                            for k, v in raw.items():
                                base = baselines.get(k, 0.5)
                                
                                # Specific dampening for cheeks to avoid "sharp/skeletal" look
                                current_sensitivity = sensitivity
                                if 'cheek' in k.lower() or 'Cheek' in k:
                                    current_sensitivity *= 0.5 # Extra soft for cheeks

                                if 'Pronounced' in k or 'Flatten' in k:
                                    # Depth is small, use different scaling
                                    dna_params[k] = np.clip(0.5 + (v - base) * 2.0 * current_sensitivity, 0.0, 1.0)
                                else:
                                    # Calculate % deviation from baseline and apply sensitivity
                                    deviation = (v - base) / base
                                    dna_params[k] = np.clip(0.5 + (deviation * current_sensitivity), 0.0, 1.0)
                            
                            # Reset headSize to neutral 0.5 (safe baseline)
                            dna_params['headSize'] = 0.5
                            dna_params['height'] = 0.5 

                            
                            if not accumulated_ratios:
                                accumulated_ratios = {k: 0.0 for k in dna_params.keys()}
                                
                            for k, v in dna_params.items():
                                accumulated_ratios[k] += v

                            h, w, _ = frame_rgb.shape
                            # Sample skin color from forehead and cheeks
                            color_points = [151, 117, 346]
                            frame_color = np.array([0.0, 0.0, 0.0])
                            pts_sampled = 0
                            
                            for idx in color_points:
                                lx = int(landmarks[idx].x * w)
                                ly = int(landmarks[idx].y * h)
                                if 0 <= lx < w and 0 <= ly < h:
                                    frame_color += frame_rgb[ly, lx]
                                    pts_sampled += 1
                                    
                            if pts_sampled > 0:
                                accumulated_color += (frame_color / pts_sampled)
                            
                            calibration_frames += 1

                            if calibration_frames >= calibration_max_frames:
                                is_calibrating = False
                                final_dna = {
                                    k: v / calibration_max_frames for k, v in accumulated_ratios.items()
                                }
                                final_color = (accumulated_color / calibration_max_frames).astype(int)
                                
                                # Add skin color DNA
                                final_dna['skinRedness'] = final_color[0] / 255.0
                                final_dna['skinGreenness'] = final_color[1] / 255.0
                                final_dna['skinBlueness'] = final_color[2] / 255.0

                                cal_payload = {
                                    "type": "calibration",
                                    "dna": final_dna,
                                    "skin_color": final_color.tolist()
                                }
                                udp_socket.sendto(json.dumps(cal_payload).encode('utf-8'), target_addr)

                                print(f"\n--- CALIBRATION COMPLETE ---")
                                print("UMA DNA Parameters:")
                                for k, v in final_dna.items():
                                    print(f"  {k:20s}: {v:.4f}")
                                print(f"\nSkin Color (RGB): {final_color.tolist()}")
                                print(f"----------------------------\n")

                    if args.debug:
                        h, w, _ = frame.shape
                        for idx in [1, 33, 263, 61, 291]: 
                            lm = landmarks[idx]
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                        
                        for idx in [151, 117, 346]:
                            lm = landmarks[idx]
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)
                        
                        y_pos = 30
                        cv2.putText(frame, f"Pitch: {pitch_deg:.1f} Yaw: {yaw_deg:.1f} Roll: {roll_deg:.1f}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        if is_calibrating:
                            progress = int((calibration_frames / calibration_max_frames) * 100)
                            cv2.putText(frame, f"CALIBRATING: {progress}%", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        elif final_ratios:
                            cv2.putText(frame, f"CALIBRATED", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            color_bgr = (int(final_color[2]), int(final_color[1]), int(final_color[0]))
                            cv2.rectangle(frame, (450, 10), (490, 50), color_bgr, -1)
                            cv2.rectangle(frame, (450, 10), (490, 50), (255, 255, 255), 1)
                        
                        col_width = 180
                        row_height = 15
                        start_y = 60
                        
                        for i, (shape_name, val) in enumerate(bs_map.items()):
                            col = i // 25
                            row = i % 25
                            x_pos = 20 + (col * col_width)
                            y_pos = start_y + (row * row_height)
                            
                            color = (0, 255, 0) if val > 0.1 else (200, 200, 200)
                            cv2.putText(frame, f"{shape_name}: {val:.2f}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # --- NOWA LOGIKA SYNCHRONIZACJI WIDEO DO ORYGINALNEGO FPS ---
            process_time = time.time() - start_time
            wait_time_ms = 1  # Domyślny, minimalny narzut czasu (1 ms)

            # Obliczanie ile trzeba odczekać (tylko gdy odtwarzamy z pliku wideo i klatka się załadowała)
            if not is_paused and is_video_file and frame_interval > 0:
                remaining_time = frame_interval - process_time
                if remaining_time > 0:
                    wait_time_ms = int(remaining_time * 1000)
                    if wait_time_ms <= 0:
                        wait_time_ms = 1

            if args.debug:
                display_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if is_paused:
                    cv2.putText(display_frame, "PAUSED", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Debug - Headless Pipeline", display_frame)
                
                # Używamy wyliczonego czasu jako argumentu opóźnienia w OpenCV
                key = cv2.waitKey(wait_time_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    is_paused = not is_paused
                elif key == ord('c') and not is_calibrating:
                    print("Starting calibration... keep neutral face.")
                    is_calibrating = True
                    calibration_frames = 0
                    accumulated_ratios = {}
                    accumulated_color = np.array([0.0, 0.0, 0.0])
            else:
                # W trybie "headless" używamy time.sleep do wstrzymania pętli
                if wait_time_ms > 1:
                    time.sleep(wait_time_ms / 1000.0)

            # Statystyki wydajności
            if not is_paused:
                process_time_ms = process_time * 1000
                total_time += process_time_ms
                frames_processed += 1

                if frames_processed % 60 == 0:
                    avg_time = total_time / 60
                    inference_fps = 1000.0 / avg_time if avg_time > 0 else 0
                    now = time.time()
                    wall_elapsed = now - stats_wall_start
                    effective_fps = 60.0 / wall_elapsed if wall_elapsed > 0 else 0
                    target_str = f" | Target: {source_fps:.1f}" if (is_video_file and source_fps > 0) else ""
                    print(f"[Stats] Frames: {frames_processed} | Inference: {avg_time:.2f}ms ({inference_fps:.1f} fps) | Effective: {effective_fps:.1f} fps{target_str}")
                    total_time = 0.0
                    stats_wall_start = now

        if args.debug:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()