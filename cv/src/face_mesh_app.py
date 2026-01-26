import cv2
import mediapipe as mp
import time
import sys
import os
import urllib.request
import zmq
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def download_model():
    # We use the same model, but we will enable a different output feature
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    filename = "face_landmarker.task"
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename

def remap(value, low, high):
    return max(0.0, min(1.0, (value - low) / (high - low)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python face_mesh_blendshapes.py path/to/video.mp4")
        return

    video_path = sys.argv[1]
    model_path = download_model()

    # Inicjalizacja ZeroMQ (Publisher) do komunikacji z modułem animacji
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,  # <--- THIS IS THE KEY CHANGE
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO 
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        
        # Timing setup
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame_time = 1 / source_fps if source_fps > 0 else 0.033
        frame_time_ms = 1000 / source_fps if source_fps > 0 else 33
        frame_number = 0
        
        is_paused = False
        debug_mode = False
        current_frame = None
        
        # Variables for display
        l_open, r_open, mouth_open = 0, 0, 0
        timestamp = 0

        print("------------------------------------------------")
        print("Using ML Blendshapes for better accuracy.")
        print("Press SPACE to PAUSE/RESUME.")
        print("Press 'b' to print all current blendshapes.")
        print("Press 'd' to toggle debug mode (show all landmarks).")
        print("Press 'q' to QUIT.")
        print("------------------------------------------------")

        # Force window to open on primary screen (top-left)
        cv2.namedWindow('Face Mesh Blendshapes')
        cv2.moveWindow('Face Mesh Blendshapes', 0, 0)

        while cap.isOpened():
            loop_start = time.time()

            if not is_paused:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp = int(frame_number * frame_time_ms)
                frame_number += 1
                
                result = landmarker.detect_for_video(mp_image, timestamp)
                
                if result.face_blendshapes:
                    # Blendshapes return a list of 52 scores (0.0 to 1.0)
                    # We map them to a dictionary for easy name access
                    shapes = result.face_blendshapes[0]
                    bs_map = { cat.category_name: cat.score for cat in shapes }

                    # 1. Eye Openness
                    # The model outputs 'eyeBlinkLeft' (0=Open, 1=Closed).
                    # We invert it because you asked for "Openness" (0=Closed, 1=Open).
                    l_open = 1.0 - bs_map.get('eyeBlinkLeft', 0.0)
                    r_open = 1.0 - bs_map.get('eyeBlinkRight', 0.0)

                    # Remap openness to ensure fully closed eyes are 0.0
                    l_open = remap(l_open, 0.2, 0.8)
                    r_open = remap(r_open, 0.2, 0.8)

                    # 3. Mouth Openness (jawOpen)
                    mouth_open = bs_map.get('jawOpen', 0.0)

                    # Przygotowanie danych do wysłania przez ZeroMQ
                    payload = {
                        "timestamp": timestamp,
                        "eyes": {"left": l_open, "right": r_open},
                        "mouth": {"open": mouth_open}
                    }
                    socket.send_string(json.dumps(payload))

                    # --- VISUALIZATION ---
                    # Draw text
                    cv2.putText(frame, f"L Open: {l_open:.2f}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"R Open: {r_open:.2f}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Mouth:  {mouth_open:.2f}", (20, 130), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw landmarks just for verification
                if result.face_landmarks:
                    face = result.face_landmarks[0]
                    h, w, _ = frame.shape
                    if debug_mode:
                        for lm in face:
                            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 255), -1)
                    else:
                        # Draw just the eye corners and lips to keep it clean
                        for idx in [33, 133, 362, 263, 61, 291]:
                            lm = face[idx]
                            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0, 0, 255), -1)

                current_frame = frame

            # --- DISPLAY & INPUT ---
            if current_frame is not None:
                img_to_show = current_frame.copy()

                # Resize for display to fit on screen
                h, w = img_to_show.shape[:2]
                scale = 1.0
                if w > 960:
                    scale = 960 / w
                if h * scale > 1080:
                    scale = 1080 / h
                if scale < 1.0:
                    img_to_show = cv2.resize(img_to_show, (int(w * scale), int(h * scale)))

                if is_paused:
                    cv2.putText(img_to_show, "PAUSED", (20, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add exit text
                h_display, w_display = img_to_show.shape[:2]
                cv2.putText(img_to_show, "Press 'q' to exit", (w_display - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Face Mesh Blendshapes', img_to_show)

            if is_paused:
                wait_time = 30
            else:
                proc_time = time.time() - loop_start
                wait_time = max(1, int((target_frame_time - proc_time) * 1000))

            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_paused = not is_paused
                if is_paused:
                    print(f"\n[PAUSED] Timestamp: {timestamp}ms")
                    print(f" > Left Open:  {l_open:.4f}")
                    print(f" > Right Open: {r_open:.4f}")
                    print(f" > Mouth Open: {mouth_open:.4f}")
                else:
                    print("[RESUMED]")
            elif key == ord('b'):
                if 'bs_map' in locals():
                    print(f"\n--- Blendshapes at {timestamp}ms ---")
                    for name, score in sorted(bs_map.items()):
                        print(f"{name}: {score:.4f}")
                else:
                    print("\nNo blendshapes detected yet.")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()