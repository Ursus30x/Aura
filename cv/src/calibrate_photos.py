import os
import cv2
import mediapipe as mp
import numpy as np
import json
import socket
import argparse
import math
import colorsys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Suppress warnings
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

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
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

class MultiViewCalibrator:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.sensitivity = 0.6
        
        # Biological Min/Max ranges
        self.ranges = {
            'headWidth': (1.4, 1.8),
            'neckThickness': (1.2, 1.8),
            'noseWidth': (0.12, 0.22), # Scentrowane na raw ~0.17
            'noseSize': (0.35, 0.55),  # Scentrowane na raw ~0.46
            'nosePosition': (0.35, 0.6),
            'nosePronounced': (0.5, 1.0),
            'noseFlatten': (-1.0, -0.5),
            'noseCurve': (-3.0, -1.0),
            'chinSize': (0.25, 0.5),
            'chinPronounced': (0.05, 0.3),
            'chinPosition': (0.6, 1.0),
            'mandibleSize': (1.2, 1.8),
            'jawsSize': (1.1, 1.6),
            'cheekSize': (1.2, 1.7),
            'cheekPosition': (0.6, 1.0),
            'lowCheekPronounced': (0.3, 0.7),
            'foreheadSize': (0.3, 0.6),
            'lipsSize': (0.1, 0.4),   # Widen range so standard lips map closer to 0.5 (prevents huge upper lip)
            'mouthSize': (0.4, 0.8),
            'eyeSize': (0.05, 0.15),  # Lowered min/max to prevent huge eye sockets (raw is usually ~0.06)
            'eyeSpacing': (0.35, 0.6),
        }

    def process_folder(self, folder_path):
        results = {}
        skin_colors = []
        debug_dir = "debug_calibration"
        os.makedirs(debug_dir, exist_ok=True)
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: return None

        for filename in files:
            img_path = os.path.join(folder_path, filename)
            original_frame = cv2.imread(img_path)
            if original_frame is None: continue
            
            face_found = False
            for angle in [0, 90, 180, 270]:
                if angle == 0: frame = original_frame
                elif angle == 90: frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180: frame = cv2.rotate(original_frame, cv2.ROTATE_180)
                else: frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                result = self.landmarker.detect(mp_image)
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    transform_matrix = result.facial_transformation_matrixes[0]
                    rotation = rotation_matrix_to_euler_angles(transform_matrix[:3, :3])
                    
                    view_type = self.identify_view(filename, rotation)
                    print(f"Processing {filename} as view: {view_type} (Rot: {rotation[0]:.1f}, {rotation[1]:.1f})")
                    
                    h, w, _ = frame.shape
                    view_data = self.extract_features(landmarks, view_type, w, h)
                    results[view_type] = view_data
                    
                    debug_frame = frame.copy()
                    h, w, _ = debug_frame.shape
                    for i, lm in enumerate(landmarks):
                        if i % 10 == 0: cv2.circle(debug_frame, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 0), -1)
                    for idx in [33, 263]: cv2.circle(debug_frame, (int(landmarks[idx].x*w), int(landmarks[idx].y*h)), 3, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(debug_dir, f"debug_{filename}"), debug_frame)
                    
                    if view_type == 'front':
                        skin_colors.append(self.extract_skin_color(frame_rgb, landmarks))
                    face_found = True
                    break
            
            if not face_found: print(f"No face detected in {filename}")

        if not results: return None
        
        final_dna = self.aggregate_results(results)
        final_color = np.mean(skin_colors, axis=0).astype(int).tolist() if skin_colors else [200, 180, 160]
        
        return {"type": "calibration", "ratios": final_dna, "skin_color": final_color}

    def identify_view(self, filename, rotation):
        fname = filename.lower()
        if 'front' in fname: return 'front'
        if 'left' in fname: return 'left'
        if 'right' in fname: return 'right'
        if 'top' in fname: return 'top'
        if 'bottom' in fname: return 'bottom'
        pitch, yaw, roll = rotation
        if abs(yaw) < 15 and abs(pitch) < 15: return 'front'
        if yaw > 20: return 'left'
        if yaw < -20: return 'right'
        if pitch > 15: return 'bottom'
        if pitch < -15: return 'top'
        return 'front'

    def extract_features(self, landmarks, view_type, w, h):
        def dist_2d(i1, i2): 
            dx = (landmarks[i1].x - landmarks[i2].x) * w
            dy = (landmarks[i1].y - landmarks[i2].y) * h
            return math.sqrt(dx**2 + dy**2)
            
        # MediaPipe Z is scaled roughly to X (width)
        def z_val(idx): return landmarks[idx].z * w

        # Anchor on inner eye corners instead of outer corners
        anchor = dist_2d(133, 362)
        if anchor < 0.001: return {}
        data = {
            'headWidth': dist_2d(234, 454) / anchor,
            'eyeSpacing': dist_2d(133, 362) / anchor, # Should be 1.0
            'noseWidth': dist_2d(193, 417) / anchor, # Bridge
            'noseSize': dist_2d(129, 358) / anchor,  # Nostrils
            'mouthSize': dist_2d(61, 291) / anchor,
            'chinSize': dist_2d(17, 152) / anchor,
            'chinPosition': dist_2d(1, 152) / anchor,
            'mandibleSize': dist_2d(132, 361) / anchor,
            'jawsSize': dist_2d(172, 397) / anchor,
            'cheekSize': dist_2d(116, 345) / anchor,
            'cheekPosition': dist_2d(168, 116) / anchor,
            'foreheadSize': dist_2d(10, 168) / anchor,
            'lipsSize': dist_2d(0, 17) / anchor,
            'neckThickness': dist_2d(132, 361) / anchor,
            'nosePosition': dist_2d(168, 1) / anchor,
            'eyeSize': (dist_2d(159, 145) + dist_2d(386, 374)) / (2 * anchor)
        }
        data['nosePronounced'] = abs(z_val(1) - z_val(234)) / anchor
        data['noseFlatten'] = -data['nosePronounced']
        data['chinPronounced'] = abs(z_val(152) - z_val(17)) / anchor
        data['lowCheekPronounced'] = abs(z_val(116) - z_val(1)) / anchor
        if view_type in ['left', 'right']:
            dy = (landmarks[1].y - landmarks[168].y) * h
            dz = (z_val(1) - z_val(168)) + 0.001
            data['noseCurve'] = dy / dz
        return data

    def extract_skin_color(self, img_rgb, landmarks):
        h, w, _ = img_rgb.shape
        # Points: Forehead center, Left Cheek, Right Cheek
        color_points = [151, 117, 346]
        samples = []
        
        for idx in color_points:
            lx, ly = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            # Sample a 5x5 patch instead of a single pixel to reduce noise
            patch_size = 2
            for dy in range(-patch_size, patch_size + 1):
                for dx in range(-patch_size, patch_size + 1):
                    px, py = lx + dx, ly + dy
                    if 0 <= px < w and 0 <= py < h:
                        px_color = img_rgb[py, px].astype(float)
                        samples.append(px_color)
                        
        if not samples:
            return [200, 180, 160]
            
        avg_color = np.mean(samples, axis=0)
        
        # Proper Color Correction using HSV
        # 1. Normalize RGB to 0.0 - 1.0 for colorsys
        r_norm, g_norm, b_norm = avg_color[0]/255.0, avg_color[1]/255.0, avg_color[2]/255.0
        
        # 2. Convert to HSV (Hue, Saturation, Value)
        hue, sat, val = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
        
        # 3. Aggressive anti-orange/red correction
        # UMA adds its own warmth/Subsurface Scattering. We must provide a very desaturated, pale base.
        sat = min(sat, 0.25)  # Hard clamp saturation (0.25 is quite pale)
        
        # Shift hue to a neutral yellowish-beige to counteract UMA's internal red SSS
        # Orange/Red is typically hue 0.0 to 0.1.
        if hue < 0.1 or hue > 0.9:
            hue = 0.12 # Cooler, yellowish beige
        
        val = min(1.0, val * 1.3) # Boost brightness further
        
        # 4. Convert back to RGB
        r_new, g_new, b_new = colorsys.hsv_to_rgb(hue, sat, val)
        
        return [int(r_new * 255), int(g_new * 255), int(b_new * 255)]

    def aggregate_results(self, results):
        merged_raw = {}
        all_keys = set()
        for v in results.values(): all_keys.update(v.keys())
        for key in all_keys:
            vals = [v[key] for v in results.values() if key in v]
            merged_raw[key] = np.mean(vals)
            
        print("\nRaw Aggregated Ratios:")
        for k, v in merged_raw.items(): print(f"  {k:20s}: {v:.4f}")
            
        return merged_raw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Path to directory with calibration photos')
    parser.add_argument('--port', type=int, default=5555, help='UDP port for Unity')
    args = parser.parse_args()
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path): return
    calibrator = MultiViewCalibrator(model_path)
    payload = calibrator.process_folder(args.dir)
    if payload:
        print("\nFinal DNA Calibration:")
        for k, v in payload['ratios'].items(): print(f"  {k:20s}: {v:.4f}")
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(json.dumps(payload).encode('utf-8'), ("127.0.0.1", args.port))
        with open("calibration_result.json", "w") as f: json.dump(payload, f, indent=2)
        print("Calibration complete.")

if __name__ == "__main__":
    main()
