import os
import cv2
import mediapipe as mp
import numpy as np
import json
import socket
import argparse
import math
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
            'lipsSize': (0.1, 0.3),
            'mouthSize': (0.4, 0.8),
            'eyeSize': (0.04, 0.1),
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
                    
                    view_data = self.extract_features(landmarks, view_type)
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
        
        return {"type": "calibration", "dna": final_dna, "skin_color": final_color}

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

    def extract_features(self, landmarks, view_type):
        def dist(i1, i2): return calculate_distance(landmarks[i1], landmarks[i2])
        anchor = dist(33, 263)
        if anchor < 0.001: return {}
        data = {
            'headWidth': dist(234, 454) / anchor,
            'eyeSpacing': dist(133, 362) / anchor,
            'noseWidth': dist(193, 417) / anchor, # Bridge
            'noseSize': dist(129, 358) / anchor,  # Nostrils
            'mouthSize': dist(61, 291) / anchor,
            'chinSize': dist(17, 152) / anchor,
            'chinPosition': dist(1, 152) / anchor,
            'mandibleSize': dist(132, 361) / anchor,
            'jawsSize': dist(172, 397) / anchor,
            'cheekSize': dist(116, 345) / anchor,
            'cheekPosition': dist(168, 116) / anchor,
            'foreheadSize': dist(10, 168) / anchor,
            'lipsSize': dist(0, 17) / anchor,
            'neckThickness': dist(132, 361) / anchor,
            'nosePosition': dist(168, 1) / anchor,
            'eyeSize': (dist(159, 145) + dist(386, 374)) / (2 * anchor)
        }
        data['nosePronounced'] = abs(landmarks[1].z - landmarks[234].z) / anchor
        data['noseFlatten'] = -data['nosePronounced']
        data['chinPronounced'] = abs(landmarks[152].z - landmarks[17].z) / anchor
        data['lowCheekPronounced'] = abs(landmarks[116].z - landmarks[1].z) / anchor
        if view_type in ['left', 'right']:
            data['noseCurve'] = (landmarks[1].y - landmarks[168].y) / (landmarks[1].z - landmarks[168].z + 0.001)
        return data

    def extract_skin_color(self, img_rgb, landmarks):
        h, w, _ = img_rgb.shape
        samples = []
        for idx in [151, 117, 346]:
            lx, ly = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            if 0 <= lx < w and 0 <= ly < h: samples.append(img_rgb[ly, lx])
        return np.mean(samples, axis=0) if samples else [200, 180, 160]

    def aggregate_results(self, results):
        merged_raw = {}
        all_keys = set()
        for v in results.values(): all_keys.update(v.keys())
        for key in all_keys:
            vals = [v[key] for v in results.values() if key in v]
            merged_raw[key] = np.mean(vals)
            
        print("\nRaw Aggregated Ratios:")
        for k, v in merged_raw.items(): print(f"  {k:20s}: {v:.4f}")
            
        dna = {k: 0.5 for k in self.ranges.keys()}
        for k, v in merged_raw.items():
            if k in self.ranges:
                r_min, r_max = self.ranges[k]
                t = (v - r_min) / (r_max - r_min) if r_max != r_min else 0.5
                dna[k] = np.clip(0.5 + (t - 0.5) * self.sensitivity, 0.0, 1.0)
        dna['headSize'] = 0.5
        dna['height'] = 0.5
        return dna

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
        for k, v in payload['dna'].items(): print(f"  {k:20s}: {v:.4f}")
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.sendto(json.dumps(payload).encode('utf-8'), ("127.0.0.1", args.port))
        with open("calibration_result.json", "w") as f: json.dump(payload, f, indent=2)
        print("Calibration complete.")

if __name__ == "__main__":
    main()
