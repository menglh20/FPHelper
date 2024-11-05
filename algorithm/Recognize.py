import os
import mediapipe as mp
import numpy as np
import cv2
import math
import time
import matplotlib.pyplot as plt

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

MINIMAL_CHANGE = 0.001

def calc_eye_size(landmarks):
    l_up_eye_y = np.mean([landmarks[i].y for i in [158, 159, 160]])
    l_down_eye_y = np.mean([landmarks[i].y for i in [144, 145, 153]])
    r_up_eye_y = np.mean([landmarks[i].y for i in [385, 386, 387]])
    r_down_eye_y = np.mean([landmarks[i].y for i in [373, 374, 380]])
    l_eye = max(l_down_eye_y - l_up_eye_y, 0)
    r_eye = max(r_down_eye_y - r_up_eye_y, 0)
    return l_eye, r_eye

def calc_mouth_distance(landmarks):
    l_mouse_corner_x = landmarks[61].x
    r_mouse_corner_x = landmarks[291].x
    l_face_x = np.mean([landmarks[i].x for i in [162, 127, 234, 93]])
    r_face_x = np.mean([landmarks[i].x for i in [389, 356, 454, 323]])
    l_mouth_distance = l_mouse_corner_x - l_face_x
    r_mouth_distance = r_face_x - r_mouse_corner_x
    return l_mouth_distance, r_mouth_distance

def calc_mouth_eye_distance(landmarks):
    l_mouse_corner = (landmarks[61].x, landmarks[61].y)
    r_mouse_corner = (landmarks[291].x, landmarks[291].y)
    l_eye_center = (landmarks[468].x, landmarks[468].y)
    r_eye_center = (landmarks[473].x, landmarks[473].y)
    l_mouth_eye_distance = math.dist(l_mouse_corner, l_eye_center)
    r_mouth_eye_distance = math.dist(r_mouse_corner, r_eye_center)
    return l_mouth_eye_distance, r_mouth_eye_distance

def calc_eyebrow_eye_distance(landmarks):
    l_eyebrow_y = np.mean([landmarks[i].y for i in [105, 66, 52, 65]])
    l_eye_center_y = landmarks[468].y
    r_eyebrow_y = np.mean([landmarks[i].y for i in [296, 334, 295, 282]])
    r_eye_center_y = landmarks[473].y
    l_eyebrow_eye_distance = l_eye_center_y - l_eyebrow_y
    r_eyebrow_eye_distance = r_eye_center_y - r_eyebrow_y
    return l_eyebrow_eye_distance, r_eyebrow_eye_distance

def calc_alarbase(landmarks):
    mid_alarbase_x = landmarks[1].x
    l_alarbase_x = landmarks[48].x
    r_alarbase_x = landmarks[278].x
    l_snarl_diff = mid_alarbase_x - l_alarbase_x
    r_snarl_diff = r_alarbase_x - mid_alarbase_x
    return l_snarl_diff, r_snarl_diff

def calc_eyebrow_ratio(landmarks):
    l_eyebrow_eye_distance, r_eyebrow_eye_distance = calc_eyebrow_eye_distance(landmarks)
    eyebrow_ratio = min(l_eyebrow_eye_distance, r_eyebrow_eye_distance) / max(l_eyebrow_eye_distance, r_eyebrow_eye_distance)
    return eyebrow_ratio

def calc_eyeclosure_ratio(landmarks, landmarks_rest):
    l_eye, r_eye = calc_eye_size(landmarks_rest)
    l_eye_closure, r_eye_closure = calc_eye_size(landmarks)
    l_eye_diff = max(l_eye - l_eye_closure, 0)
    r_eye_diff = max(r_eye - r_eye_closure, 0)
    if l_eye_diff < MINIMAL_CHANGE and r_eye_diff < MINIMAL_CHANGE:
        return 1
    eyeclosure_ratio = min(l_eye_diff, r_eye_diff) / max(l_eye_diff, r_eye_diff)
    return eyeclosure_ratio

def calc_smile_ratio(landmarks, landmarks_rest):
    l_mouth_eye_distance, r_mouth_eye_distance = calc_mouth_eye_distance(landmarks_rest)
    l_mouth_distance_smile, r_mouth_distance_smile = calc_mouth_eye_distance(landmarks)
    l_mouth_diff = max(l_mouth_eye_distance - l_mouth_distance_smile, 0)
    r_mouth_diff = max(r_mouth_eye_distance - r_mouth_distance_smile, 0)
    if l_mouth_diff < MINIMAL_CHANGE and r_mouth_diff < MINIMAL_CHANGE:
        return 1
    smile_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return smile_ratio

def calc_snarl_ratio(landmarks, landmarks_rest):
    l_alarbase_snarl, r_alarbase_snarl = calc_alarbase(landmarks)
    l_alarbase_diff = max(l_alarbase_snarl - calc_alarbase(landmarks_rest)[0], 0)
    r_alarbase_diff = max(r_alarbase_snarl - calc_alarbase(landmarks_rest)[1], 0)
    if l_alarbase_diff < MINIMAL_CHANGE and r_alarbase_diff < MINIMAL_CHANGE:
        return 1
    snarl_ratio = min(l_alarbase_diff, r_alarbase_diff) / max(l_alarbase_diff, r_alarbase_diff)
    return snarl_ratio

def calc_lip_pucker_ratio(landmarks, landmarks_rest):
    l_mouth_distance, r_mouth_distance = calc_mouth_distance(landmarks)
    l_mouth_diff = max(l_mouth_distance - calc_mouth_distance(landmarks_rest)[0], 0)
    r_mouth_diff = max(r_mouth_distance - calc_mouth_distance(landmarks_rest)[1], 0)
    if l_mouth_diff < MINIMAL_CHANGE and r_mouth_diff < MINIMAL_CHANGE:
        return 1
    lip_pucker_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return lip_pucker_ratio
    
def detect(video_path, debug=False):
    results = {
        'eyebrow_ratio': [],
        'eyeclosure_ratio': [],
        'smile_ratio': [],
        'snarl_ratio': [],
        'lip_pucker_ratio': []
    }
    with FaceLandmarker.create_from_options(options) as landmarker:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 获取视频的帧率和总帧数
        fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        print("fps: ", fps)
        print("total_frames: ", total_frames)
        print("resolution: ", w, h)
        
        n = 1
        resting_landmarks = None
        while cap.isOpened():
            ret, frame = cap.read()

            # 检查是否成功读取帧
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, math.floor(n * 1000 / fps))
            face_landmarks = face_landmarker_result.face_landmarks[0]
            
            if n <= 10:
                resting_landmarks = face_landmarks
            
            eyebrow_ration = calc_eyebrow_ratio(face_landmarks)
            results['eyebrow_ratio'].append(eyebrow_ration)
            
            eyeclosure_ratio = calc_eyeclosure_ratio(face_landmarks, resting_landmarks)
            results['eyeclosure_ratio'].append(eyeclosure_ratio)
            
            smile_ratio = calc_smile_ratio(face_landmarks, resting_landmarks)
            results['smile_ratio'].append(smile_ratio)
            
            snarl_ratio = calc_snarl_ratio(face_landmarks, resting_landmarks)
            results['snarl_ratio'].append(snarl_ratio)
            
            lip_pucker_ratio = calc_lip_pucker_ratio(face_landmarks, resting_landmarks)
            results['lip_pucker_ratio'].append(lip_pucker_ratio)
            
            n += 1
    # draw ratio changes
    if debug:
        for key in results:
            plt.figure(figsize=(20, 10))
            plt.yticks(np.arange(0, 1, 0.1))
            plt.plot(results[key], label=key)
            plt.legend()
            plt.savefig("../test/" + key + ".png")
            plt.close()



if __name__ == "__main__":
    detect("../test/video.mp4", debug=True)