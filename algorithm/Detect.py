import os
import mediapipe as mp
import numpy as np
import cv2
import math
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

def load_images(pic_folder_path):
    try:
        images = {
            "pic_at_rest": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_at_rest.jpg")),
            "pic_forehead_wrinkle": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_forehead_wrinkle.jpg")),
            "pic_eye_closure": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_eye_closure.jpg")),
            "pic_smile": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_smile.jpg")),
            "pic_snarl": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_snarl.jpg")),
            "pic_lip_pucker": mp.Image.create_from_file(os.path.join(pic_folder_path, "pic_lip_pucker.jpg"))
        }
        return images
    except Exception as e:
        print(str(e))
        return None

def detect(pic_folder_path, debug=False):
    images = load_images(pic_folder_path)
    if not images:
        return None, None

    detail = calc(images)

    rest_symmetry_score = sum(detail['rest symmetry'].values())
    voluntary_symmetry_score = sum(detail['voluntary symmetry'].values())
    synkinesis_score = sum(detail['synkinesis'].values())

    result = 4 * voluntary_symmetry_score - 5 * rest_symmetry_score - synkinesis_score

    return result, detail


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
    l_eye_diff = l_eye - l_eye_closure
    r_eye_diff = r_eye - r_eye_closure
    eyeclosure_ratio = min(l_eye_diff, r_eye_diff) / max(l_eye_diff, r_eye_diff)
    return eyeclosure_ratio

def calc_smile_ratio(landmarks, landmarks_rest):
    l_mouth_eye_distance, r_mouth_eye_distance = calc_mouth_eye_distance(landmarks_rest)
    l_mouth_distance_smile, r_mouth_distance_smile = calc_mouth_eye_distance(landmarks)
    l_mouth_diff = l_mouth_eye_distance - l_mouth_distance_smile
    r_mouth_diff = r_mouth_eye_distance - r_mouth_distance_smile
    smile_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return smile_ratio

def calc_snarl_ratio(landmarks, landmarks_rest):
    l_alarbase_snarl, r_alarbase_snarl = calc_alarbase(landmarks)
    l_alarbase_diff = l_alarbase_snarl - calc_alarbase(landmarks_rest)[0]
    r_alarbase_diff = r_alarbase_snarl - calc_alarbase(landmarks_rest)[1]
    snarl_ratio = min(l_alarbase_diff, r_alarbase_diff) / max(l_alarbase_diff, r_alarbase_diff)
    return snarl_ratio

def calc_lip_pucker_ratio(landmarks, landmarks_rest):
    l_mouth_distance, r_mouth_distance = calc_mouth_distance(landmarks)
    l_mouth_diff = l_mouth_distance - calc_mouth_distance(landmarks_rest)[0]
    r_mouth_diff = r_mouth_distance - calc_mouth_distance(landmarks_rest)[1]
    lip_pucker_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return lip_pucker_ratio

def calc(images):
    detail = {
        'rest symmetry': {'eye': 0, 'cheek': 0, 'mouth': 0},
        'voluntary symmetry': {'forehead wrinkle': 0, 'eye closure': 0, 'smile': 0, 'snarl': 0, 'lip pucker': 0},
        'synkinesis': {'forehead wrinkle': 0, 'eye closure': 0, 'smile': 0, 'snarl': 0, 'lip pucker': 0},
    }

    with FaceLandmarker.create_from_options(options) as landmarker:
        # rest symmetrys
        resting_landmarks = landmarker.detect(images["pic_at_rest"]).face_landmarks[0]
        l_eye, r_eye = calc_eye_size(resting_landmarks)
        eye_ratio = max(l_eye, r_eye) / min(l_eye, r_eye)
        if eye_ratio > 1.1:
            detail['rest symmetry']['eye'] = 1

        l_mouth_eye_distance, r_mouth_eye_distance = calc_mouth_eye_distance(resting_landmarks)
        mouth_ratio = max(l_mouth_eye_distance, r_mouth_eye_distance) / min(l_mouth_eye_distance, r_mouth_eye_distance)
        if mouth_ratio > 1.1:
            detail['rest symmetry']['mouth'] = 1

        # voluntary symmetryss
        for key, image in images.items():
            if key == "pic_at_rest":
                continue
            landmarks = landmarker.detect(image).face_landmarks[0]
            if key == "pic_forehead_wrinkle":
                eyebrow_ratio = calc_eyebrow_ratio(landmarks)
                detail['voluntary symmetry']['forehead wrinkle'] = int(5 * eyebrow_ratio)
            elif key == "pic_eye_closure":
                eyeclosure_ratio = calc_eyeclosure_ratio(landmarks, resting_landmarks)
                detail['voluntary symmetry']['eye closure'] = int(5 * eyeclosure_ratio)
            elif key == "pic_smile":
                smile_ratio = calc_smile_ratio(landmarks, resting_landmarks)
                detail['voluntary symmetry']['smile'] = int(5 * smile_ratio)
            elif key == "pic_snarl":
                snarl_ratio = calc_snarl_ratio(landmarks, resting_landmarks)
                detail['voluntary symmetry']['snarl'] = int(5 * snarl_ratio)
            elif key == "pic_lip_pucker":
                lip_pucker_ratio = calc_lip_pucker_ratio(landmarks, resting_landmarks)
                detail['voluntary symmetry']['lip pucker'] = int(5 * lip_pucker_ratio)
        
        # synkinesis
        for key, image in images.items():
            if key == "pic_at_rest":
                continue
            landmarks = landmarker.detect(image).face_landmarks[0]
            ratios = {
                "forehead wrinkle": calc_eyebrow_ratio(landmarks),
                "eye closure": calc_eyeclosure_ratio(landmarks, resting_landmarks),
                "smile": calc_smile_ratio(landmarks, resting_landmarks),
                "snarl": calc_snarl_ratio(landmarks, resting_landmarks),
                "lip pucker": calc_lip_pucker_ratio(landmarks, resting_landmarks)
            }
            other_ratios = [ratio for k, ratio in ratios.items() if k != key.split('_')[1]]
            min_ratio = min(other_ratios)
            detail['synkinesis'][key.split('_')[1]] = int(3 * (1 - min_ratio))

    return detail

if __name__ == "__main__":
    result, detail = detect("../test/pic/")
    print(result)
    print(detail)
