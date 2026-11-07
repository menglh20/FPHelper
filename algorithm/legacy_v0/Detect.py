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
    mid_mouse_x = (landmarks[0].x + landmarks[17].x) / 2
    l_mouth_distance = mid_mouse_x - l_mouse_corner_x
    r_mouth_distance = r_mouse_corner_x - mid_mouse_x
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
    l_eye_diff = abs(l_eye - l_eye_closure)
    r_eye_diff = abs(r_eye - r_eye_closure)
    eyeclosure_ratio = min(l_eye_diff, r_eye_diff) / max(l_eye_diff, r_eye_diff)
    return eyeclosure_ratio

def calc_smile_ratio(landmarks, landmarks_rest):
    l_mouth_eye_distance, r_mouth_eye_distance = calc_mouth_eye_distance(landmarks_rest)
    l_mouth_distance_smile, r_mouth_distance_smile = calc_mouth_eye_distance(landmarks)
    l_mouth_diff = abs(l_mouth_eye_distance - l_mouth_distance_smile)
    r_mouth_diff = abs(r_mouth_eye_distance - r_mouth_distance_smile)
    smile_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return smile_ratio

def calc_snarl_ratio(landmarks, landmarks_rest):
    l_alarbase_snarl, r_alarbase_snarl = calc_alarbase(landmarks)
    l_alarbase_diff = abs(l_alarbase_snarl - calc_alarbase(landmarks_rest)[0])
    r_alarbase_diff = abs(r_alarbase_snarl - calc_alarbase(landmarks_rest)[1])
    snarl_ratio = min(l_alarbase_diff, r_alarbase_diff) / max(l_alarbase_diff, r_alarbase_diff)
    return snarl_ratio

def calc_lip_pucker_ratio(landmarks, landmarks_rest):
    l_mouth_distance, r_mouth_distance = calc_mouth_distance(landmarks)
    l_mouth_diff = abs(l_mouth_distance - calc_mouth_distance(landmarks_rest)[0])
    r_mouth_diff = abs(r_mouth_distance - calc_mouth_distance(landmarks_rest)[1])
    lip_pucker_ratio = min(l_mouth_diff, r_mouth_diff) / max(l_mouth_diff, r_mouth_diff)
    return lip_pucker_ratio

def calc(images):
    detail = {
        'rest symmetry': {'eye': 0, 'cheek': 0, 'mouth': 0},
        'voluntary symmetry': {'forehead wrinkle': 0, 'eye closure': 0, 'smile': 0, 'snarl': 0, 'lip pucker': 0},
        'synkinesis': {'forehead_wrinkle': 0, 'eye_closure': 0, 'smile': 0, 'snarl': 0, 'lip_pucker': 0},
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
            
        
        ratios = {}
        for key, image in images.items():
            if key == "pic_at_rest":
                continue
            landmarks = landmarker.detect(image).face_landmarks[0]
            ratios[key] = {}
            ratios[key]["forehead_wrinkle"] = calc_eyebrow_ratio(landmarks)
            ratios[key]["eye_closure"] = calc_eyeclosure_ratio(landmarks, resting_landmarks)
            ratios[key]["smile"] = calc_smile_ratio(landmarks, resting_landmarks)
            ratios[key]["snarl"] = calc_snarl_ratio(landmarks, resting_landmarks)
            ratios[key]["lip_pucker"] = calc_lip_pucker_ratio(landmarks, resting_landmarks)
            
        print(ratios)
        
        # safety check
        for key in ratios:
            for subkey in ratios[key]:
                if ratios[key][subkey] < 0:
                    ratios[key][subkey] = 0
                if ratios[key][subkey] > 1:
                    ratios[key][subkey] = 1

        # voluntary symmetrys
        detail['voluntary symmetry']['forehead wrinkle'] = math.ceil(5 * ratios["pic_forehead_wrinkle"]["forehead_wrinkle"])
        detail['voluntary symmetry']['eye closure'] = math.ceil(5 * ratios["pic_eye_closure"]["eye_closure"])
        detail['voluntary symmetry']['smile'] = math.ceil(5 * ratios["pic_smile"]["smile"])
        detail['voluntary symmetry']['snarl'] = math.ceil(5 * ratios["pic_snarl"]["snarl"])
        detail['voluntary symmetry']['lip pucker'] = math.ceil(5 * ratios["pic_lip_pucker"]["lip_pucker"])
        
        # synkinesis
        min_ratio = min([ratios["pic_forehead_wrinkle"][key] for key in ratios["pic_forehead_wrinkle"] if key != "forehead_wrinkle"])
        min_ratio = 0.6 if min_ratio > 0.6 else min_ratio
        detail['synkinesis']['forehead_wrinkle'] = math.ceil(5 * (0.6 - min_ratio))
        
        min_ratio = min([ratios["pic_eye_closure"][key] for key in ratios["pic_eye_closure"] if key != "eye_closure"])
        min_ratio = 0.6 if min_ratio > 0.6 else min_ratio
        detail['synkinesis']['eye_closure'] = math.ceil(5 * (0.6 - min_ratio))
        
        min_ratio = min([ratios["pic_smile"][key] for key in ratios["pic_smile"] if key != "smile"])
        min_ratio = 0.6 if min_ratio > 0.6 else min_ratio
        detail['synkinesis']['smile'] = math.ceil(5 * (0.6 - min_ratio))
        
        min_ratio = min([ratios["pic_snarl"][key] for key in ratios["pic_snarl"] if key != "snarl"])
        min_ratio = 0.6 if min_ratio > 0.6 else min_ratio
        detail['synkinesis']['snarl'] = math.ceil(5 * (0.6 - min_ratio))
        
        min_ratio = min([ratios["pic_lip_pucker"][key] for key in ratios["pic_lip_pucker"] if key != "lip_pucker"])
        min_ratio = 0.6 if min_ratio > 0.6 else min_ratio
        detail['synkinesis']['lip_pucker'] = math.ceil(5 * (0.6 - min_ratio))

    return detail

if __name__ == "__main__":
    result, detail = detect("../test/pic2/")
    print(result)
    print(detail)
