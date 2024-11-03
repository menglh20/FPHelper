import os
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
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
    running_mode=VisionRunningMode.IMAGE)


def detect(pic_folder_path, debug=False):
    try:
        pic_at_rest = mp.Image.create_from_file(pic_folder_path + "pic_at_rest.jpg")
        pic_forehead_wrinkle = mp.Image.create_from_file(pic_folder_path + "pic_forehead_wrinkle.jpg")
        pic_eye_closure = mp.Image.create_from_file(pic_folder_path + "pic_eye_closure.jpg")
        pic_smile = mp.Image.create_from_file(pic_folder_path + "pic_smile.jpg")
        pic_snarl = mp.Image.create_from_file(pic_folder_path + "pic_snarl.jpg")
        pic_lip_pucker = mp.Image.create_from_file(pic_folder_path + "pic_lip_pucker.jpg")
    except Exception as e:
        print(str(e))

    detail = calc(pic_at_rest, pic_forehead_wrinkle, pic_eye_closure, pic_smile, pic_snarl, pic_lip_pucker)

    rest_symmetry_score = 0
    voluntary_symmetry_score = 0
    synkinesis_score = 0

    rest_symmetry_score = sum(detail['rest symmetry'].values())
    voluntary_symmetry_score = sum(detail['voluntary symmetry'].values())
    synkinesis_score = sum(detail['synkinesis'].values())

    result = 4 * voluntary_symmetry_score - 5 * \
        rest_symmetry_score - synkinesis_score

    return result, detail


def calc_eye_size(landmarks):
    l_up_eye_y = (landmarks[158].y + landmarks[159].y + landmarks[160].y) / 3
    l_down_eye_y = (landmarks[144].y + landmarks[145].y + landmarks[153].y) / 3
    r_up_eye_y = (landmarks[385].y + landmarks[386].y + landmarks[387].y) / 3
    r_down_eye_y = (landmarks[373].y + landmarks[374].y + landmarks[380].y) / 3
    l_eye = max(l_down_eye_y - l_up_eye_y, 0)
    r_eye = max(r_down_eye_y - r_up_eye_y, 0)
    return l_eye, r_eye


def calc_mouth_distance(landmarks):
    l_mouse_corner_x = landmarks[61].x
    r_mouse_corner_x = landmarks[291].x
    l_face_x = (landmarks[162].x + landmarks[127].x + landmarks[234].x + landmarks[93].x) / 4
    r_face_x = (landmarks[389].x + landmarks[356].x + landmarks[454].x + landmarks[323].x) / 4
    l_mouth_distance = l_mouse_corner_x - l_face_x
    r_mouth_distance = r_face_x - r_mouse_corner_x
    return l_mouth_distance, r_mouth_distance


def calc_mouth_eye_distance(landmarks):
    l_mouse_corner = (landmarks[61].x, landmarks[61].y)
    r_mouse_corner = (landmarks[291].x, landmarks[291].y)
    l_eye_center = (landmarks[468].x, landmarks[468].y)
    r_eye_center = (landmarks[473].x, landmarks[473].y)
    l_mouth_eye_distance = math.sqrt((l_mouse_corner[0] - l_eye_center[0]) ** 2 + (l_mouse_corner[1] - l_eye_center[1]) ** 2)
    r_mouth_eye_distance = math.sqrt((r_mouse_corner[0] - r_eye_center[0]) ** 2 + (r_mouse_corner[1] - r_eye_center[1]) ** 2)
    return l_mouth_eye_distance, r_mouth_eye_distance


def calc_eyebrow_eye_distance(landmarks):
    l_eyebrow_y = (landmarks[105].y + landmarks[66].y + landmarks[52].y + landmarks[65].y) / 4
    l_eye_center_y = landmarks[468].y
    r_eyebrow_y = (landmarks[296].y + landmarks[334].y + landmarks[295].y + landmarks[282].y) / 4
    r_eye_center_y = landmarks[473].y
    l_eyebrow_eye_distance = l_eye_center_y - l_eyebrow_y
    r_eyebrow_eye_distance = r_eye_center_y - r_eyebrow_y
    return l_eyebrow_eye_distance, r_eyebrow_eye_distance


def calc_alarbase(landmarks):
    mid_alarbase_x = landmarks[1].x
    l_alarbase_x = landmarks[219].x
    r_alarbase_x = landmarks[439].x
    l_snarl_diff = mid_alarbase_x - l_alarbase_x
    r_snarl_diff = r_alarbase_x - mid_alarbase_x
    return l_snarl_diff, r_snarl_diff


def calc(pic_at_rest, pic_forehead_wrinkle, pic_eye_closure, pic_smile, pic_snarl, pic_lip_pucker):
    detail = {
        'rest symmetry': {
            'eye': 0,
            'cheek': 0,
            'mouth': 0,
        },
        'voluntary symmetry': {
            'forehead wrinkle': 0,
            'eye closure': 0,
            'smile': 0,
            'snarl': 0,
            'lip pucker': 0,
        },
        'synkinesis': {
            'forehead wrinkle': 0,
            'eye closure': 0,
            'smile': 0,
            'snarl': 0,
            'lip pucker': 0,
        },
    }
    
    with FaceLandmarker.create_from_options(options) as landmarker:
        ### rest symmetry ###
        face_landmarker_result = landmarker.detect(pic_at_rest)
        resting_landmarks = face_landmarker_result.face_landmarks[0]
        # eye symmetry
        l_eye, r_eye = calc_eye_size(resting_landmarks)
        eye_ratio = l_eye / r_eye if l_eye > r_eye else r_eye / l_eye
        print('eye_ratio:', eye_ratio)
        if eye_ratio > 1.1:
            detail['rest symmetry']['eye'] = 1
        # cheek symmetry
        # TODO
        # mouth symmetry
        l_mouth_eye_distance, r_mouth_eye_distance = calc_mouth_eye_distance(resting_landmarks)
        mouth_ratio = l_mouth_eye_distance / r_mouth_eye_distance if l_mouth_eye_distance > r_mouth_eye_distance else r_mouth_eye_distance / l_mouth_eye_distance
        print('mouth_ratio:', mouth_ratio)
        if mouth_ratio > 1.1:
            detail['rest symmetry']['mouth'] = 1
            
        ### voluntary symmetry ###
        # forehead wrinkle
        face_landmarker_result = landmarker.detect(pic_forehead_wrinkle)
        forehead_wrinkle_landmarks = face_landmarker_result.face_landmarks[0]
        l_eyebrow_eye_distance, r_eyebrow_eye_distance = calc_eyebrow_eye_distance(forehead_wrinkle_landmarks)
        eyebrow_ratio = l_eyebrow_eye_distance / r_eyebrow_eye_distance if l_eyebrow_eye_distance < r_eyebrow_eye_distance else r_eyebrow_eye_distance / l_eyebrow_eye_distance
        print('eyebrow_ratio:', eyebrow_ratio)
        if eyebrow_ratio > 0.8:
            detail['voluntary symmetry']['forehead wrinkle'] = 5
        elif eyebrow_ratio > 0.6:
            detail['voluntary symmetry']['forehead wrinkle'] = 4
        elif eyebrow_ratio > 0.4:
            detail['voluntary symmetry']['forehead wrinkle'] = 3
        elif eyebrow_ratio > 0.2:
            detail['voluntary symmetry']['forehead wrinkle'] = 2
        else:
            detail['voluntary symmetry']['forehead wrinkle'] = 1
        # eye closure
        face_landmarker_result = landmarker.detect(pic_eye_closure)
        eye_closure_landmarks = face_landmarker_result.face_landmarks[0]
        l_eye_closure, r_eye_closure = calc_eye_size(eye_closure_landmarks)
        l_eye_diff = l_eye - l_eye_closure
        r_eye_diff = r_eye - r_eye_closure
        eyeclosure_ratio = l_eye_diff / r_eye_diff if l_eye_diff < r_eye_diff else r_eye_diff / l_eye_diff
        print('eyeclosure_ratio:', eyeclosure_ratio)
        if eyeclosure_ratio > 0.8:
            detail['voluntary symmetry']['eye closure'] = 5
        elif eyeclosure_ratio > 0.6:
            detail['voluntary symmetry']['eye closure'] = 4
        elif eyeclosure_ratio > 0.4:
            detail['voluntary symmetry']['eye closure'] = 3
        elif eyeclosure_ratio > 0.2:
            detail['voluntary symmetry']['eye closure'] = 2
        else:
            detail['voluntary symmetry']['eye closure'] = 1
        # smile
        face_landmarker_result = landmarker.detect(pic_smile)
        smile_landmarks = face_landmarker_result.face_landmarks[0]
        l_mouth_distance_smile, r_mouth_distance_smile = calc_mouth_eye_distance(smile_landmarks)
        l_mouth_distance_resting, r_mouth_distance_resting = calc_mouth_eye_distance(resting_landmarks)
        l_mouth_diff = l_mouth_distance_resting - l_mouth_distance_smile
        r_mouth_diff = r_mouth_distance_resting - r_mouth_distance_smile
        smile_ratio = l_mouth_diff / r_mouth_diff if l_mouth_diff < r_mouth_diff else r_mouth_diff / l_mouth_diff
        print('smile_ratio:', smile_ratio)
        if smile_ratio > 0.8:
            detail['voluntary symmetry']['smile'] = 5
        elif smile_ratio > 0.6:
            detail['voluntary symmetry']['smile'] = 4
        elif smile_ratio > 0.4:
            detail['voluntary symmetry']['smile'] = 3
        elif smile_ratio > 0.2:
            detail['voluntary symmetry']['smile'] = 2
        else:
            detail['voluntary symmetry']['smile'] = 1
        # snarl
        face_landmarker_result = landmarker.detect(pic_snarl)
        snarl_landmarks = face_landmarker_result.face_landmarks[0]
        l_alarbase_snarl, r_alarbase_snarl = calc_alarbase(snarl_landmarks)
        l_alarbase_resting, r_alarbase_resting = calc_alarbase(resting_landmarks)
        l_alarbase_diff = l_alarbase_snarl - l_alarbase_resting
        r_alarbase_diff = r_alarbase_snarl - r_alarbase_resting
        print('l_alarbase_diff:', l_alarbase_diff)
        print('r_alarbase_diff:', r_alarbase_diff)
        snarl_ratio = l_alarbase_diff / r_alarbase_diff if l_alarbase_diff < r_alarbase_diff else r_alarbase_diff / l_alarbase_diff
        print('snarl_ratio:', snarl_ratio)
        if snarl_ratio > 0.8:
            detail['voluntary symmetry']['snarl'] = 5
        elif snarl_ratio > 0.6:
            detail['voluntary symmetry']['snarl'] = 4
        elif snarl_ratio > 0.4:
            detail['voluntary symmetry']['snarl'] = 3
        elif snarl_ratio > 0.2:
            detail['voluntary symmetry']['snarl'] = 2
        else:
            detail['voluntary symmetry']['snarl'] = 1
        # lip pucker
    return detail


if __name__ == "__main__":
    result, detail = detect("../test/pic/")
    print(result)
    print(detail)