import os
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math
import time


def detect(folder_path, debug=False):
    pic_at_rest = cv2.imread(folder_path + "pic_at_rest.jpg")
    pic_forehead_wrinkle = cv2.imread(folder_path + "pic_forehead_wrinkle.jpg")
    pic_eye_closure = cv2.imread(folder_path + "pic_eye_closure.jpg")
    pic_smile = cv2.imread(folder_path + "pic_smile.jpg")
    pic_snarl = cv2.imread(folder_path + "pic_snarl.jpg")
    pic_lip_pucker = cv2.imread(folder_path + "pic_lip_pucker.jpg")

    detail = calc(pic_at_rest, pic_forehead_wrinkle, pic_eye_closure, pic_smile, pic_snarl, pic_lip_pucker)

    rest_symmetry_score = 0
    voluntary_symmetry_score = 0
    synkinesis_score = 0

    rest_symmetry_score = sum(detail['rest symmetry'].values())
    voluntary_symmetry_score = sum(detail['voluntary symmetry'].values())
    synkinesis_score = sum(detail['synkinesis'].values())

    result = 4 * voluntary_symmetry_score - 5 * \
        rest_symmetry_score - synkinesis_score

    return result, str(detail)


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
    return detail
