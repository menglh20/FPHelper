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

def calc_eyebrow_distance(landmarks):
    l_eyebrow_y = np.mean([landmarks[i].y for i in [105, 66, 52, 65]])
    l_eye_center_y = landmarks[468].y
    r_eyebrow_y = np.mean([landmarks[i].y for i in [296, 334, 295, 282]])
    r_eye_center_y = landmarks[473].y
    l_eyebrow_distance = l_eye_center_y - l_eyebrow_y
    r_eyebrow_distance = r_eye_center_y - r_eyebrow_y
    return l_eyebrow_distance, r_eyebrow_distance

def calc_alarbase(landmarks):
    mid_alarbase_x = landmarks[1].x
    l_alarbase_x = landmarks[48].x
    r_alarbase_x = landmarks[278].x
    l_alarbase = mid_alarbase_x - l_alarbase_x
    r_alarbase = r_alarbase_x - mid_alarbase_x
    return l_alarbase, r_alarbase

def recognize(video_path, debug=False):
    results = {
        'forehead_wrinkle': [],
        'eye_closure': [],
        'smile': [],
        'snarl': [],
        'lip_pucker': []
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
        while cap.isOpened():
            ret, frame = cap.read()

            # 检查是否成功读取帧
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, math.floor(n * 1000 / fps))
            face_landmarks = face_landmarker_result.face_landmarks[0]
            
            eyebrow_distance = (calc_eyebrow_distance(face_landmarks)[0] + calc_eyebrow_distance(face_landmarks)[1]) / 2
            results['forehead_wrinkle'].append(eyebrow_distance)
            
            eyesize = (calc_eye_size(face_landmarks)[0] + calc_eye_size(face_landmarks)[1]) / 2
            results['eye_closure'].append(eyesize)
            
            mouth_eye_distance = (calc_mouth_eye_distance(face_landmarks)[0] + calc_mouth_eye_distance(face_landmarks)[1]) / 2
            results['smile'].append(mouth_eye_distance)
            
            alarbase = (calc_alarbase(face_landmarks)[0] + calc_alarbase(face_landmarks)[1]) / 2
            results['snarl'].append(alarbase)
            
            mouth_distance = (calc_mouth_distance(face_landmarks)[0] + calc_mouth_distance(face_landmarks)[1]) / 2
            results['lip_pucker'].append(mouth_distance)
            
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
    # Cut according image
    output_path = "../test/pic/"
    os.makedirs(output_path, exist_ok=True)
    
    index_forehead_wrinkle = np.argmax(results['forehead_wrinkle'])
    index_eye_closure = np.argmin(results['eye_closure'])
    index_smile = np.argmin(results['smile'])
    index_snarl = np.argmax(results['snarl'])
    index_lip_pucker = np.argmax(results['lip_pucker'])
    print("forehead_wrinkle: ", index_forehead_wrinkle)
    print("eye_closure: ", index_eye_closure)
    print("smile: ", index_smile)
    print("snarl: ", index_snarl)
    print("lip_pucker: ", index_lip_pucker)
    
    cap = cv2.VideoCapture(video_path)
    for i in range(total_frames):
        ret, frame = cap.read()
        if i == index_forehead_wrinkle:
            cv2.imwrite(output_path + "pic_forehead_wrinkle.jpg", frame)
        if i == index_eye_closure:
            cv2.imwrite(output_path + "pic_eye_closure.jpg", frame)
        if i == index_smile:
            cv2.imwrite(output_path + "pic_smile.jpg", frame)
        if i == index_snarl:
            cv2.imwrite(output_path + "pic_snarl.jpg", frame)
        if i == index_lip_pucker:
            cv2.imwrite(output_path + "pic_lip_pucker.jpg", frame)
        if i == 10:
            cv2.imwrite(output_path + "pic_at_rest.jpg", frame)



if __name__ == "__main__":
    recognize("../test/video.mp4", debug=True)