import cv2
import joblib
import mediapipe as mp
import numpy as np
import face_recognition
import os


class Action():
    def __init__(self, video_url):
        self.pose_knn = joblib.load('Model/PoseKeypoint.joblib')
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.prevTime = 0
        self.keyXYZ = [
            "nose_x",
            "nose_y",
            "nose_z",
            "left_eye_inner_x",
            "left_eye_inner_y",
            "left_eye_inner_z",
            "left_eye_x",
            "left_eye_y",
            "left_eye_z",
            "left_eye_outer_x",
            "left_eye_outer_y",
            "left_eye_outer_z",
            "right_eye_inner_x",
            "right_eye_inner_y",
            "right_eye_inner_z",
            "right_eye_x",
            "right_eye_y",
            "right_eye_z",
            "right_eye_outer_x",
            "right_eye_outer_y",
            "right_eye_outer_z",
            "left_ear_x",
            "left_ear_y",
            "left_ear_z",
            "right_ear_x",
            "right_ear_y",
            "right_ear_z",
            "mouth_left_x",
            "mouth_left_y",
            "mouth_left_z",
            "mouth_right_x",
            "mouth_right_y",
            "mouth_right_z",
            "left_shoulder_x",
            "left_shoulder_y",
            "left_shoulder_z",
            "right_shoulder_x",
            "right_shoulder_y",
            "right_shoulder_z",
            "left_elbow_x",
            "left_elbow_y",
            "left_elbow_z",
            "right_elbow_x",
            "right_elbow_y",
            "right_elbow_z",
            "left_wrist_x",
            "left_wrist_y",
            "left_wrist_z",
            "right_wrist_x",
            "right_wrist_y",
            "right_wrist_z",
            "left_pinky_x",
            "left_pinky_y",
            "left_pinky_z",
            "right_pinky_x",
            "right_pinky_y",
            "right_pinky_z",
            "left_index_x",
            "left_index_y",
            "left_index_z",
            "right_index_x",
            "right_index_y",
            "right_index_z",
            "left_thumb_x",
            "left_thumb_y",
            "left_thumb_z",
            "right_thumb_x",
            "right_thumb_y",
            "right_thumb_z",
            "left_hip_x",
            "left_hip_y",
            "left_hip_z",
            "right_hip_x",
            "right_hip_y",
            "right_hip_z",
            "left_knee_x",
            "left_knee_y",
            "left_knee_z",
            "right_knee_x",
            "right_knee_y",
            "right_knee_z",
            "left_ankle_x",
            "left_ankle_y",
            "left_ankle_z",
            "right_ankle_x",
            "right_ankle_y",
            "right_ankle_z",
            "left_heel_x",
            "left_heel_y",
            "left_heel_z",
            "right_heel_x",
            "right_heel_y",
            "right_heel_z",
            "left_foot_index_x",
            "left_foot_index_y",
            "left_foot_index_z",
            "right_foot_index_x",
            "right_foot_index_y",
            "right_foot_index_z"
        ]
        self.cap = cv2.VideoCapture(video_url)
        self.total_image_name = []
        self.total_face_encoding = []

        self.path = 'img'
        self.name = 'Unknown'
        self.add()

    def detect_action(self):
        res_point = []
        with self.mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                success, image = self.cap.read()
                frame = image.copy()
                image1 = image.copy()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks:
                    for index, landmarks in enumerate(results.pose_landmarks.landmark):
                        res_point.append(landmarks.x)
                        res_point.append(landmarks.y)
                        res_point.append(landmarks.z)
                    shape1 = int(len(res_point) / len(self.keyXYZ))
                    res_point = np.array(res_point).reshape(shape1, len(self.keyXYZ))
                    pred = self.pose_knn.predict(res_point)
                    res_point = []
                    if pred == 0:
                        cv2.putText(frame, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 1)
                    else:
                        cv2.putText(frame, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 1)
                # Draw the pose annotation on the image.
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                #描绘人体关键点
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                face_locations = face_recognition.face_locations(image1)
                face_encodings = face_recognition.face_encodings(image1, face_locations)
                # 在这个视频帧中循环遍历每个人脸
                for (top, right, bottom, left), face_encoding in zip(
                        face_locations, face_encodings):
                    # 看看面部是否与已知人脸相匹配。
                    for i, v in enumerate(self.total_face_encoding):
                        match = face_recognition.compare_faces(
                            [v], face_encoding, tolerance=0.5)
                        self.name = "Unknown"
                        if match[0]:
                            self.name = self.total_image_name[i]
                            break
                    # 画出一个框，框住脸
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # 画出一个带名字的标签，放在框下
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),
                                  cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, self.name, (left + 6, bottom - 6), font, 1.0,
                                (255, 255, 255), 1)

                cv2.imshow('result', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        self.cap.release()
        cv2.destroyAllWindows()

    def add(self):
        for fn in os.listdir(self.path):  # fn 表示的是文件名q
            print(self.path + "/" + fn)
            self.total_face_encoding.append(
                face_recognition.face_encodings(
                    face_recognition.load_image_file(self.path + "/" + fn))[0])
            fn = fn[:(len(fn) - 4)]  # 截取图片名（这里应该把images文件中的图片名命名为为人物名）
            self.total_image_name.append(fn)  # 图片名字列表


a = Action(video_url=0)
a.detect_action()
