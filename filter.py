import face_recognition
import face_recognition_models
import dlib
from PIL import Image, ImageDraw, ImageFont
import argparse
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, rotate, resize
import numpy as np


class Detector:
    def __init__(self, face_location = []):
        self.face_location = face_location

    def detect(self):
        self.face_location = face_recognition.face_locations(rgb_small_frame)

    def tracker_viewer(self):
        print(face_location)


class Predictor:
    def __init__(self, face_landmark = []):
        self.face_landmark = face_landmark

    def locate(self):
        self.face_landmark = face_recognition.face_landmarks(rgb_small_frame)

    def face_part_viewer(self):
        print(face_landmark)


class Filter:
    def __init__(self, detector, predictor):

        self.detector = detector
        self.predictor = predictor

    def main(self):

        vs = VideoStream().start()
        max_width = 500
        frame = vs.read()
        frame = resize(frame, width=max_width)
        fps = vs.stream.get(cv2.CAP_PROP_FPS)
        animation_length = fps * 10
        current_animation = 0
        glasses_on = fps * 3

        cv2.namedWindow('THUG LIFE', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('THUG LIFE', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        deal = Image.open("sunglasses.png")
        text = Image.open('tlife.png')

        dealing = False

        while True:
            frame = vs.read()
            frame = resize(frame, width=max_width)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            rects = detector(img_gray, 0)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for rect in rects:
                face = {}
                shades_width = rect.right() - rect.left()


                shape = predictor(img_gray, rect)
                shape = face_utils.shape_to_np(shape)


                left_eye = shape[36:42]
                right_eye = shape[42:48]


                left_eye_center = left_eye.mean(axis=0).astype("int")
                right_eye_center = right_eye.mean(axis=0).astype("int")


                dY = left_eye_center[1] - right_eye_center[1]
                dX = left_eye_center[0] - right_eye_center[0]
                angle = np.rad2deg(np.arctan2(dY, dX))

                current_deal = deal.resize((shades_width, int(shades_width * deal.size[1] / deal.size[0])),
                               resample=Image.LANCZOS)
                current_deal = current_deal.rotate(angle, expand=True)
                current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

                face['glasses_image'] = current_deal
                left_eye_x = left_eye[0,0] - shades_width // 4
                left_eye_y = left_eye[0,1] - shades_width // 6
                face['final_pos'] = (left_eye_x, left_eye_y)



                if dealing:
                    if current_animation < glasses_on:
                        current_y = int(current_animation / glasses_on * left_eye_y)
                        img.paste(current_deal, (left_eye_x, current_y), current_deal)
                    else:
                        img.paste(current_deal, (left_eye_x, left_eye_y), current_deal)
                        img.paste(text, (75, img.height // 2 - 32), text)

            if dealing:
                current_animation += 1

                if current_animation > animation_length:
                    dealing = False
                    current_animation = 0
                else:
                    frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            cv2.imshow("deal generator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("d"):
                dealing = not dealing

        cv2.destroyAllWindows()
        vs.stop()


predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_68_point_model)

snapchat = Filter(detector, predictor)
snapchat.main()
