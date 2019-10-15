#!/usr/bin/env python
# coding: utf-8
#get_ipython().system('git clone https://github.com/shaoanlu/face_toolbox_keras.git')
#get_ipython().run_line_magic('cd', 'face_toolbox_keras')

#get_ipython().system('gdown https://drive.google.com/uc?id=1H37LER8mRRI4q_nxpS3uQz3DcGHkTrNU')
#get_ipython().system('mv lresnet100e_ir_keras.h5 ./models/verifier/insightface/lresnet100e_ir_keras.h5')



import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt


def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1]*ratio))}, {str(int(im.shape[0]*ratio))}).")
        return cv2.resize(im, (0,0), fx=ratio, fy=ratio)
    return im


# Test images are obtained on https://www.pexels.com/
im = cv2.imread("images/test2.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
plt.imshow(im)
#plt.show()


from models.detector import face_detector


fd = face_detector.FaceAlignmentDetector(
    lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"# 2DFAN-4_keras.h5, 2DFAN-1_keras.h5
)


# ## Detect faces

bboxes = fd.detect_face(im, with_landmarks=False)


assert len(bboxes) > 0, "No face detected."

# Display detected face
x0, y0, x1, y1, score = bboxes[0] # show the first detected face
x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
print(x0,y0,x1,y1)


plt.imshow(im[x0:x1, y0:y1, :])
#plt.show()

bboxes, landmarks = fd.detect_face(im, with_landmarks=True)
#print(landmarks)

# Display landmarks
plt.figure(figsize=(15,8))
num_faces = len(bboxes)
for i in range(num_faces):
    try:
        plt.subplot(1, num_faces, i+1)
        plt.imshow(fd.draw_landmarks(im, landmarks[i], color=(0,255,0)))
        plt.imsave(f'{i}_landmark.jpg',fd.draw_landmarks(im, landmarks[i], color=(0,255,0)))
    except:
        pass
#plt.show()


from models.parser import face_parser


prs = face_parser.FaceParser()


# ## Parse without deteciton

out = prs.parse_face(im)
#print(out)


plt.imshow(out[0])
plt.imsave('im_parse.jpg',out[0])

# Show parsing result with annotations

from utils.visualize import show_parsing_with_annos
show_parsing_with_annos(out[0])

'''
im = cv2.imread("images/test2.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
plt.imshow(im)


# Set detector into FaceParser
try:
    fd
except:
    from detector import face_detector
    fd = face_detector.FaceAlignmentDetector()

prs.set_detector(fd)
# prs.remove_detector()

out = prs.parse_face(im, with_detection=True)


#plt.figure(figsize=(15,8))
num_faces = len(out)
for i in range(num_faces):
    try:
        plt.subplot(1, num_faces, i+1)
        plt.imshow(out[i])
    except:
        pass
'''

from models.detector.iris_detector import IrisDetector


# ## Detect iris, eyelibs and pulpils

im = cv2.imread("images/test5.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
plt.imshow(im)


idet = IrisDetector()

idet.set_detector(fd)

eye_lms = idet.detect_iris(im)
print(eye_lms)

# Display detection result
plt.figure(figsize=(15,10))
draw = idet.draw_pupil(im, eye_lms[0][0,...]) # draw left eye
draw = idet.draw_pupil(draw, eye_lms[0][1,...]) # draw right eye
bboxes = fd.detect_face(im, with_landmarks=False)
x0, y0, x1, y1, _ = bboxes[0].astype(np.int32)
plt.subplot(1,2,1)
plt.imshow(draw)
#plt.imsave('iris.jpg',draw)
plt.subplot(1,2,2)
plt.imshow(draw[x0:x1, y0:y1])
#plt.imsave('iris_local.jpg',draw[x0:x1, y0:y1])



'''
from models.verifier.face_verifier import FaceVerifier

im1 = cv2.imread("images/test0.jpg")[..., ::-1]
im1 = resize_image(im1) # Resize image to prevent GPU OOM.
im2 = cv2.imread("images/BO1.jpg")[..., ::-1]
im2 = resize_image(im2) # Resize image to prevent GPU OOM.
im3 = cv2.imread("images/DT.jpg")[..., ::-1]
im3 = resize_image(im3) # Resize image to prevent GPU OOM.

fv = FaceVerifier(classes=512, extractor="facenet") # extractor="insightface"

fv.set_detector(fd)

# ## Verify if two given faces are the same person

# Face verification
result1, distance1 = fv.verify(im1, im2, threshold=0.5, with_detection=True, return_distance=True)
result2, distance2 = fv.verify(im1, im3, threshold=0.5, with_detection=True, return_distance=True)


plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title(f"Source face")
plt.imshow(im1)
plt.subplot(1,3,2)
plt.title(f"Same person: {str(result1)}\n Cosine distance: {str(round(distance1, 2))}")
plt.imshow(im2)
plt.subplot(1,3,3)
plt.title(f"Same person: {str(result2)}\n Cosine distance: {str(round(distance2, 2))}")
plt.imshow(im3)


from models.estimator.gender_age_estimator import GenderAgeEstimator

im = cv2.imread("images/BO1.jpg")[..., ::-1]
im = resize_image(im) # Resize image to prevent GPU OOM.
h, w, _ = im.shape
plt.imshow(im)

gae = GenderAgeEstimator(model_type="insightface")

gae.set_detector(fd)

gender, age = gae.predict_gender_age(im, with_detection=True)

print("Gender: female") if gender == 0 else print("Gender: male")
print(f"Age: {str(age)}")
'''
