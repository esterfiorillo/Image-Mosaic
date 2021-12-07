import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage import color

def autocrop(image, threshold=2):

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[:, rows[0]: rows[-1] + 1]
    else:
        image = image[:, :1]

    return image


def detectAndDescribe_SIFT(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	descriptor = cv2.xfeatures2d.SIFT_create()
	(kps, features) = descriptor.detectAndCompute(image, None)

	kps = np.float32([kp.pt for kp in kps])

	return (kps, features)


def detectAndDescribe_ORB(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()
	kps = orb.detect(image,None)
	kps, features = orb.compute(image, kps)
	kps = np.float32([kp.pt for kp in kps])

	return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB):
	# Computar matches
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	matches = []
	# iterar sobre os matches
	for m in rawMatches:
		if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
			matches.append((m[0].trainIdx, m[0].queryIdx))
	# para computar a homografia é necessário pelo menos 4 matches
	if len(matches) > 4:
		# construir dois sets de pontos
		ptsA = np.float32([kpsA[i] for (_, i) in matches])
		ptsB = np.float32([kpsB[i] for (i, _) in matches])
		# calcular homografia entre os dois sets
		(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
			4.0)
		return (matches, H, status)
	#se não houver mais de 4 pontos para calcular a matriz de homografia
	return None


def mosaico(images):
	(imageB, imageA) = images
	(kpsA, featuresA) = detectAndDescribe_SIFT(imageA)
	(kpsB, featuresB) = detectAndDescribe_SIFT(imageB)

	M = matchKeypoints(kpsA, kpsB, featuresA, featuresB)
	if M is None:
		return None

	(matches, H, status) = M
	result = cv2.warpPerspective(imageA, H,
		(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
	result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

	return result