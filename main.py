import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread

# Prompt user to select sign language (ASL or ISL)
print("What sign Language do you want to use? 1. ASL, 2. ISL")
inp = int(input())

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Set TensorFlow log level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the corresponding trained model based on the user input
if inp == 1:
    model = load_model('cnn_model_asl.h5')
else:
    model = load_model('cnn_model_isl.h5')


# Function to load hand histogram from file
def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist


# Define dimensions for image processing
image_x, image_y = 400, 400

# Function to preprocess image for Keras model
def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

# Function to predict the sign class using Keras model
def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

# Function to get the predicted text based on the predicted class
def get_pred_text_from_db(inp, pred_class):
	if inp == 1:
		conn = sqlite3.connect("gesture_db_asl.db")
	else:
		conn = sqlite3.connect("gesture_db_isl.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

# Function to get the predicted sign from contour of hand region
def get_pred_from_contour(inp, contour, thresh):
	# Get the bounding rectangle around the contour
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	# Crop the thresholded image around the bounding rectangle
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	# Make the image square by adding borders around the shorter side
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	# Predict the sign class and get the predicted text if the probability is high enough
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 50:
		print(pred_probab * 100)
		text = get_pred_text_from_db(inp, pred_class)
	return text


# Load the hand histogram from file
hist = get_hand_hist()

# Define the region of interest for image
# Set the values for x, y, w, h
# These values determine the region of interest for hand detection
x, y, w, h = 300, 100, 300, 300

# Set a boolean to determine whether to use text-to-speech or not
is_voice_on = True


# This function takes an image and the input mode and returns the image, contours, and thresholded image
def get_img_contour_thresh(img, inp):
	# Flip the image horizontally for mirror effect
	img = cv2.flip(img, 1)

	# Convert the image to the HSV color space
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Use the histogram to calculate the back projection of the image
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

	# Create an elliptical structuring element for morphological operations
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

	# Apply a morphological filter to the back projection image
	cv2.filter2D(dst, -1, disc, dst)

	# Apply a Gaussian blur to the image
	blur = cv2.GaussianBlur(dst, (11, 11), 0)

	# Apply a median blur to the image
	blur = cv2.medianBlur(blur, 15)

	# Threshold the image using OTSU's method
	thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# Merge the thresholded image with itself three times to create a three-channel image
	thresh = cv2.merge((thresh, thresh, thresh))

	# Convert the image to grayscale
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

	# Crop the image to the region of interest
	thresh = thresh[y:y + h, x:x + w]

	# Find the contours in the image
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

	# Return the original image, the contours, and the thresholded image
	return img, contours, thresh


# This function takes a string of text and speaks it using text-to-speech
def say_text(text):
	# If text-to-speech is not turned on, exit the function
	if not is_voice_on:
		return
	# Wait for any previous speech to finish before starting new speech
	while engine._inLoop:
		pass
	# Use the text-to-speech engine to say the given text
	engine.say(text)
	engine.runAndWait()


# This function is the main text recognition loop
def text_mode(inp, cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		# Read an image from the camera
		img = cam.read()[1]

		# Resize the image to 640x480
		img = cv2.resize(img, (640, 480))

		# Get the image, contours, and thresholded image from get_img_contour_thresh()
		img, contours, thresh = get_img_contour_thresh(img, inp)

		# Keep track of the previous text for comparison
		old_text = text
		word = ""
		count_same_frame = 0

		# Find the largest contour and predict the text from it
		if len(contours) > 0:
			contour = max(contours, key=cv2.contourArea)

			# If the contour area is greater than 9500, predict the text
			if cv2.contourArea(contour) > 9500:
				text = get_pred_from_contour(inp, contour, thresh)
				print(text)

				# If the predicted text is the same as the previous text, increment count_same_frame
				# Else, reset count_same_frame to 0
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				# If count_same_frame is greater than 7, say the text and add it to word
				if count_same_frame > 7:
					if len(text) == 1:
						Thread(target=say_text, args=(text,)).start()
					word = word + text
					count_same_frame = 0

			# If the contour area is less than 1000, say the word and reset word and text
			elif cv2.contourArea(contour) < 1000:
				if word != '':
					Thread(target=say_text, args=(word,)).start()
				text = ""
				word = ""
		# If there are no contours, say the word and reset word and text
		else:
			if word != '':
				Thread(target=say_text, args=(word,)).start()
			text = ""
			word = ""

		# Draw the blackboard and the predicted text and word
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
		if inp == 1:
			cv2.putText(blackboard, "Predicted ASL text- " + text, (30, 100), cv2.FONT_HERSHEY_PLAIN, 1,
						(255, 255, 255))
		else:
			cv2.putText(blackboard, "Predicted ISL text- " + text, (30, 100), cv2.FONT_HERSHEY_PLAIN, 1,
						(255, 255, 255))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))

		# If the voice is on, display "Voice on"; else, display "Voice off"
		if is_voice_on:
			cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		else:
			cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))

		# Draw the rectangle around the hand in the image
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# Concatenate the image and the blackboard and show the result
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)

		# Wait for a keypress
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			# Quit if the 'q' or 'c' keys are pressed
			break
		if keypress == ord('v') and is_voice_on:
			# Turn off voice if 'v' is pressed and voice is currently on
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			# Turn on voice if 'v' is pressed and voice is currently off
			is_voice_on = True
	#Check the final key pressed, and return either 2 or 0 based on it
	if keypress == ord('c'):
		return 2
	else:
		return 0

#Main function that uses the camera to recognize ASL or ISL gestures
def recognize(inp):
	cam = cv2.VideoCapture("/dev/video1")
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(inp, cam)
		else:
			break

#Call keras_predict() function with a 50x50 zero array and call the recognize() function with the input argument
keras_predict(model, np.zeros((50, 50), dtype = np.uint8))
recognize(inp)