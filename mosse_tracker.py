import numpy as np
import cv2
import vessel_segmentation as vs
from utils import pre_process, linear_mapping
import time

class MosseTracker:
	'''
	Minimum Output Sum of Squared Error tracker
	'''
	def __init__(self, sigma, num_pretrain, learning_rate):
		self.sigma = sigma
		self.num_pretrain = num_pretrain
		self.learning_rate = learning_rate
		self.Ai = None
		self.Bi = None
		self.G = None
		self.pos = None

	def pre_train(self, training_img):
		init_frame = cv2.cvtColor(training_img, cv2.COLOR_BGR2GRAY)
		init_frame = init_frame.astype(np.float32)
		
		# Select Object to Track [x, y, width, height]
		init_gt = cv2.selectROI('initial_img', training_img, False, False)        
		init_gt = np.array(init_gt).astype(np.int64)       
		
		# Compute Gaussian Response
		g = np.zeros((init_frame.shape[0:2])).astype(np.float32)
		g[(init_gt[1] + init_gt[3]//2), (init_gt[0] + init_gt[2]//2)] = 1.0
		gaussian_response = cv2.GaussianBlur(g, (0,0), self.sigma) * init_frame
		
		# start to create the training set ...
		# get the goal..
		g = gaussian_response[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
		init_frame = init_frame[init_gt[1]:init_gt[1]+init_gt[3], init_gt[0]:init_gt[0]+init_gt[2]]
		self.G = np.fft.fft2(g)

		# start to do the pre-training...
		Ai = np.zeros(self.G.shape)
		Bi = np.zeros(self.G.shape)
		for _ in range(self.num_pretrain):
			fi = pre_process(init_frame)
			Ai = Ai + self.G * np.conjugate(np.fft.fft2(fi))
			Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
		
		self.Ai = Ai * self.learning_rate
		self.Bi = Bi * self.learning_rate
		self.pos = init_gt.copy()

	def track(self, current_frame):
		# for idx in range(len(frame_list)):
		frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		frame_gray = frame_gray.astype(np.float32)
		
		# import ipdb;ipdb.set_trace()
		Hi = self.Ai / self.Bi
		fi = frame_gray[self.pos[1]:self.pos[1]+self.pos[3], self.pos[0]:self.pos[0]+self.pos[2]]
		fi = pre_process(fi)
		Gi = Hi * np.fft.fft2(fi)
		gi = linear_mapping(np.fft.ifft2(Gi))
		
		# find the max self.pos...
		max_pos = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
		
		# update the position...
		self.pos[1] += max_pos[0] - gi.shape[0] // 2
		self.pos[0] += max_pos[1] - gi.shape[1] // 2        

		# get the current fi..
		fi = frame_gray[self.pos[1]:self.pos[1]+self.pos[3], self.pos[0]:self.pos[0]+self.pos[2]]
		fi = pre_process(fi)

		# online update...
		self.Ai = self.learning_rate * (self.G * np.conjugate(np.fft.fft2(fi))) + (1 - self.learning_rate) * self.Ai
		self.Bi = self.learning_rate * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.learning_rate) * self.Bi

		return self.pos


if __name__ == "__main__":
	import argparse
	import os

	# import ipdb; ipdb.set_trace()
	parse = argparse.ArgumentParser()
	parse.add_argument('--learning_rate', type=float, default=0.125, help='learning rate')
	parse.add_argument('--sigma', type=float, default=2, help='')
	parse.add_argument('--num_pretrain', type=int, default=128, help='number of images to pretrain on')
	parse.add_argument('--rotate', action='store_true', help='if image to be rotated duing pretraining')
	parse.add_argument('--record', action='store_true', help='record all frames')
	parse.add_argument('--path', type=str, help='path to images', required=True)
	parse.add_argument('--img_extension', type=str, help='extension of image (jpg, png, pgm, etc.', default='.jpg')
	args = parse.parse_args()

	# Get images
	frame_list = []
	for frame in os.listdir(args.path):
		if os.path.splitext(frame)[1] == args.img_extension:
			frame_list.append(os.path.join(args.path, frame))

	# In-place ascending sort of frames
	frame_list.sort()

	# Initialize tracker
	tracker = MosseTracker(args.sigma, args.num_pretrain, args.learning_rate)

	# Train on first frame
	training_frame = cv2.imread(frame_list[0])
	tracker.pre_train(training_frame)

	# Track in remaining frames
	for idx, frame_name in enumerate(frame_list[1:]):
		frame = cv2.imread(frame_name)
		box_coordinates = tracker.track(frame) # box_coordinates = [min_x, min_y, max_x, max_y]
		
		# Visualize tracking
		print("frame: ", idx)
		# cv2.rectangle(frame, (box_coordinates[0], box_coordinates[1]), (box_coordinates[0]+box_coordinates[2], box_coordinates[1]+box_coordinates[3]), (255, 0, 0), 2)
		# cv2.imshow('tracking ...', frame)
		# cv2.waitKey(1)