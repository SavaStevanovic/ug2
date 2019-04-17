import pandas as pd
import os
import argparse
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove tensorflow verbose

##
## Argument parser definition
## 
#	param <imagenet_equivalencies>			{
#												Path to the file detailing the equivalencies between UG2 superclasses and ImageNet sub-classes in a one_hot format
#												e.g. ~/UG2INet_onehot.txt
#											}
# 	param ['dataset', 'video'] <input_type> { Type of input file the evaluator will receive (dataset directory or single video sequence directory) 
# 												e.g. if dataset: input_folder= ~/UAV_Collection, where the structure of UAV_Collection-[Video1/, ...,VideoN/]/[Seq0/, 1/, ..., m/, seq1.txt, 1.txt, ..., m.txt]
# 												if video input_folder= ~/UAV_Collection/Video1/[Seq0/,...,SeqM/, seq0.txt, ..., seqm.txt]
# 											}
# 	param <input folder> 					{ Input folder as described in input type }  									
# 	param <output_folder> 					{ Directory to store the evaluation results in a txt file, 
# 												by default it is saved in a new directory created inside input_folder
# 											}								
#	param <output_filename> 				{ If not specified the evaluation file will be named Collection_results.txt (for input_type = dataset)
#												or the video folder name (for input_type = video)
#											}
#	param <-save_crops> 					{ Including it will make the algorithm save the processed image inputs in a folder inside each video sub directory (Video1/Crops/[seq0/, seq1/, ..., seqm/]) }
#	param <-backup_save>					{ Including it will save the evaluation results be saved after evaluating each video (when input_type = dataset) rather than waiting for the whole video collection to be processed}
#	
#	usage python ch12_sequenceEvaluation.py '~/UG2INet_onehot.txt' dataset '~/UAV Collection/Sequences/' ['UAV Collection/Evaluation_Results/'] [Evaluation_output.txt] [-s, -b]
#	usage python ch12_sequenceExtraction.py '~/UG2INet_onehot.txt' video '~/UAV Collection/Sequences/Video1/' [-s]
parser = argparse.ArgumentParser(description='UG2+ Track 1, Sub-Challenge 2: Frame Sequence classification', 
	prog='Sequence extractor')
parser.add_argument('imagenet_equivalencies', help='Path to UG2INet_onehot.txt file detailing the equivalencies between UG2 and ImageNet classes (e.g., /UAV Collection/UG2ImageNet.txt)')
parser.add_argument('input_type', choices=['dataset', 'video'], help='Input type: dataset: folder containing multiple video directories || video: video directory containing object sequence sub-directories')
parser.add_argument('input_folder', help='Directory path (e.g.,if input_type == video: /Video1Sequences/[0,1,...], else: /UAV Collection/Sequences/[Video1, Video2, ...] ')
parser.add_argument('output_folder', nargs = '?', 	help='Location to store classification results (e.g., /Video1Sequences/)')
parser.add_argument('output_filename', nargs = '?', 	help='Filename to store the classification results (e.g., Video1.txt)')
parser.add_argument('-save_crops', '-s', default=False, action='store_true',
	help='True|False: whether to save the cropped object images fed to the classification networks')
parser.add_argument('-backup_save', '-b', default=False, action='store_true',
	help='True|False: whether to save the evaluation results of each video as soon as they are ready when evaluating an entire dataset')
## Print parser help
parser.print_help()

## Parse arguments
args = parser.parse_args()

##
## Verify that all argument dependencies are met
##

if args.output_folder is None:
	args.output_folder = os.path.join(args.input_folder, 'Evaluation_results')


if not os.path.exists(args.output_folder): 
	os.makedirs(args.output_folder)

print '******************************************'

print '------------------------------------------'
print 'Settings:'
print '------------------------------------------'
print '\n'.join('\t{0}:\n\t\t{1}'.format(k,v) for k,v in vars(args).items())
print '******************************************'

##
## @brief      { Given an annotated frame, and the bounding box of the object of interest, it crops it 
## 				(ensuring the cropped image is a square) and then resizes it to 224x224 (VGG16 input size) 
## 				}
##
## @param      image         The input frame
## @param      bounding_box  The bounding box of the object of interest (xmax:x1, xmin: x2, ymax: y1, ymin:y2)
##
## @return     { description_of_the_return_value }
##
def crop_image (image, bounding_box):
	MIN_ROI_SIZE = 224
	roi_width = bounding_box['xmax'] - bounding_box['xmin']
	roi_height = bounding_box['ymax'] - bounding_box['ymin']
	roi_xcenter = bounding_box['xmin'] + (roi_width/2)
	roi_ycenter = bounding_box['ymin'] + (roi_height/2)
	frame_width = image.shape[1]
	frame_height = image.shape[0]
	roi_size = MIN_ROI_SIZE
	# If any of the dimensions of the crop region is larger than 224
	# we use that dimension's size as the ROI size (if its bigger than the frame width or height we use the min of them)
	if roi_width > MIN_ROI_SIZE or roi_height > MIN_ROI_SIZE:
		roi_size = max(roi_width, roi_height)
		if roi_size > frame_width or roi_size > frame_height:
			roi_size = min(frame_width, frame_height)	
	halfRoi = -(-roi_size // 2) # ceiling
	# Get new xmin and ymin locations while ensuring the roi is within the frame
	if roi_xcenter - halfRoi >= 0:
		if roi_xcenter + halfRoi <= frame_width:
			roi_xmin = roi_xcenter - halfRoi
		else: 
			roi_xmin = frame_width - roi_size
	else:
		roi_xmin = 0
	if roi_ycenter - halfRoi >= 0:
		if roi_ycenter + halfRoi <= frame_height:
			roi_ymin = roi_ycenter - halfRoi
		else:
			roi_ymin = frame_height - roi_size
	else:
		roi_ymin = 0	
	return cv2.resize(image[roi_ymin:(roi_ymin+roi_size), roi_xmin:(roi_xmin+roi_size)], (MIN_ROI_SIZE, MIN_ROI_SIZE))

##
## @brief      { Calls VGG16 pre-proessing functions to ensure the image has the correct type and dimensions }
##
## @param      image  The -cropped- image containing the object of interest
##
## @return     { the pre-processed image ready for evaluation }
##
#
##
def preprocess_image (image):
	image = image[...,::-1].astype(np.float32)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	return image

##
## @brief      { Crops the object of interest along all the frames in a sequence and returns it in an array ready for VGG16 to .predict the image array }
##
## @param      folder_path       The folder path containing the sequence images (e.g. ~/Video1/Sequence0/[0.png, 1.png, ...])
## @param      annotations_file  The annotations file for that given sequence (contains the bounding box for the object, its class and visibility)
## @param      ug2inet           The ug2inet_onehot equivalencies
##
## @return     { Numpy array with the cropped object images with size (X, 224,224,3), where X is the number of images in the sequence }
##
def video_frames_processing (folder_path, annotations_file, ug2inet):
	images = None
	annotations = pd.read_csv(os.path.join(folder_path, annotations_file), sep=' ')
	sequence = annotations_file[:-4]
	sequence_folder = os.path.join(folder_path, sequence)

	label = annotations['label'].unique()[0]
	if not label in ug2inet.index:
		print '\t\t\tWARNING: NON SUPPORTED CLASS >>{0}<<: skipping sequence'.format(label)
		return None, False
	if args.save_crops:	
		save_loc = os.path.join(folder_path, 'Crops', sequence)
		if not os.path.exists(save_loc): 
			os.makedirs(save_loc)

	
	for idx, annotation in annotations.iterrows():
		# check whether the object is visible in this frame before trying to extract it
		if not (annotation.lost == 0 and annotation.occluded == 0): continue

		frame_path = os.path.join(sequence_folder, str(annotation.frame) + '.png')
		frame_img = cv2.imread(frame_path)
		if frame_img is None:
			print 'Image {0} does not exist'.format(frame_path)
			exit()

		bounding_box = {'xmax': annotation.xmax, 'xmin': annotation.xmin, 'ymax': annotation.ymax, 'ymin': annotation.ymin}
		object_img = crop_image(frame_img, bounding_box)
		if args.save_crops:	
			cv2.imwrite(os.path.join(save_loc, str(annotation.frame) + '.png'), object_img)

		object_img = preprocess_image(object_img)
		if images is None:
			images = object_img
		else:
			images = np.concatenate((images, object_img))

	return images, label

##
## @brief      { Divides a numpy array into batches of size ~batch_size }
##
## @param      images      The numpy array
## @param      batch_size  The batch size
##
## @return     { A list of numpy arrays os size ~batch_size }
##
def batch_generation(images, batch_size):
	total_batches = -(images.shape[0]//-batch_size)
	batches = np.array_split(images, total_batches)
	return batches

##
## @brief      { Taking into account Memory constraints it might be unfeasible to store all images in a sequence in memory
##  				thus, we predict them in batches obtain their confidence and return it}
##
## @param      model       The model to predict the class confidence (in this case we are using VGG16)
## @param      images      The numpy array of images
## @param      batch_size  The batch size
##
## @return     { An array containing the prediction results for all the images}
##
def batch_prediction(model, images, batch_size=32):
	batches = batch_generation(images, batch_size)
	predictions = []
	for batch in batches:
		p = model.predict_on_batch(batch)
		predictions += [list(frame_prediction) for frame_prediction in p]
	return predictions


##
## @brief      { Video evaluation: evaluates the classification performance of each object sequence in the video over different metrics }
##
## @param      video_folder  The video folder containing all sequence folders and their respective annotation files
## @param      model         The model we will use to predict each image classes
## @param      ug2inet       The ug2inet_onehot equivalencies (equivalencies between UG2 super-classes and ImageNet classes)
##
## @return     { A list of directories, where each directory contains a sequence evaluation results over different metrics }
##
def video_evaluation(video_folder, model, ug2inet):	
	files = [filename for filename in os.listdir(video_folder) if not os.path.isdir(os.path.join(video_folder, filename))]
	
	video_name = os.path.basename(video_folder)
	print ('\tProcessing {0} sequence frames for video {1}'.format(len(files), video_folder))
	video_eval_results = []
	for sequence_annotation in files:
		sequence_evaluation = {'Video': video_name, 'Sequence': sequence_annotation[:-4]}

		print ('\t\tProcessing Sequence {0}'.format(sequence_annotation))
		images, sequence_evaluation['Class'] = video_frames_processing(video_folder, sequence_annotation, ug2inet)
		if not sequence_evaluation['Class']: continue 
		print ('\t\t\tFinished processing, classifying sequence with object class "{0}"'.format(sequence_evaluation['Class']))
		
		model_output = batch_prediction(model, images)
		predictions = [list(frame_prediction) for frame_prediction in model_output]		
		onehot_true_label = ug2inet.ix[sequence_evaluation['Class']].values.tolist()
		
		## Since all frames in a given sequence have the same super-class label
		## the true labels we compare them to are the same 
		## (e.g. a sequence depicts a car over multiple frames, thus all true labels should be car)
		true_values = [onehot_true_label for sequence_prediction in predictions]

		print ('\t\t\tEvaluating classification results...')
		##
		## Evaluation metric calculation
		## 	Label Ranking Average Precision: Evaluates the average fraction of labels ranked above a particular label l element-of super-class_i which are actually in super-class_i
		## 		Value ranges (0.0 -> 1.0), Best value = 1.0
				
		sequence_evaluation['LRAP'] = metrics.label_ranking_average_precision_score(true_values, predictions)		

		video_eval_results.append(sequence_evaluation)				
	
	return video_name, video_eval_results

## Load the equivalencies file detailing the equivalencies between UG2 super-classes and ImageNet classes in a one_hot format
ug2inet = pd.read_csv(args.imagenet_equivalencies, sep='\t')
ug2inet.set_index('UG2Class', inplace=True)

## Load pre-trained evaluation model
model = VGG16(weights='imagenet')

##
## Evaluation of classification performance over different metrics
## 
if args.input_type == 'video':
	print 'Processing Video'
	# Format the input_folder as a path
	args.input_folder = os.path.abspath(args.input_folder)
	# Evaluate the performance of all the sequences in the video over different metrics
	video_name, video_eval = video_evaluation(args.input_folder, model, ug2inet)
	##
	## Save the results
	# If the output file_name was not specified set it to the name of the video folder
	if args.output_filename is None:
		args.output_filename = str(video_name)+ '.txt'
	output_path = os.path.join(args.output_folder, args.output_filename)
	print ('\tSaving classification results to {0}'.format(output_path))
	video_eval = pd.DataFrame(video_eval)
	video_eval.to_csv(output_path, sep = '\t', index=False)
else:
	print 'Processing Video Collection'
	# Obtain the video directories contained in the input_folder
	input_subdirs = [os.path.join(args.input_folder, filename) for filename in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, filename)) and filename!='Evaluation_results']
	
	# Used to store the classification performance for all videos
	collection_eval = []

	# If the output file_name was not specified set it to the name of the video folder
	if args.output_filename is None:
		args.output_filename = 'Collection_results.txt'
	output_path = os.path.join(args.output_folder, args.output_filename)

	# For each Video: Evaluate the performance of all the sequences in the video over different metrics
	for video_folder in input_subdirs:
		video_name, video_eval = video_evaluation(video_folder, model, ug2inet)		
		# append the results to the results of previous videos
		collection_eval+=video_eval
		# if backup_save was specified as an input parameter, save the evaluation results obtained until now into a file
		if args.backup_save:
			collection_eval_df = pd.DataFrame(collection_eval)
			collection_eval_df.to_csv(output_path[:-4]+'_b.txt', sep = '\t', index=False)
	
	collection_eval_df = pd.DataFrame(collection_eval)
	collection_eval_df = collection_eval_df[['Class', 'LRAP']].groupby(['Class']).mean()
	collection_eval_df.loc['TOTAL AVG'] = collection_eval_df.mean()[0]
	print collection_eval_df
	collection_eval_df.to_csv(output_path, sep = '\t')