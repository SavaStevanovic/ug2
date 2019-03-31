import pandas as pd
import os
import argparse

#
# @brief      { Runs ffmpeg on local system, extracting and saving video frames
#             }
#
# @param      video           The video location (e.g., /Videos/video1.mp4)
# @param      outputLocation  The output location (e.g., /Frames/video1/)
#
# @return     { 0 (unless there is an error with extraction)}
#
def extractFramesFromVideo(video, outputLocation):
	# Ensure output location directory exists
	if not os.path.exists(outputLocation): 
		os.makedirs(outputLocation)
	cmd = 'ffmpeg -i "{0}" -start_number 0 -loglevel warning "{1}"'.format(video, os.path.join(outputLocation, '%d.png'))
	return os.system(cmd)

#
# @brief      { Copies all the frames in a sequence ans sequence annotations to a sequence folder}
#
# @param      sequence  The sequence number
# @param      framesOrigin   The folder containing all the frames 
# @param      outputFolder   The output directory in which each Sequence sub-directory will be created
#
# @return     { 0 (unless there is an error with copying) }
#
def extractSeqFrames(sequence, annotations, framesOrigin, outputFolder):
	# Ensure output location directory exists	
	if not os.path.exists(dst): 
		os.makedirs(dst)
	##
	## Visible Frame Selection
	##
	#  A frame is "visible" if the object of interest is not occluded and is inside the frame
	visibleFrames = annotations[(annotations['trackID'] == sequence) & (annotations['lost'] == 0) & (annotations['occluded'] == 0)][['frame']]
	if len(visibleFrames.index) == 0: 
		print('\t\tSequence {0} has no visible frames'.format(sequence))
		return 0
	## Start and end frames are the first and last visible frames
	start_frame = visibleFrames.min()[0]
	end_frame = visibleFrames.max()[0]
	print('\t\tFrames {0} -> {1}'.format(start_frame, end_frame))
	##
	## System copy of all frames in the present from the start to the end of the
	## visible sequence (clipping start and end frames of the sequence in which
	## the object of interest is not visible). 
	## 
	## We include frames inside the sequence in which the object might be 
	## invisible since it will become visible later on.
	##
	cmd = ''
	for frame in range(start_frame, end_frame+1):
		src = os.path.join(framesOrigin, str(frame) + '.png')
		cmd += 'cp "{0}" "{1}";\n'.format(src, outputFolder)

	parentDir = os.path.abspath(os.path.join(outputFolder, os.pardir))
	annFileLoc = os.path.join(parentDir, '{0}.txt'.format(os.path.basename(outputFolder)))
	print('\t\tSaving annotations file in {0}'.format(annFileLoc))

	annotations[(annotations['trackID'] == sequence) & 
	(annotations['frame'] >= start_frame) & 
	(annotations['frame'] <= end_frame)].to_csv(annFileLoc, index=False, sep=' ')
	return os.system(cmd)	


##
## Argument parser definition
## 
# 	param ['frames_folder', 'video_file'] <input_type> { Type of input file the sequence extractor will receive (video, or video frames) 
# 															if video, the video frames will be extracted using ffmpeg and saved in -frames_folder location
# 															if video frames, the extractor will proceed to extract frame sequences using the annotation_file
# 														}
# 	param <annotation_file> 							{ .txt file with the annotations for the video of interest }  									
# 	param <-frames_folder> 								{ Directory containing all the extracted frames of the video of interest 
# 															named a %d.png format (e.g.Frames/video1/[0.png, 1.png,...])
# 														}								
#	param <-video_path> 								{ Path to the video of interest (only necessary if input_type == video_file) }
#	param <output> 										{ Directory in which the extracted sequences will be saved (e.g., Sequences/video1/[Seq0, Seq1, ...]) }
#	
#	usage [input_type == video_file] python ch12_sequenceExtraction.py -video_path '~/UAV Collection/Videos/AV00115.mp4' -frames_folder '~/UAV Collection/Frames/AV00115/' video_file '~/UAV Collection/Annotations/AV00115.txt'
#	usage [input_type == frames_folder] python ch12_sequenceExtraction.py -frames_folder '~/UAV Collection/Frames/AV00115/' frames_folder '~/UAV Collection/Annotations/AV00115.txt' 


parser = argparse.ArgumentParser(description='UG2+ Track 1, Sub-Challenge 2: Sequence extraction from annotations', 
	prog='Sequence extractor')
parser.add_argument('input_type', choices=['frames_folder', 'video_file'], help='Input type: frames_folder: folder containing extracted video frames || video_file: original video file locations frames will be extracted in -frames_folder')
parser.add_argument('annotation_file', help='Annotation file path (e.g., /Annotations/video1_anns.txt)')
parser.add_argument('-frames_folder', '-f', help='Frames file path (e.g., /Video1Frames/), if input_type == video_file: frames will be extracted and saved in frames_folder location')
parser.add_argument('-video_path', '-v', help='Video file path (e.g., /Videos/video1.mp4)')
parser.add_argument('-output', '-o', 
	default='Sequences/',
	help='Location to store sequence images (e.g., /Sequences/video1/). ----Default: Sequences/')

## print(parser help
parser.print_help()

## Parse arguments
args = parser.parse_args()

##
## Verify that all argument dependencies are met
##
if args.input_type == 'frames_folder' and args.frames_folder is None: 
	parser.error('\nERROR: input_type frames_folder requires -frames_folder to be specified')
if args.input_type == 'video_file' and (args.video_path is None or args.frames_folder is None): 
	parser.error('\nERROR: input_type video_file requires -video_path and -frames_folder to be specified')

print('******************************************')

print('------------------------------------------')
print('Settings:')
print('------------------------------------------')
print('\n'.join('\t{0}:\n\t\t{1}'.format(k,v) for k,v in vars(args).items()))
print('******************************************')

if args.input_type == 'video_file':
	print('Extracting frames from video {0} at location {1}\n'.format(args.video_path, args.frames_folder))
	if extractFramesFromVideo(args.video_path, args.frames_folder):
		print('ERROR: Frame extraction failed')
		exit()
	print('Successful frame extraction')
	print('******************************************')

if not os.path.exists(args.frames_folder):
	parser.error('\nERROR: frames_folder location does not exist')

print('Retrieving annotation information...')
annotations = pd.read_csv(args.annotation_file, sep=' ')
# print(annotations[:5])
sequenceList = annotations['trackID'].unique()

print('\nFound <<<{0}>>> sequences in the video.'.format(len(sequenceList)))

print('\nExtracting frame sequences...')

for sequence in sequenceList:
	print('\tExtracting sequence {0}'.format(sequence))
	
	## Folder in which the sequence frames will be saved (e.g., Sequences/Video1/Seq0/)
	dst = os.path.join(args.output, '{0}'.format(str(sequence)))
	if extractSeqFrames(sequence, annotations, args.frames_folder, dst):	 
		print('\n\tERROR extracting one or more frames from sequence {0}\n'.format(sequence))

print('------------------------------------------')
print('Sequence extraction finished')
print('------------------------------------------')

