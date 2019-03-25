import os

path_videos = './UG2 Dataset/UAV Collection/Videos'
path_annotations = './UG2 Dataset/UAV Collection/Annotations'
path_frames = './UG2 Dataset/UAV Collection/Frames'
path_sequences = './UG2 Dataset/UAV Collection/Sequences'

for r, d, f in os.walk(path_videos):
    for file in f:
        file_name = file.split('.')[0]
        os.system(
            "python ch1_2_sequenceExtraction.py video_file \"{0}\" -video_path \"{1}\" -frames_folder \"{2}\" -output \"{3}\""
            .format(
                path_annotations + '/' + file_name + '.txt',
                path_videos + '/' + file,
                path_frames + '/' + file_name + '/',
                path_sequences + '/' + file_name
            )
        )
        os.system(
            "python ch1_2_sequenceExtraction.py frames_folder \"{0}\" -frames_folder \"{1}\" -output \"{2}\""
            .format(
                path_annotations + '/' + file_name + '.txt',
                path_frames + '/' + file_name + '/',
                path_sequences + '/' + file_name
            )
        )