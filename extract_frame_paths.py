import os

path_annotations = './UG2 Dataset/UAV Collection/Annotations/'
path_frames= './UG2 Dataset/UAV Collection/Frames'
frame_paths=open("frames_target.txt", "a+")
for dr, dirs, files in os.walk(path_annotations):
    for file in files:
        annotation= open(dr+'/'+file,"r")
        for i, a in enumerate(annotation):
            if 'trackID xmin ymin xmax ymax frame lost occluded generated label\n'!=a:
                frame_paths.write(path_frames+'/'+file.split('.')[0] +'/'+str(i-1)+'.png '+ a.split()[-1] +os.linesep)
frame_paths.close()