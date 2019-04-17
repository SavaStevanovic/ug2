import shutil, os
path_frames= './input'
for dr, dirs, _ in os.walk(path_frames):
    for dir in dirs:
        t_dir=dr+'/'+dir
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)
        for dr1, _, files in os.walk(t_dir):
            k_dir='./output/alg0/'+dir
            if not os.path.exists(k_dir):
                os.makedirs(k_dir)
            for f in files:
                shutil.copy(t_dir+'/'+f, k_dir)