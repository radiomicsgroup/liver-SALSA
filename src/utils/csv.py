import os
import pandas as pd

path_folder = '/path/to/folder'
pts_inference = sorted([ i for i in os.listdir(path_folder)])
print(len(pts_inference), *pts_inference, sep = ', ')

paths = []
for pt in pts_inference:
    scan = [i for i in os.listdir(os.path.join(path_folder, pt)) if '.nii.gz' in i ]
    paths.append(os.path.join(path_folder, pt, scan[0]))

df = pd.DataFrame(paths, columns = ['PATHS'])
df.to_csv(os.path.join('/path/to/csv.csv'), index = False)
