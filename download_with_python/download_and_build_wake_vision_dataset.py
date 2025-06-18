from pathlib import Path
import shutil
import tarfile
import csv
import sys
import os

APIKEY = sys.argv[1]

#download the dataset
os.system(f"curl -L -O -J -H X-Dataverse-key:{APIKEY} https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/1HOPXC")

#unzip it
os.system("unzip dataverse_files.zip")

#delete zip file
os.system("rm dataverse_files.zip")

#move all extracted files in the current folder
os.system("mv dataverse_files/* .")

#delete void folder
os.system("rmdir dataverse_files")

#build dataset
path_to_dataset = Path('wake_vision')

folders_and_file_names = list()

#extract validation images metadata
img_names_0 = set()
img_names_1 = set()

with open('wake_vision_validation.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:]))
    for image_path, category, *_ in data[1:] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'validation/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'validation/1', 'file_names': img_names_1})

#extract test images metadata
img_names_0 = set()
img_names_1 = set()

with open('wake_vision_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:-1]))
    for image_path, category, *_ in data[1:-1] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'test/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'test/1', 'file_names': img_names_1})

#extract train (large) images metadata
img_names_0 = set()
img_names_1 = set()

with open('wake_vision_train_large.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:-1]))
    for image_path, category, *_ in data[1:] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'train_large/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'train_large/1', 'file_names': img_names_1})

#extract train (quality) images metadata
img_names_0 = set()
img_names_1 = set()

with open('wake_vision_train_bbox.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:-1]))
    for image_path, category, *_ in data[1:] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'train_quality/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'train_quality/1', 'file_names': img_names_1})


for element in folders_and_file_names :
    element['folder'].mkdir(parents=True)

path_to_unlabeled_images = path_to_dataset / 'unlabeled_images'
path_to_unlabeled_images.mkdir(parents=True)

#extract all compressed images and copy them in the corresponding folders
for zipped_file in Path('.').glob('*.tar.gz') :
    print(zipped_file)
    tar = tarfile.open(zipped_file, 'r:gz')
    tar.extractall()
    tar.close()
    images = set(Path('.').glob('*.jpg'))
    
    #copy extracted images to the respective folder
    for folder in folders_and_file_names :
        for image in images & folder['file_names'] :
            shutil.copy(image, folder['folder'])
   
    #gather unlabeled images
    for folder in folders_and_file_names :
        images = images - folder['file_names']
    for image in images :
        shutil.copy(image, path_to_unlabeled_images)
    
    #delete extracted images
    images = set(Path('.').glob('*.jpg'))
    for image in images : 
        image.unlink()

for zipped_file in Path('.').glob('*.tar.gz') :
    zipped_file.unlink()

for csv_file in Path('.').glob('*.csv') :
    csv_file.unlink()

print(f"Dataset saved in folder: {path_to_dataset}")
