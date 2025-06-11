from urllib.request import urlretrieve
from pathlib import Path
import subprocess
import argparse
import csv

parser = argparse.ArgumentParser(description="Wake Vision binary classification dataset builder. It builds a binary classification dataset for the target class. It does not apply Wake Vision pre-processing for standardized evaluation. See appendix J of Wake Vision paper for further information: https://arxiv.org/pdf/2405.00892")

parser.add_argument("target", type=str,
    help="Target class (e.g. Bird).\n For available classes see: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv")
parser.add_argument(
    '--num_processes',
    type=int,
    default=8,
    help='Number of parallel processes to use (default is 8).')
args = parser.parse_args()

target = args.target
num_processes = args.num_processes

confidence_threshold = 0.7

dataset_name = f"Wake_Vision_{target}"
path_to_dataset = Path(dataset_name)
path_to_dataset.mkdir()

print("The entire process requires time. Please be patient.")
print("Downloading required files...")

#download required files from OpenImageV7
class_descriptions_boxable_filename = 'oidv7-class-descriptions-boxable.csv'

if not Path(class_descriptions_boxable_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv', filename=class_descriptions_boxable_filename)

test_annotations_filename = 'test-annotations-bbox.csv'

if not Path(test_annotations_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv', filename=test_annotations_filename)
    
validation_annotations_filename = 'validation-annotations-bbox.csv'

if not Path(validation_annotations_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv', filename=validation_annotations_filename)

train_annotations_filename = 'oidv6-train-annotations-bbox.csv'

if not Path(train_annotations_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv', filename=train_annotations_filename)

train_large_machine_annotations_filename = 'oidv7-train-annotations-machine-imagelabels.csv'

if not Path(train_large_machine_annotations_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-machine-imagelabels.csv', filename=train_large_machine_annotations_filename)
    
train_large_human_annotations_filename = 'oidv7-train-annotations-human-imagelabels.csv'

if not Path(train_large_human_annotations_filename).exists() :
    urlretrieve('https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv', filename=train_large_human_annotations_filename)

downloader_filename = 'downloader.py'

label_name = None

#check if target exists and fetch corresponding label code
with open(class_descriptions_boxable_filename, newline='') as csvfile :
    reader = csv.DictReader(csvfile)
    for row in reader :
        if row['DisplayName'] == target :
            label_name = row['LabelName']
            break

if label_name == None :
    print("Target not found.")
    exit()

splits = ('train', 'validation', 'test')
annotations_files = (train_annotations_filename, validation_annotations_filename, test_annotations_filename)

print(f"Target: {target}")

for split, annotations_filename in zip(splits, annotations_files) :
    target_images = set()
    background_images = set()
    
    print(f"Preparing {split} split...")

    with open(annotations_filename, newline='') as csvfile :
        reader = csv.DictReader(csvfile)
        for row in reader :
            if row['LabelName'] == label_name :
                if row['IsDepiction'] == '0' :
                    target_images.add(row['ImageID'])
                else :
                    background_images.add(row['ImageID'])
            else :
                background_images.add(row['ImageID'])
                
    print(len(target_images))
    print(len(background_images))
    
    #balance 
    n_targets = len(target_images)
    n_backgrounds = len(background_images)
    
    delta = n_targets - n_backgrounds
    
    if delta > 0 :
        n_targets = n_backgrounds
    else :
        n_backgrounds = n_targets
    
    #sort entries for reproducibility
    target_images = list(target_images)
    background_images = list(background_images)    
    
    target_images.sort()
    background_images.sort()
    
    #balance
    target_images = target_images[:n_targets]
    background_images = background_images[:n_backgrounds]
    
    #prepare for download
    path_to_target_images_list = Path('target_images.txt')
    path_to_background_images_list = Path('background_images.txt')
    
    with open(path_to_target_images_list, 'w') as f:
       for image in target_images :
           print(f"{split}/{image}", file=f)
    
    with open(path_to_background_images_list, 'w') as f:
       for image in background_images :
           print(f"{split}/{image}", file=f)
    
    path_to_target_images = path_to_dataset / split / target
    path_to_background_images = path_to_dataset / split / 'background'

    path_to_target_images.mkdir(parents=True, exist_ok=True)
    path_to_target_images.mkdir(parents=True, exist_ok=True)

    #download
    subprocess.run(["python3", downloader_filename, str(path_to_target_images_list), f"--download_folder={str(path_to_target_images)}", f"--num_processes={num_processes}"])
    subprocess.run(["python3", downloader_filename, str(path_to_background_images_list), f"--download_folder={str(path_to_background_images)}", f"--num_processes={num_processes}"])

    path_to_target_images_list.unlink()
    path_to_background_images_list.unlink()

    #save for preparing train large split    
    if split == 'train' :
        train_quality_images = set(target_images) | set(background_images)
        (path_to_dataset / 'train').rename(Path(path_to_dataset / 'train_quality'))

#train large split
print(f"Preparing train_large split...")
split = 'train'

target_images = set()
background_images = set()

with open(train_large_human_annotations_filename, newline='') as csvfile :
    reader = csv.DictReader(csvfile)
    for row in reader :
        if row['LabelName'] == label_name :
            if row['Confidence'] == '1' :
                target_images.add(row['ImageID'])
            elif row['Confidence'] == '0' :
                background_images.add(row['ImageID'])
        else :
            background_images.add(row['ImageID'])

with open(train_large_machine_annotations_filename, newline='') as csvfile :
    reader = csv.DictReader(csvfile)
    for row in reader :
        if row['LabelName'] == label_name :
            if float(row['Confidence']) >= confidence_threshold :
                if not row['ImageID'] in background_images :
                    target_images.add(row['ImageID'])
            else :
                background_images.add(row['ImageID'])

#remove images already present in train_quality
target_images = target_images - train_quality_images
background_images = background_images - train_quality_images

print(len(target_images))
print(len(background_images))
    
#balance 
n_targets = len(target_images)
n_backgrounds = len(background_images)
    
delta = n_targets - n_backgrounds
    
if delta > 0 :
    n_targets = n_backgrounds
else :
    n_backgrounds = n_targets
    
#sort entries for reproducibility
target_images = list(target_images)
background_images = list(background_images)    
    
target_images.sort()
background_images.sort()
    
#balance
target_images = target_images[:n_targets]
background_images = background_images[:n_backgrounds]
    
path_to_target_images_list = Path('target_images.txt')
path_to_background_images_list = Path('background_images.txt')

with open(path_to_target_images_list, 'w') as f:
   for image in target_images :
       print(f"{split}/{image}", file=f)
    
with open(path_to_background_images_list, 'w') as f:
   for image in background_images :
       print(f"{split}/{image}", file=f)

path_to_target_images = path_to_dataset / split / target
path_to_background_images = path_to_dataset / split / 'background'

path_to_target_images.mkdir(parents=True, exist_ok=True)
path_to_target_images.mkdir(parents=True, exist_ok=True)

subprocess.run(["python3", downloader_filename, str(path_to_target_images_list), f"--download_folder={str(path_to_target_images)}", f"--num_processes={num_processes}"])
subprocess.run(["python3", downloader_filename, str(path_to_background_images_list), f"--download_folder={str(path_to_background_images)}", f"--num_processes={num_processes}"])

path_to_target_images_list.unlink()
path_to_background_images_list.unlink()

(path_to_dataset / 'train').rename(Path(path_to_dataset / 'train_large'))

print(f"Dataset saved in {path_to_dataset.resolve()}")
