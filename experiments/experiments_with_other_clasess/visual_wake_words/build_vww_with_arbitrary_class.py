from urllib.request import urlretrieve
from pycocotools.coco import COCO
from pathlib import Path
import subprocess
import argparse
import math
import os

parser = argparse.ArgumentParser(description="Visual Wake Words binary dataset builder")

parser.add_argument("target", type=str,
    help="Target class (e.g. bird).\n For available classes see: https://cocodataset.org/#explore")

args = parser.parse_args()

target_class = args.target

test_split = 0.2

print("\nPlease be patient, it will take some time...\n")

annotations_filename = 'annotations_trainval2014.zip'

if not Path(annotations_filename).exists() :
    urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', filename=annotations_filename)
    os.system(f"chmod 777 -R {annotations_filename}")

train_filename = 'train2014.zip'

if not Path(train_filename).exists() :
    urlretrieve('http://images.cocodataset.org/zips/train2014.zip', filename=train_filename)
    os.system(f"chmod 777 -R {train_filename}")
    
val_filename = 'val2014.zip'

if not Path(val_filename).exists() :
    urlretrieve('http://images.cocodataset.org/zips/val2014.zip', filename=val_filename)
    os.system(f"chmod 777 -R {val_filename}")

os.system('unzip annotations_trainval2014 -d data')
#os.system('rm annotations_trainval2014.zip')
os.system('unzip train2014 -d data')
#os.system('rm train2014.zip')
os.system('unzip val2014 -d data')
#os.system('rm val2014.zip')

os.system(f"mkdir -p {target_class}/1")
os.system(f"mkdir -p {target_class}/0")

#The label 1 is assigned as long as it has at least one bounding box
#corresponding to the object of interest (e.g. person) with the box area greater
#than 0.5% of the image area. - from https://arxiv.org/pdf/1906.05721.pdf
#The label 0 is assigned otherwise
#1 is target
def filter_ids(ids, foreground_class_of_interest_id=1, small_object_area_threshold=0.005) :
  filtered_ids = list()

  for id in ids :
    imAnn = coco.loadImgs(id)[0]
    Anns = coco.loadAnns(coco.getAnnIds(id))

    image_area = imAnn['height'] * imAnn['width']

    for annotation in Anns:
      normalized_object_area = annotation['area'] / image_area
      category_id = int(annotation['category_id'])
      # Filter valid bounding boxes
      if category_id == foreground_class_of_interest_id and \
          normalized_object_area > small_object_area_threshold:
        #if one is found, then the image is good
        #so you can append its index to the filtered ones
        filtered_ids.append(id)
        #and stop the search
        break
  discarded_ids = list(set(ids) - set(filtered_ids))

  return filtered_ids, discarded_ids

#Extracting images from COCO validation set 2014
pathToInstances = './data/annotations/instances_val2014.json'
coco = COCO(pathToInstances)

#get all images ids
allIds = coco.getImgIds()
img_count_val = len(allIds)
targetIds = coco.getImgIds(catIds=coco.getCatIds([target_class]))
backgroundIds = list(set(allIds) - set(targetIds))

targetIds, discardedIds = filter_ids(targetIds)
backgroundIds = list(set(backgroundIds) | set(discardedIds))

#load images metadata
targetImages = coco.loadImgs(targetIds)
backgroundImages = coco.loadImgs(backgroundIds)

#write images' names onto a textual file
sourceFile = open('target_images.txt', 'w')
for im in targetImages :
  print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('background_images.txt', 'w')
for im in backgroundImages :
  print('./data/val2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

#move images from COCO to the new dataset using the textual files
print('Moving COCO 2014 validation images...')
os.system(f"cat target_images.txt | xargs -I % mv % {target_class}/1")
os.system(f"cat background_images.txt | xargs -I % mv % {target_class}/0")

os.system('rm target_images.txt')
os.system('rm background_images.txt')

#Extracting images from COCO training set 2014
pathToInstances = './data/annotations/instances_train2014.json'
coco = COCO(pathToInstances)

#get all images ids
allIds = coco.getImgIds()
img_count_val = len(allIds)
targetIds = coco.getImgIds(catIds=coco.getCatIds([target_class]))
backgroundIds = list(set(allIds) - set(targetIds))

targetIds, discardedIds = filter_ids(targetIds)
backgroundIds = list(set(backgroundIds) | set(discardedIds))

#load images metadata
targetImages = coco.loadImgs(targetIds)
backgroundImages = coco.loadImgs(backgroundIds)

#write images' names onto a textual file
sourceFile = open('target_images.txt', 'w')
for im in targetImages :
  print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

sourceFile = open('background_images.txt', 'w')
for im in backgroundImages :
  print('./data/train2014/' + im['file_name'], file = sourceFile)
sourceFile.close()

#move images from COCO to the new dataset using the textual files
print('Moving COCO 2014 training images...')
os.system(f"cat target_images.txt | xargs -I % mv % {target_class}/1")
os.system(f"cat background_images.txt | xargs -I % mv % {target_class}/0")

os.system('rm target_images.txt')
os.system('rm background_images.txt')

os.system('rm -rf data')

n_of_targetImages = len(list((Path(target_class) / '1').glob('*.jpg')))
n_of_backgroundImages = len(list((Path(target_class) / '0').glob('*.jpg')))

print(f"Number of target images: {n_of_targetImages}")
print(f"Number of background images: {n_of_backgroundImages}")

print("Balancing dataset")

delta = n_of_targetImages - n_of_backgroundImages

images_to_remove_count = abs(delta)

print(f"Number of images to remove: {images_to_remove_count}")

if delta > 0 :
    #remove images from target class
    for image in sorted((Path(target_class) / '1').glob('*.jpg')) :
        image.unlink()
        images_to_remove_count = images_to_remove_count - 1
        if images_to_remove_count <= 0 :
            break
elif delta < 0 :
    #remove images from background class
    for image in sorted((Path(target_class) / '0').glob('*.jpg')) :
        image.unlink()
        images_to_remove_count = images_to_remove_count - 1
        if images_to_remove_count <= 0 :
            break

print(f"Number of target images: {len(list((Path(target_class) / '1').glob('*.jpg')))}")
print(f"Number of background images: {len(list((Path(target_class) / '0').glob('*.jpg')))}")

#test split
os.system(f"mkdir -p {target_class}/test/1")

images = sorted((Path(target_class) / '1').glob('*.jpg'))
test_images = images[:int(math.floor(test_split*len(images)))]

for image in test_images :
    image.rename(Path(target_class) / 'test' / '1' / image.stem)
    
os.system(f"mkdir -p {target_class}/training")
os.system(f"mv {target_class}/1 {target_class}/training")
    
os.system(f"mkdir -p {target_class}/test/0")

images = sorted((Path(target_class) / '0').glob('*.jpg'))
test_images = images[:int(math.floor(test_split*len(images)))]

for image in test_images :
    image.rename(Path(target_class) / 'test' / '0' / image.stem)
    
os.system(f"mkdir -p {target_class}/training")
os.system(f"mv {target_class}/0 {target_class}/training")

path = os.path.abspath(f"./{target_class}/")

os.system(f"chmod 777 -R {path}")

print(f"\nDataset saved in: {path}\n")
