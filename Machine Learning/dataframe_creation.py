from .image_preprocessing import get_roi
from .features_extraction import get_CD_r, get_exudate

import cv2 as cv
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

cdrList = []
dcdList = []
exudateList = []
targets = []

dataset_path = '../glaucoma/*'
image_count = 0

for f in glob.glob(dataset_path):
    print(f, end=' ')
    try:
        image = cv.imread(f)
        image = get_roi(image)
        discRadius, cupRadius = get_CD_r(image[0])
        result = [cupRadius/discRadius, discRadius - cupRadius]
        result.append(get_exudate(image[1]))
    except (cv.error, IndexError, ValueError, TypeError, AttributeError) as e:
        print('failed Due to => %s'%(e))
        continue

    cdr = result[0]
    dcd = result[1]
    exudate = result[2]

    cdrList.append(cdr)
    dcdList.append(dcd)
    exudateList.append(exudate)
    targets.append(0)
    
    image_count += 1

    print('Success!')
print(image_count)

dataset_path = '../normal/*'
image_count = 0
for f in glob.glob(dataset_path):
    print(f, end=' ')
    try:
        image = cv.imread(f)
        image = get_roi(image)
        discRadius, cupRadius = get_CD_r(image[0])
        result = [cupRadius/discRadius, discRadius - cupRadius]
    except (cv.error, IndexError, ValueError, TypeError, AttributeError) as e:
        print('failed Due to => %s'%(e))
        continue

    cdr = result[0]
    dcd = result[1]
    exudate = result[2]

    cdrList.append(cdr)
    dcdList.append(dcd)
    exudateList.append(exudate)
    targets.append(1)
    
    image_count += 1

    print('Success!')
print(image_count)

dataset_path = '../other/*'
image_count = 0
for f in glob.glob(dataset_path):
    print(f, end=' ')
    try:
        image = cv.imread(f)
        image = get_roi(image)
        discRadius, cupRadius = get_CD_r(image[0])
        result = [cupRadius/discRadius, discRadius - cupRadius]
    except (cv.error, IndexError, ValueError, TypeError, AttributeError) as e:
        print('failed Due to => %s'%(e))
        continue

    cdr = result[0]
    dcd = result[1]
    exudate = result[2]

    cdrList.append(cdr)
    dcdList.append(dcd)
    exudateList.append(exudate)
    targets.append(2)
    
    image_count += 1

    print('Success!')
print(image_count)

data = {'CDR' : cdrList, 'DCD' : dcdList, 'EXUDATE' : exudateList, 'target' : targets}
df = pd.DataFrame.from_dict(data)
df.to_csv(r'dataset.csv',  header=True)

def feature_selection():
    df = pd.read_csv('dataset_new.csv')
    df = df.drop(columns=['Unnamed: 0'])
    df.corr()
    corr = df.corr()
    sns.heatmap(round(corr,2), annot=True)
    plt.show()