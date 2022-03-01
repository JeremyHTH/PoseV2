import sys
sys.path.append('src/Pose_util')

import Pose_util.PoseModule as pm
from knn import knn
import csv


def main():
    
    print(read_csv('src/knn/data.csv'))

def read_csv(link):
    with open(link,'r',newline='') as f:
        rows = csv.reader(f)

        datalist = list(map(lambda x: ','.join(x),rows))

    return datalist

if __name__== "__main__":
    main()