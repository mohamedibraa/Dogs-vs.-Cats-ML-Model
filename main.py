import numpy as np
import timeit
import os
import cv2
from tqdm import tqdm
import random
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split

DATADIR = r'D:\University\forth year\f_term\Machine learning\assignment3\dogs-vs-cats\train\train'

X=[]
Y=[]
x_test=[]
y_test=[]
img=os.listdir(DATADIR)
img, X_test, y_train, y_testt = train_test_split(img, np.zeros(25000), test_size=0.0000001, random_state=42)

i=0
while(i<len(img)):  # iterate over each image per dogs and cats

    img_array = cv2.imread(os.path.join(DATADIR,img[i]),cv2.IMREAD_GRAYSCALE)
    img_array=cv2.resize(img_array,(128,64))
    fd, hog_image = hog(img_array, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=True)
    X.append(fd)



    if img[i].__contains__("cat"):
        Y.append(0)
    else:
        Y.append(1)

    img_array = cv2.imread(os.path.join(DATADIR, img[(i+1)%len(img)]), cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (128, 64))
    fd, hog_image = hog(img_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    x_test.append(fd)
    if img[(i+1)%len(img)].__contains__("cat"):
        y_test.append(0)
    else:
        y_test.append(1)

    print("helloooooooooooooo   ", i, Y[-1])
    i=i+random.randint(4,10)




C = 0.1  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X, Y)
sig_svc = svm.SVC(kernel='sigmoid', degree=3, C=C).fit(X, Y)

tic = timeit.default_timer()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
toc = timeit.default_timer()

print("polynominal execution time: ",toc-tic)

print("test svc",svc.score(x_test,y_test))
print("test lin_svc",lin_svc.score(x_test,y_test))
print("test rbf_svc",rbf_svc.score(x_test,y_test))
print("test poly_svc",poly_svc.score(x_test,y_test))
print("test sig_svc",sig_svc.score(x_test,y_test))

print("train svc",svc.score(X,Y))
print("train lin_svc",lin_svc.score(X,Y))
print("train rbf_svc",rbf_svc.score(X,Y))
print("train poly_svc",poly_svc.score(X,Y))
print("train sig_svc",sig_svc.score(X,Y))

