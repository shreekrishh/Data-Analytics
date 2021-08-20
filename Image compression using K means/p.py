import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os
import random


def load_image():
    img = cv2.imread("large.png")
    return np.asarray(img)




def initialize(cluster):

    emp=[]
    for i in range(cluster):
        emp.append(random.sample(range(256), 3))
    emp = np.asarray(emp)
    emp = emp.reshape(cluster, 3)

    return emp


def closest_centroids(train, centroids):
    k = len(centroids)
    grid = np.zeros((len(train), k))

    for i in range(k):
        grid[:, i] = np.sum((train - centroids[i]) ** 2, axis=1)
    index = np.argmin(grid, axis=1)
    return index


def update_centroids(index, cluster, train):
    m, n = train.shape
    centroids = np.zeros((cluster, n))
    for i in range(cluster):
        t = np.mean(train[np.where(index == i)], axis=0)
        centroids[i] = t
    return centroids


def k_means(img, cluster, iters=10):
    centroids = initialize(cluster)
    train = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    for i in range(iters):
        index = closest_centroids(train, centroids)
        centroids = update_centroids(index, cluster, train)
    return centroids, index


def func():
    try:
        image_path = sys.argv[1]
        assert os.path.isfile(image_path)
    except (IndexError, AssertionError):
        print('Please specify an image')
    img = load_image()
    w, h, d = img.shape
    print('Image found with width: {}, height: {}, depth: {}'.format(w, h, d))
    cluster = 8
    train = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    colors, p= k_means(img = img, cluster = cluster, iters=10)
    index = closest_centroids(train, colors)
    idx = np.array(index, dtype=np.uint8)
    x_reconstructed = np.array(colors[idx, :] , dtype=np.uint8).reshape((w, h, d))
    isWritten = cv2.imwrite('C:\Quant Club\Task 1- KMeans\moulik.png', x_reconstructed)
    cv2.imshow('window', x_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == '__main__':
    func()
