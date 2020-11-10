'''
利用均方误差和结构相似度评价相似度
缺点：对于色相不同但是是同一类的图片评价不高，对于图像的几何变换无法判断
'''
import os
import numpy as np
import cv2

def ssim(img1 , img2):
    '''
    Structural Similarity Index, range[0,1]
    :param img1:
    :param img2:
    :return:
    '''
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = np.sqrt(((img1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((img2 - mu2) ** 2).mean())
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim


def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    imageA = cv2.imread(imageA)
    imageB = cv2.imread(imageB)
    h, w = imageA.shape[0:2]
    imageB = cv2.resize(imageB, (w, h))
    imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2YCrCb)
    imageA = cv2.split(imageA)[0]
    imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2YCrCb)
    imageB = cv2.split(imageB)[0]

    image1 = np.array(imageA, dtype=np.uint8)
    image2 = np.array(imageB, dtype=np.uint8)

    s = round(ssim(image1, image2), 5)
    return s

def ssim_eval_similarity(base, lst):
    res = {}
    for i in range(len(lst)):
        tmp = compare_images(base, lst[i])
        res[lst[i]] = tmp
    return res


if __name__ == "__main__":
    # PATH = 'test_img'
    PATH = 'horse'
    base = ''
    lst = []
    for root, dirs, files in os.walk(PATH):
        for name in files:
            if os.path.dirname(os.path.join(root, name)) != PATH:
                base = os.path.join(root, name)
            else:
                lst.append(os.path.join(root, name))
    res = ssim_eval_similarity(base, lst)
    print(res)
