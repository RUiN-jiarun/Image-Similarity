'''
利用颜色矩来评价相似度
缺点：对于色相相似种类不同的难以区分
'''
import cv2
import numpy as np
import os

def color_moments(filename):
    '''
    Compute low order moments(1,2,3) based on HSV color space
    :param filename:
    :return:
    '''
    img = cv2.imread(filename)
    if img is None:
        return
    # Get a thumbnail
    # img = cv2.resize(img, (900, 600))
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_var = np.var(h)
    s_var = np.var(s)
    v_var = np.var(v)
    color_feature.extend([h_var, s_var, v_var])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    color_feature.extend([h_skewness, s_skewness, v_skewness])
    color_feature = np.asarray(color_feature)

    return color_feature

def cosine_coef(x, y):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    coef = np.dot(x, y) / (x_norm * y_norm)
    coef = round(coef, 5)
    return coef

def cm_eval_similarity(base, lst):
    cms = []
    res = {}
    for i in range(len(lst)):
        cms.append(color_moments(lst[i]))
    ori = color_moments(base)
    for i in range(len(cms)):
        res[lst[i]] = cosine_coef(ori, cms[i])
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
    res = cm_eval_similarity(base, lst)
    print(res)




