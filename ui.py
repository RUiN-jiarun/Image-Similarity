from ssim import ssim_eval_similarity
from color_moments import cm_eval_similarity
from matplotlib import pyplot as plt
import matplotlib
import cv2
import os
import tkinter as tk


def select_pic(dir):
    base_img = ''
    lst = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if os.path.dirname(os.path.join(root, name)) != dir:
                base_img = os.path.join(root, name)
            else:
                lst.append(os.path.join(root, name))
    return base_img, lst


def color_moments_eval(dir):
    print(dir)
    base_img, lst = select_pic(dir)
    l = list(cm_eval_similarity(base_img, lst).items())
    l.sort(key=lambda ele:ele[1], reverse=True)
    # print(l)
    show_result(base_img, l)
    plt.show()


def ssim_eval(dir):
    base_img, lst = select_pic(dir)
    l = list(ssim_eval_similarity(base_img, lst).items())
    l.sort(key=lambda ele:ele[1], reverse=True)
    # print(l)
    show_result(base_img, l)
    plt.show()


def show_result(base_img, l):
    plt.title('')
    base = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    plt.xticks([])
    plt.yticks([])
    res1 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
    plt.xticks([])
    plt.yticks([])
    res2 = plt.subplot2grid((3, 4), (1, 0))
    plt.xticks([])
    plt.yticks([])
    res3 = plt.subplot2grid((3, 4), (1, 1))
    plt.xticks([])
    plt.yticks([])
    res4 = plt.subplot2grid((3, 4), (1, 2))
    plt.xticks([])
    plt.yticks([])
    res5 = plt.subplot2grid((3, 4), (1, 3))
    plt.xticks([])
    plt.yticks([])
    res6 = plt.subplot2grid((3, 4), (2, 0))
    plt.xticks([])
    plt.yticks([])
    res7 = plt.subplot2grid((3, 4), (2, 1))
    plt.xticks([])
    plt.yticks([])
    res8 = plt.subplot2grid((3, 4), (2, 2))
    plt.xticks([])
    plt.yticks([])
    res9 = plt.subplot2grid((3, 4), (2, 3))
    plt.xticks([])
    plt.yticks([])
    base_img = cv2.imread(base_img)
    img1 = cv2.imread(l[0][0])
    img2 = cv2.imread(l[1][0])
    img3 = cv2.imread(l[2][0])
    img4 = cv2.imread(l[3][0])
    img5 = cv2.imread(l[4][0])
    img6 = cv2.imread(l[5][0])
    img7 = cv2.imread(l[6][0])
    img8 = cv2.imread(l[7][0])
    img9 = cv2.imread(l[8][0])

    base.imshow(base_img[:, :, ::-1])
    base.set_title('Basic image')
    res1.imshow(img1[:, :, ::-1])
    res1.set_title(l[0][1])
    res2.imshow(img2[:, :, ::-1])
    res2.set_title(l[1][1])
    res3.imshow(img3[:, :, ::-1])
    res3.set_title(l[2][1])
    res4.imshow(img4[:, :, ::-1])
    res4.set_title(l[3][1])
    res5.imshow(img5[:, :, ::-1])
    res5.set_title(l[4][1])
    res6.imshow(img6[:, :, ::-1])
    res6.set_title(l[5][1])
    res7.imshow(img7[:, :, ::-1])
    res7.set_title(l[6][1])
    res8.imshow(img8[:, :, ::-1])
    res8.set_title(l[7][1])
    res9.imshow(img9[:, :, ::-1])
    res9.set_title(l[8][1])


def ui():
    win = tk.Tk()
    win.title('Image Similarity Compute')
    win.geometry('300x150')
    w = tk.Label(win, text="Please input the folder name:")
    w.pack()
    e = tk.Entry(win, show=None)
    e.pack()

    button1 = tk.Button(win, text="color moments evaluation", command=lambda: color_moments_eval(dir=e.get()))
    button1.pack()
    button2 = tk.Button(win, text="SSIM evaluation", command=lambda: ssim_eval(dir=e.get()))
    button2.pack()

    win.mainloop()


if __name__ == "__main__":
    ui()


