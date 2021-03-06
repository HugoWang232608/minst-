# plot.py
# 绘制图表
# @Time: 2021/1/19 16:10
# @Author: Hugo Wang
# @IDE: PyCharm
from matplotlib import pyplot as plt
import numpy as np

# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
    #                                      100 * np.max(predictions_array), class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
