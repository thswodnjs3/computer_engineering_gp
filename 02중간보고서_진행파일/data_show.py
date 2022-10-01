import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show(data, feature):
    # pandas 출력 개수 제한 없애기
    pd.set_option('display.max_seq_items', None)

    # 각 feature와 해당 feature의 개수
    features, counts = np.unique(data[[feature]].to_numpy(), return_counts=True)

    # feature의 개수와 각 feature의 개수, 전체 개수 평균 출력
    print("\033[34mNumber of class : {}\033[0m".format(len(features)))
    print(data[feature].value_counts())
    print("\033[34mMean : {}\033[0m".format(round(data[feature].value_counts().mean(), 1)))

    # histogram 출력
    plt.figure(figsize=(30, 12))
    plt.title(feature, fontsize=30)
    plt.hist(data[feature])
    plt.axhline(y=data[feature].value_counts().mean(), color='r', label='Mean')
    plt.xticks(rotation=90, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=30)
    plt.show()

    # pandas 출력 개수 원상복귀
    # pd.set_option('display.max_seq_items', 'default')