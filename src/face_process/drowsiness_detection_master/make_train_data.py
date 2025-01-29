
# coding: utf-8

# In[15]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import metrics
#%matplotlib inline
#plt.style.use('ggplot')

knn = cv2.ml.KNearest_create()
train_data = np.empty((0, 2), dtype=np.float32)  # 학습 데이터 초기화
labels = np.empty((0, 1), dtype=np.int32)  # 라벨 초기화

def start(sample_size=25):
    global train_data, labels
    new_train_data = np.random.randint(0, 40, size=(sample_size, 2)).astype(np.float32)
    new_labels = classify_label(new_train_data).astype(np.int32).reshape(-1, 1)

    # 초기 학습 데이터 설정
    train_data = new_train_data
    labels = new_labels
    print(f"🔍 Training data initialized with {sample_size} samples.")

    # KNN 학습
    knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)
    return train_data, labels

def run(new_data):
    global train_data, labels

    # 새로운 데이터를 float32로 변환
    new_data = np.array(new_data, dtype=np.float32).reshape(1, -1)

    # 새로운 데이터를 학습 데이터에 추가
    train_data = np.vstack([train_data, new_data])
    new_label = np.array([[classify_new_data(new_data)]], dtype=np.int32)
    labels = np.vstack([labels, new_label])

    # KNN 모델 다시 학습
    knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)

    # KNN 실행
    ret, results, neighbor, dist = knn.findNearest(new_data, 5)
    print(f"✅ New Data Classified as: {results[0][0]}")
    return int(results[0][0])


    
#'num_samples' is number of data points to create
#'num_features' means the number of features at each data point (in this case, x-y conrdination values)
def generate_data(num_samples, num_features=2):
    data_size = (num_samples, num_features)
    data = np.random.randint(0, 40, size=data_size).astype(np.float32)

    # 🔍 디버깅용 출력 추가
    print(f"🔍 Generated data shape: {data.shape}, dtype: {data.dtype}")

    return data.reshape(-1, 2)  # 🔹 명확히 (N, 2)로 변환




#I determined the drowsiness-driving-risk-level based on the time which can prevent driving accident.
def classify_label(data):
    labels = []
    for d in data:
        if d[1] < d[0] - 15:
            labels.append(2)
        elif d[1] >= (d[0] / 2 + 15):
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels, dtype=np.int32)


def binding_label(train_data, labels) :
    power = train_data[labels==0]
    nomal = train_data[labels==1]
    short = train_data[labels==2]
    return power, nomal, short

def plot_data(po, no, sh) :
    plt.figure(figsize = (10,6))
    plt.scatter(po[:,0], po[:,1], c = 'r', marker = 's', s = 50)
    plt.scatter(no[:,0], no[:,1], c = 'g', marker = '^', s = 50)
    plt.scatter(sh[:,0], sh[:,1], c = 'b', marker = 'o', s = 50)
    plt.xlabel('x is second for alarm term')
    plt.ylabel('y is 10s for time to close eyes')

#We don't use below two functions. 
def accuracy_score(acc_score, test_score) :
    """Function for Accuracy Calculation"""
    print("KNN Accuracy :",np.sum(acc_score == test_score) / len(acc_score))
    #A line below this comment is exactly same with above one.
    #print(metrics.accuracy_score(acc_score, test_score))
    
def precision_score(acc_score, test_score) :
    """Function for Precision Calculation"""
    true_two = np.sum((acc_score == 2) * (test_score == 2))
    false_two = np.sum((acc_score != 2) * (test_score == 2))
    precision_two = true_two / (true_two + false_two)
    print("Precision for the label '2' :", precision_two)
    
    true_one = np.sum((acc_score == 1) * (test_score == 1))
    false_one = np.sum((acc_score != 1) * (test_score == 1))
    precision_one = true_one / (true_one + false_one)
    print("Precision for the label '1' :", precision_one)
    
    true_zero = np.sum((acc_score == 0) * (test_score == 0))
    false_zero = np.sum((acc_score != 0) * (test_score == 0))
    precision_zero = true_zero / (true_zero + false_zero)
    print("Precision for the label '0' :", precision_zero)

def classify_new_data(new_data):
    """ 실시간으로 새로운 데이터를 분류하는 함수 """
    x, y = new_data[0]
    if y < x - 15:
        return 2
    elif y >= (x / 2 + 15):
        return 0
    else:
        return 1
