import pandas as pd
import numpy as np
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder
data = pd.read_csv("Iris (3).csv")
real_data = data.drop(['Id',"PetalLengthCm", 'PetalWidthCm', 'Species'], axis=1)
# Bài 1a: Áp dụng với các bạn chưa/ít kinh nghiệm về lập trình
#  Tính trung bình cộng chiều dài, chiều rộng đài hoa của từng loại (3 loại hoa
# average = real_data.mean()
# print(average)

# Bài 2a: Áp dụng với các bạn chưa/ít kinh nghiệm về lập trình
#  Xác định chiều dài, chiều rộng đài hoa lớn nhất của từng loại (3 loại hoa)
# max_lengpetal = max(real_data['SepalLengthCm'])
# max_widthpetal = max(real_data['SepalWidthCm'])
# print("Max of lengthPetal: {}, Max of WidthPetal: {}".format(max_lengpetal, max_widthpetal))
# Bài 3a: Áp dụng với các bạn chưa/ít kinh nghiệm về lập trình


# ENTER DATA INPUT
length = float(input("length:"))
width = float(input("width:"))
new_point = np.array([(length, width)])
# # cach 1 : 1KNN
# # split: train 100%
# data3a1 = real_data
# target3a1 = data['Species']
# #preprocessing
# scaler = StandardScaler()
# target_scaled = scaler.fit_transform(data3a1)
# encoder = OneHotEncoder(sparse_output=False)
# target3a1 = encoder.fit_transform(target3a1.values.reshape(-1,1))
# # call model
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(target_scaled, target3a1)
# # du doan
# new_point_scaler = scaler.transform(new_point)
# prediction_encoded = knn.predict(new_point_scaler)
# # change to old label
# # Chuyển đổi dự đoán one-hot thành nhãn gốc
# prediction_index = np.argmax(prediction_encoded)
# prediction_label = encoder.categories_[0][prediction_index]
# print(prediction_label)

# CACH 2 :
# data3a2 = data[['SepalLengthCm', 'SepalWidthCm']]
# target3a2 = data['Species']
# data3a2_numpy = data3a2.to_numpy()
# def distance(new_point, point):
#     return sqrt(np.sum((new_point-point) **2))
# distance = np.array([distance(new_point, flower) for flower in data3a2_numpy])
# min_index = np.argmin(distance)
# print(target3a2.iloc[min_index])

#  3a b: Trung binh con nho nhat den tat ca cac diem tung loai
# data3ab = data[['SepalLengthCm', 'SepalWidthCm', 'Species']]
# class_0 = data3ab.loc[data3ab['Species'] == 'Iris-setosa'].drop('Species', axis = 1)
# class_1 = data3ab.loc[data3ab['Species'] == 'Iris-versicolor'].drop('Species', axis = 1)
# class_2 = data3ab.loc[data3ab['Species'] == 'Iris-virginica'].drop('Species', axis = 1)
# # convert dataframe to numpy array
#
#
# dis_class_0 = pairwise_distances(new_point, class_0).mean()
# dis_class_1 = pairwise_distances(new_point, class_1).mean()
# dis_class_2 = pairwise_distances(new_point, class_2).mean()
#
# distance = np.array([dis_class_0, dis_class_1, dis_class_2])
# min_index = np.argmin(distance)
# class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# print(class_names[min_index])

#1b :
# new_data = data[['SepalLengthCm', 'Species']]
# data_numpy = new_data.to_numpy()
# class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# import pandas as pd
# import numpy as np
#
# # Giả sử `data` là DataFrame đã được định nghĩa
# # Chọn các cột cần thiết
# new_data = data[['SepalLengthCm', 'Species']]
#
# # Chuyển DataFrame thành mảng NumPy
# data_numpy = new_data.to_numpy()
#
# # Tạo từ điển để lưu trữ dữ liệu cho từng lớp
# class_dict = {name: [] for name in class_names}
#
# # Phân loại các điểm vào các lớp tương ứng
# for row in data_numpy:
#     class_dict[row[1]].append(row[0])
#
# # Hàm lấy 10 giá trị lớn nhất
# def get_top_10(values):
#     return sorted(values, reverse=True)[:10]
#
# # In kết quả
# for class_name in class_names:
#     print(class_name)
#     top_10 = get_top_10(class_dict[class_name])
#     print(top_10)

# 2b :
# new_data = data[['SepalLengthCm', 'SepalWidthCm', 'Species']]
# data2b_numpy = new_data.to_numpy()
# class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# sum_column = data2b_numpy[:, 0] + data2b_numpy[:, 1]
#
# data2b_numpy = np.hstack((data2b_numpy, sum_column.reshape(-1, 1)))
# 3ba)Điểm có tọa độ gần nó nhất. Nếu có 2 điểm thuộc 2 loại hoa khác nhau cùng
# nằm gần nhất với input, thì xác định sang điểm kế tiếp
# data3bc = data[['SepalLengthCm', 'SepalWidthCm']]
# target3bc_numpy = data['Species'].to_numpy()
# data3bc_numpy = data3bc.to_numpy()
# list_distance = np.sqrt(np.sum((data3bc_numpy - new_point) ** 2,axis = 1))
# min_indices = np.where(list_distance == np.min(list_distance))[0]
# while len(min_indices) > 1:
#     list_distance = np.where(list_distance == np.min(list_distance), np.inf, list_distance)
#     min_indices = np.where(list_distance == np.min(list_distance))[0]
# print(target3bc_numpy[min_indices[0]])

# 3bb 7 điểm gần nó nhất (Vote đa số). Nếu không xác định được class chiếm đa số,
# in ra bông hoa này ko thuộc class nào

# data3bc = data[['SepalLengthCm', 'SepalWidthCm']].to_numpy()
# target3bc_numpy = data['Species'].to_numpy()
# distance = np.sqrt(np.sum((data3bc-new_point) ** 2, axis = 1))
# distance = np.argsort(distance)[:7]
# result = target3bc_numpy[distance]
# dict = {}
# for i in result:
#     if i in dict:
#         dict[i] += 1
#     else:
#         dict[i] = 1
#
# max_count = max(dict.values())
# most_frequent = [key for key, value in dict.items() if value == max_count]
# if len(most_frequent) > 1:
#     print("none of class")
# else:
#     print(most_frequent[0])
# 3bc
# data3bc = data[['SepalLengthCm', 'SepalWidthCm']].to_numpy()
# target3bc_numpy = data['Species'].to_numpy()
# distance = np.sqrt(np.sum((data3bc-new_point) ** 2, axis = 1))
# distance = np.where(distance <= 2)[0]
# result = target3bc_numpy[distance]
# dict = {}
# for i in result:
#     if i in dict:
#         dict[i] += 1
#     else:
#         dict[i] = 1
#
# max_count = max(dict.values())
# most_frequent = [key for key, value in dict.items() if value == max_count]
# if len(most_frequent) > 1:
#     print("none of class")
# else:
#     print(most_frequent[0])