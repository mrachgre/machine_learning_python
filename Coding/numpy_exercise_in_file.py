# Ex1: Write a NumPy program to reverse an array (first element becomes last).
# import numpy as np
# Input = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20])
#
# res = np.flip(Input)
# Ex2: Write a NumPy program to test whether each element of a 1-D array is also present in a second array
# Input Array1: [ 0 10 20 40 60]
#       Array2: [10, 30, 40]
# import numpy as np
# array1 = np.array([0, 10, 20, 40, 60])
# array2 = np.array([10, 30, 40])
# print(np.in1d(array1, array2))
# in1d la ham can tim
# Ex3: Write a NumPy program to find the indices of the maximum and minimum values along the given axis of an array
# Input Array [1,6,4,8,9,-4,-2,11]
# import numpy as np
# Input = np.array([1, 6, 4, 8, 9, -4, -2, 11])
# i = Input.argmin()
# k = Input.argmax()
# print(i, k)
# Ex4: Read the entire file story.txt and write a program to print out top 100 words occur most
import re
with open('story.txt') as f:
    string1 = f.read()
    i = re.findall(r'\b\w+\b', string1)
    dict1 = {}
    for k in i:
        if k not in dict1:
            dict1[k] = 1
        elif k in dict1:
            dict1[k] += 1
    sorted_dict = dict(sorted(dict1.items(), key = lambda item: item[1], reverse = True))
    number = 100
    for i, j in sorted_dict.items():
        if number > 0:
            print(i, j)
            number -= 1
        else:
            break


#




