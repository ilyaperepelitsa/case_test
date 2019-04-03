import math
import pandas as pd

math.degrees(math.radians(5))
math.degrees(math.radians(355))

math

def angle_distance(x,y):
    a = x - y
    a = (a + 180) % 360 - 180
    return abs(a)

def angle_midpoint(x,y):
    vals = [x, y]
    vals.sort(reverse = False)
    x, y = vals
    a = x - y

    a = (a) % 360
    if a < 90:
        return ((abs(a) + 180) / 2) + y
    else:
        return ((abs(a)) / 2) + y

# def angle_distance_2(x,y):
#     a = x - y
#     a = (a - 180) % 360 + 180
#     return abs(a)

def angle_distance_2(x,y):
    a = x - y
    # a = ((a - 180) % 360 + 180) % 360
    a = (a - 180) % 360 + 180

    # vals = [x, y]
    # vals.sort(reverse = False)
    # x, y = vals
    #
    return ((a/2) + y) % 360
    # return a
    # return abs(a)


angle_distance(0, 540)
angle_distance_2(0, 90)
# angle_midpoint(0, 181)

angle_distance(10, angle_midpoint(0, 180))


import numpy as np
import matplotlib.pyplot as plt
num_samples = 1000
# make a simple unit circle
theta = np.linspace(0, 2*np.pi, num_samples)
# theta
a, b = 1 * np.cos(theta), 1 * np.sin(theta)

# generate the points
# theta = np.random.rand((num_samples)) * (2 * np.pi)
# r = np.random.rand((num_samples))


# x, y = np.cos(theta), np.sin(theta)
# plots
plt.figure(figsize=(7,7))
plt.plot(a, b, linestyle='-', linewidth=2)
dict_vals = {"A" : 305,"B" : 335,"C" : 140}
for i in dict_vals.items():
    nu_theta = math.radians(i[1])
    y, x = np.cos(nu_theta), np.sin(nu_theta)
    plt.plot(x, y, marker='o', linestyle=':', label=i[0])

plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.grid()
plt.legend(loc='upper right')
plt.show(block=True)


plt.hist(np.random.rand((num_samples)))



A = np.random.rand((500000)) * 360
B = np.random.rand((500000)) * 360
a_b_data = pd.concat([pd.Series(A), pd.Series(B)], axis = 1)
a_b_data = a_b_data.rename({0: "A", 1: "B"}, axis = "columns")
a_b_data["C"] = a_b_data.apply(lambda x: angle_distance_2(x["A"], x["B"]), axis = 1)
a_b_data["X"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_A"] = a_b_data.apply(lambda x: angle_distance(x["A"], x["X"]), axis = 1)
a_b_data["distance_B"] = a_b_data.apply(lambda x: angle_distance(x["B"], x["X"]), axis = 1)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]
### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]
### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]


a_b_data["C"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]
### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]
### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]


###############
# B ADDS BETWEEN 0 AND 90
A = np.random.rand((500000)) * 360
B = A + np.random.rand(500000)*90
a_b_data = pd.concat([pd.Series(A), pd.Series(B)], axis = 1)
a_b_data = a_b_data.rename({0: "A", 1: "B"}, axis = "columns")
a_b_data["C"] = a_b_data.apply(lambda x: angle_distance_2(x["A"], x["B"]), axis = 1)
a_b_data["X"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_A"] = a_b_data.apply(lambda x: angle_distance(x["A"], x["X"]), axis = 1)
a_b_data["distance_B"] = a_b_data.apply(lambda x: angle_distance(x["B"], x["X"]), axis = 1)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]
### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]

a_b_data["C"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]



###############
# B ADDS BETWEEN 90 AND 180
A = np.random.rand((500000)) * 360
B = A + np.random.rand(500000)*90 + 90

a_b_data = pd.concat([pd.Series(A), pd.Series(B)], axis = 1)
a_b_data = a_b_data.rename({0: "A", 1: "B"}, axis = "columns")
a_b_data["C"] = a_b_data.apply(lambda x: angle_distance_2(x["A"], x["B"]), axis = 1)
a_b_data["X"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_A"] = a_b_data.apply(lambda x: angle_distance(x["A"], x["X"]), axis = 1)
a_b_data["distance_B"] = a_b_data.apply(lambda x: angle_distance(x["B"], x["X"]), axis = 1)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]

a_b_data["C"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]



###############
# B ADDS 180
A = np.random.rand((500000)) * 360
B = A + 180

a_b_data = pd.concat([pd.Series(A), pd.Series(B)], axis = 1)
a_b_data = a_b_data.rename({0: "A", 1: "B"}, axis = "columns")
a_b_data["C"] = a_b_data.apply(lambda x: angle_distance_2(x["A"], x["B"]), axis = 1)
a_b_data["X"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_A"] = a_b_data.apply(lambda x: angle_distance(x["A"], x["X"]), axis = 1)
a_b_data["distance_B"] = a_b_data.apply(lambda x: angle_distance(x["B"], x["X"]), axis = 1)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)

### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]

a_b_data["C"] = pd.Series(np.random.rand((500000)) * 360)
a_b_data["distance_C"] = a_b_data.apply(lambda x: angle_distance(x["C"], x["X"]), axis = 1)
### WE WIN
((a_b_data["distance_C"] > a_b_data["distance_A"] )& (a_b_data["distance_C"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### A WINS
((a_b_data["distance_A"] > a_b_data["distance_C"] )& (a_b_data["distance_A"] > a_b_data["distance_B"])).sum() / a_b_data.shape[0]

### B WINS
((a_b_data["distance_B"] > a_b_data["distance_A"] )& (a_b_data["distance_B"] > a_b_data["distance_C"])).sum() / a_b_data.shape[0]
