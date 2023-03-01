import matplotlib.pyplot as plt
import numpy as np

# F1 score
er = [(200, 35.25), (500, 52.2), (2000, 81.9), (5120, 90.11)]
der = [(200, 34.68), (500, 47.2), (2000, 67.37), (5120, 69.67)]
derpp = [(200, 36.62), (500, 40.59), (2000, 73.24), (5120, 81.15)]

fig, ax = plt.subplots()

plt.plot(*zip(*er), '-o', label = "ER")
plt.plot(*zip(*der), '-o', label = "DER")
plt.plot(*zip(*derpp), '-o', label = "DERPP")

plt.title('GrainSpaceA - F1 score vs buffer size')
plt.xlabel('Buffer size')
plt.ylabel('F1 score')
plt.legend()


# ax.set_xscale('log')
# ax.set_xticks([100,1000,10000,100000]) 

plt.show()




# Accuracy
# er = [(200, 91.7), (500, 94.75), (2000, 97.45), (5120, 98.53)]
# der = [(200, 93.14), (500, 94.18), (2000, 96.05), (5120, 96.43)]
# derpp = [(200, 93.36), (500, 93.67), (2000, 96.63), (5120, 97.51)]
# fig, ax = plt.subplots()

# plt.plot(*zip(*er), '-o', label = "ER")
# plt.plot(*zip(*der), '-o', label = "DER")
# plt.plot(*zip(*derpp), '-o', label = "DERPP")

# plt.title('GrainSpaceA - Accuracy vs buffer size')
# plt.xlabel('Buffer size')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()


# Epochs

# er = [(1, 92.62), (5, 94.72), (20, 96.64), (35, 97.08), (50, 97.45)]
# fig, ax = plt.subplots()

# plt.plot(*zip(*er), '-o', label = "ER")

# plt.title('GrainSpaceA - Accuracy vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()


# plt.show()

# er = [(1, 35.48), (5, 68.1), (20, 75.48), (35, 78.21), (50, 81.9)]
# fig, ax = plt.subplots()

# plt.plot(*zip(*er), '-o', label = "ER")

# plt.title('GrainSpaceA - F1 score vs Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('F1 score')
# plt.legend()


# plt.show()



# Accuracy
# er_A = [(200, 91.7), (500, 94.75), (2000, 97.45), (5120, 98.53)]
# er_B = [(200, 94.15), (500, 95.48), (2000, 97.08), (5120, 98.42)]
# # F1 score
# er_A_f = [(200, 35.25), (500, 52.2), (2000, 81.9), (5120, 90.11)]
# er_B_f = [(200, 44.39), (500, 58.03), (2000, 77.27), (5120, 88.18)]

# fig, ax = plt.subplots()

# plt.plot(*zip(*er_A), '-o', label = "GrainSpace A Accuracy", color="#2887c7")
# plt.plot(*zip(*er_B), '-o', label = "GrainSpace B Accuracy", color="#eb712f")
# plt.plot(*zip(*er_A_f), '-o', label = "GrainSpace A F1 score", color="#2887c7")
# plt.plot(*zip(*er_B_f), '-o', label = "GrainSpace B F1 score", color="#eb712f")


# plt.title('Accuracy of ER on GrainSpace A & B')
# plt.xlabel('Buffer size')
# plt.ylabel('Accuracy & F1 score')
# plt.legend()


# plt.show()

