import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms


dataset = np.loadtxt("wine_quality\winequality-red.csv", delimiter=",", skiprows=1)

# range
for i in range(dataset.shape[1]):
    print(f"Column {i}, min:{dataset[:,i].min()}, max:{dataset[:,i].max()}, mean: {np.mean(dataset[:,i])}")

# correlation matrix
print(np.mean(dataset, axis=0))
corr_matrix = np.corrcoef(dataset.T)

corr2 = abs(corr_matrix)

p1 = plt.figure()
heatmap = sns.heatmap(corr2, annot=True)

#plt.show()

means = np.mean(dataset, axis=0)
stds = np.std(dataset, axis=0)
normalized = (dataset-means)/stds

corr_matrix2 = np.corrcoef(normalized.T)
p2 = plt.figure()
heatmap = sns.heatmap(corr_matrix2, annot=True)

p1.show()
p2.show()

#val = input("Continue")

'''
not correlated with output: 3, 5, 8
mutually correlated: 0 -> 2, 7, 8
eliminate: 3, 5, 8, 0, 7
'''
selected = np.zeros(shape=(dataset.shape[0], 7))

forRemoval = [0, 3, 5, 7, 8]

j=0
for i in range(0, 12):
    if i not in forRemoval:
        selected[:,j] = normalized[:,i]
        j += 1

#val = input("Continue")

X, y = selected[:,:-1], dataset[:, -1]

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.8, random_state=23)

def exportDatasets():
    dataset = np.loadtxt("wine_quality\winequality-red.csv", delimiter=",", skiprows=1)
    selected = np.zeros(shape=(dataset.shape[0], 6))
    means = np.mean(dataset, axis=0)
    stds = np.std(dataset, axis=0)
    normalized = (dataset-means)/stds

    forRemoval = [0, 2, 3, 5, 7, 8]

    j=0
    for i in range(0, 12):
        if i not in forRemoval:
            selected[:,j] = normalized[:,i]
            j += 1
    
    X, y = selected[:,:-1], dataset[:, -1]

    sets = ms.train_test_split(X, y, train_size=0.8, random_state=23)
    return sets

sets = exportDatasets()

#input("Press any key to exit")
x=1