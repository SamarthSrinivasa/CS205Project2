import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

data_path = '/Users/samarthsrinivasa/Documents/School/Classes/CS205/BasketBallStats/2021-2022 NBA Player Stats - Playoffs.csv'
data = pd.read_csv(data_path, delimiter=';', encoding='latin1')
print("Data loaded successfully. Here are the first few rows:")
print(data.head())

target = data['G']  

features = data.drop(['Rk', 'Player', 'Pos', 'Tm', 'G', 'Age', 'GS'], axis=1)

features = features.select_dtypes(include=[np.number])

feature_names = features.columns.tolist()

print("Features before normalization:")
print(features.describe())

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

data_normalized = np.column_stack((target, features_normalized))

print("First few rows of normalized data:")
print(data_normalized[:5])

dataLength = len(data_normalized) 

def leave_one_out_cross_validation(data, current_set):
    if len(current_set) == 0:
        return 0  
    X = data[:, [0] + [f + 1 for f in current_set]] 
    y = data[:, 0] 
    distances = pairwise_distances(X[:, 1:], X[:, 1:])  
    np.fill_diagonal(distances, np.inf)

    correct_classifications = 0

    for i in range(dataLength):
        nearest_neighbor_idx = np.argmin(distances[i])
        if y[i] == y[nearest_neighbor_idx]:
            correct_classifications += 1

    accuracy = correct_classifications / dataLength
    return accuracy

current_set_of_features = list(range(features_normalized.shape[1]))
best_so_far_accuracy = leave_one_out_cross_validation(data_normalized, current_set_of_features)
print("Initial set of features: ", current_set_of_features)
print("Initial accuracy: ", best_so_far_accuracy)

best_set = current_set_of_features.copy()
bestAcc = best_so_far_accuracy
lastAcc = best_so_far_accuracy

for i in range(1, len(current_set_of_features) + 1):
    print("On the " + str(i) + "th level of the search tree")

    best_so_far_accuracy = 0
    removableFeat = None

    for k in current_set_of_features:
        eliminationQueue = [f for f in current_set_of_features if f != k]

        back_accuracy = leave_one_out_cross_validation(data_normalized, eliminationQueue)
        
        if back_accuracy > best_so_far_accuracy:
            best_so_far_accuracy = back_accuracy
            removableFeat = k

    if removableFeat is not None:
        print("On level " + str(i) + " I removed feature " + str(removableFeat) + " to current set, accuracy: ", best_so_far_accuracy)
        current_set_of_features.remove(removableFeat)
    else:
        print(f"No removable feature found at level {i}")

    if best_so_far_accuracy > bestAcc:
        bestAcc = best_so_far_accuracy
        best_set = current_set_of_features.copy()
    if lastAcc > best_so_far_accuracy:
        print("Accuracy has decreased!\n")
    lastAcc = best_so_far_accuracy

    if len(current_set_of_features) == 0:
        break

print("This is the best set for backward elimination: ", best_set)

retained_features = [feature_names[i] for i in best_set]
print("Retained features: ", retained_features)
