import math
import random
import numpy as np
from copy import deepcopy

data1 = []
with open("CS205_small_Data__7.txt", "r") as file:
    for line in file:
        data1.append(line.split())

data2 = np.array(data1)
data = data2.astype(float)

columnLength = len(data)
rowLength = len(data[0])

current_set_of_features = []
best_set = []

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0

    for i in range(columnLength):
    #object_to_classify = df2.iloc[i]

        object_to_classify = data[i][1:]
        label_object_to_classify = data[i][0]
        #print(label_object_to_classify)

        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = math.inf
        nearest_neighbor_label = math.inf

        for k in range(columnLength):
            sets_to_add = []

            if feature_to_add != 0:
                sets_to_add.append(data[i][feature_to_add] - data[k][feature_to_add])
            for j in range(len(current_set)):
                sets_to_add.append(data[i][current_set[j]] - data[k][current_set[j]])

        #print("Ask if " + str(i) + " is nearest neighbors with " + str(k))
            if k != i and sets_to_add:

                sets_to_add = np.square(sets_to_add)
                distance = np.sqrt(sum((sets_to_add)))

                #distance = math.sqrt(sum((sets_to_add)**2))
                    #print(distance)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
                    #print(nearest_neighbor_label)
           
    #print("Object " + str(i) + " is class " + str(label_object_to_classify))    
        if (label_object_to_classify == nearest_neighbor_label and nearest_neighbor_label != math.inf):
            number_correctly_classified += 1

    accuracy = number_correctly_classified/columnLength 
    #print(y)
    return accuracy 

bestAcc = 0
lastAcc = 0

for i in range(1, rowLength):
    print("Current Set of Features: ", current_set_of_features)
    #print("On the " + str(i) + "th level of the search tree")
    #feature_to_add_at_this_level = 0

    best_so_far_accuracy = 0

    for k in range(1, rowLength):
        accuracy = 0
        if k not in current_set_of_features:
            #print("--Considering adding the " + str(k) + " feature")
            accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
            #print("Accuracy of Features ", current_set_of_features, " with ", k, "added:", accuracy)
        if accuracy > best_so_far_accuracy:
            best_so_far_accuracy = accuracy
            feature_to_add_at_this_level = k
            #print(best_so_far_accuracy) 
            #print(current_set_of_features)           

    #current_set_of_features[i] = feature_to_add_at_this_level
    #if feature_to_add_at_this_level != 0:
    current_set_of_features.append(feature_to_add_at_this_level)

    print("On level " + str(i) + " I added feature " + str(feature_to_add_at_this_level) + " to current set, accuracy: ", best_so_far_accuracy)    

    if best_so_far_accuracy > bestAcc:
        bestAcc = best_so_far_accuracy
        best_set = deepcopy(current_set_of_features)
    if lastAcc > best_so_far_accuracy:
        print("Accuracy has decreased!\n")
    lastAcc = best_so_far_accuracy

print("This is the best set for forward selection: ", best_set)


current_set_of_features = []
best_set = []

bestAcc = 0
lastAcc = 0

for i in range(1, rowLength):
    current_set_of_features.append(i)

for i in range(1, rowLength):

    print("Current Set of Features: ", current_set_of_features)
    print("On the " + str(i) + "th level of the search tree")

    best_so_far_accuracy = 0
    removableFeat = 0

    for k in current_set_of_features:

        eliminationQueue = deepcopy(current_set_of_features)
        eliminationQueue.remove(k)

        back_accuracy = leave_one_out_cross_validation(data, eliminationQueue, 0)
        
        #if k not in current_set_of_features:
            #print("--Considering adding the " + str(k) + " feature")
            #accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
            #print("Accuracy of Features ", current_set_of_features, " with ", k, "added:", accuracy)
        if back_accuracy > best_so_far_accuracy:
            best_so_far_accuracy = back_accuracy
            removableFeat = k
            #print(best_so_far_accuracy) 
            #print(current_set_of_features)           

    #current_set_of_features[i] = feature_to_add_at_this_level
    #if feature_to_add_at_this_level != 0:
    #current_set_of_features.append(feature_to_add_at_this_level)

    print("On level " + str(i) + " I removed feature " + str(removableFeat) + " to current set, accuracy: ", best_so_far_accuracy)    

    #if current_set_of_features: 
    #    current_set_of_features.remove(removableFeat)
    if removableFeat in current_set_of_features:
        current_set_of_features.remove(removableFeat)
    else:
        print(f"Error: Feature {removableFeat} is not in the current set of features.")


    if best_so_far_accuracy > bestAcc:
        bestAcc = best_so_far_accuracy
        best_set = deepcopy(current_set_of_features)
    if lastAcc > best_so_far_accuracy:
        print("Accuracy has decreased!\n")
    lastAcc = best_so_far_accuracy


print("This is the best set for backwards elimination: ", best_set)
