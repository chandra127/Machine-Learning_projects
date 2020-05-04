import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random
from matplotlib import style

from collections import Counter
style.use('fivethirtyeight')

def K_nearst_neighbors(data, predict,k=3):
    if len(data)>=k:
        warnings.warn("k is set to a value leass than total voting groups ")
    distances=[]
    for group in data:
        for features in data[group]:
            Eulidian_distance= np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([Eulidian_distance, group])
    votes=[i[1] for i in sorted(distances)[:k]]
    #print(distances)
    #print(sorted(distances))
    #print(votes)
    #print(Counter(votes).most_common(1))
    vote_result= Counter(votes).most_common(1)[0][0]
    confidence= Counter(votes).most_common(1)[0][1] / k


    return vote_result,confidence

#using our algorithm for large dataset

accuracies=[]
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()

    #df = pd.read_csv('breast-cancer-wisconsin.data', na_values='?', header=0).fillna(-99999).drop('id', axis=1)

    #print(full_data[:10])
    #print(df.head())

    random.shuffle(full_data)

    #we are divinding data into train data and test data
    test_size=0.2
    train_set={2:[], 4:[]}
    test_set={2:[], 4:[]}
    train_data=full_data[:-int(test_size*len(full_data))] #80% of total data
    test_data=full_data[-int(test_size*len(full_data)):]
    #print(full_data)
    #print(train_data)

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    #print(train_set)

    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    #print(test_set)

    """1. for i in train data is every single list inside train_data (every list of 10 digits)
    2. i[-1] is the piece of data that is either 2 or 4 within that list 
    3. train_set[i[-1]] identifies whether the data will be appended to the 2 or 4 dictionary within the training set
    4. append(i[:-1]) adds everything up until the answer data to the training set"""


    correct=0
    total=0

    for group in test_set:
        for data in test_set[group]:
            vote,confidence=K_nearst_neighbors(train_set,data,k=20)
            #print(vote)
            if group==vote:
                correct += 1

            total += 1

    #print("Accuracy", correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))

#Example
"""dataset={'k':[[1,2], [2,3], [3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features=[5,7]

result=K_nearst_neighbors(dataset,new_features,k=3)
print(result)
[[ plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],color=result)
plt.show()"""
