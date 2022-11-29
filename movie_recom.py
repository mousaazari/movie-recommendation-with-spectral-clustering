import csv
import pickle
from collections import Counter
from itertools import chain
from sklearn.cluster import SpectralClustering
import numpy as np

def find_similarity_pearson(user_item_rating):
    simmat = np.zeros([len(movies.keys()), len(movies.keys())])
    for st, m1 in enumerate(movies.keys()):
        if st % 1000 == 0:
            r1 = np.average(user_item_rating[:, movies[m1]])
            u_m1 = np.where(user_item_rating[:, movies[m1]] != 0)

        for j in range(st, len(movies.keys())):
            m2 = movies.keys()[j]
            r2 = np.average(user_item_rating[:, movies[m2]])
            u_m2 = np.where(user_item_rating[:, movies[m2]] != 0)
            u = list(set(u_m1[0]).intersection(set(u_m2[0])))
            # Pearson similarity
            if len(u) != 0:
                co_ratings = user_item_rating[np.ix_(u, [int(movies[m1]), int(movies[m2])])]
                num = sum((co_ratings[:, 0] - r1) * (co_ratings[:, 1] - r2))
                den = ((sum((co_ratings[:, 0] - r1) ** 2)) ** 0.5) * ((sum((co_ratings[:, 1] - r2) ** 2)) ** 0.5)
                corr = num * 1.0 / den
                simmat[st][j] = corr
                if j != st:
                    simmat[j][st] = corr

    return (simmat)



movies = {}
tags = {}
with open('movies.csv', newline='',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        try:
            movies[int(row[0])]=[row[1],row[2],[]]
        except:
            print(row[0])
with open('tags.csv', newline='',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[2] not in tags:
            try:
                tags[row[2]] = [int(row[1])]
            except:
                print(row[1])
        else:
            try:
                tags[row[2]].append(int(row[1]))
            except:
                print(row[1])
count = 0
temp = {}
for i, j in tags.items():
    if len(j) < 10:
        count +=1
    else:
        temp[i] = j
tags = temp

for i, j in tags.items():
    for k in j:
        if i not in movies[k]:
            movies[k].append(i)
del tags
ids = {}
simmat = []
deletedrows=[]
for i, j in movies.items():
    if len(j)<13:
        deletedrows.append(i)
for i in deletedrows:
    del movies[i]
print(len(movies))
count=0
for i, j in movies.items():
    ids[i] = count
    tuple = [0 for z in range(len(movies))]
    for x in j:
        index = 0
        for k, l in movies.items():
            if x in l and x != []:
                tuple[index]+=1
            index+=1
    simmat.append(tuple)
    print(tuple)
    count += 1
    print(count)

with open('simmat.txt', 'wb') as fp:
    pickle.dump(simmat, fp)

user_item_rating = {}
with open('ratings.csv', newline='',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        try:
            if int(row[0]) in user_item_rating and int(row[1]) in ids:
                user_item_rating[int(row[0])].append([int(row[1]),float(row[2])])
            elif int(row[0]) not in user_item_rating and int(row[1]) in ids:
                user_item_rating[int(row[0])] = [[int(row[1]),float(row[2])]]
        except:
            print(row[0], ' ', row[1], ' ', row[2])

scl = SpectralClustering(assign_labels="discretize",random_state=1,affinity='precomputed')
clusters = scl.fit_predict(simmat)
print(clusters)

with open('clusters.txt', 'wb') as fp:
    pickle.dump(clusters, fp)
with open('movies.txt', 'wb') as fp:
    pickle.dump(movies, fp)
with open('ids.txt', 'wb') as fp:
    pickle.dump(ids, fp)


idsreversed = {}
for i,j in ids.items():
    idsreversed[j]=i
with open('ratings.txt', 'wb') as fp:
    pickle.dump(user_item_rating , fp)
with open('idsreversed.txt', 'wb') as fp:
    pickle.dump(idsreversed,fp)

with open('ratings.txt', 'rb') as handle:
    user_item_rating = np.array(pickle.load(handle))
with open('clusters.txt', 'rb') as handle:
    clusters = pickle.load(handle)
with open('movies.txt', 'rb') as handle:
    movies = pickle.load(handle)
with open('simmat.txt', 'rb') as handle:
    simmat = np.array(pickle.load(handle))
with open('ids.txt', 'rb') as handle:
    ids = pickle.load(handle)
with open('idsreversed.txt', 'rb') as handle:
    idsreversed = pickle.load(handle)

count = 0
clusternum = 8
accuracies = []
for i,j in user_item_rating.items():
    print('new user')
    firstpart = []
    types = []
    mostcommon = []
    mostcommonelements=[]
    predicted=[]
    count2 = 0
    count3 = 0
    count4 = 0
    for k in j:
        if count2<int(2*len(j)/3):
            firstpart.append(k)
            clusterNo = clusters[ids[k[0]]]
            tuple = []
            for m in range(10):
                max = -1
                maxindex = -1
                for l in range(len(simmat)):
                    if simmat[ids[k[0]]][l] > max and clusters[l] == clusterNo and idsreversed[l] not in tuple and idsreversed[l] not in firstpart:
                        max = simmat[ids[k[0]]][l]
                        maxindex = idsreversed[l]
                tuple.append(maxindex)
            types.append(tuple)

        if count2==int(2*len(j)/3):
            mostcommon = Counter(chain.from_iterable(types))
            mostcommonelements = list(mostcommon.most_common(int(len(j)/3)))
            for p in mostcommonelements:
                predicted.append(p[0])
            print(predicted)

        if count2>=int(2*len(j)/3):
            if k[0] in predicted:
                print('Hit!')
                count3+=1
            else:
                print('Miss!')
                count4+=1
        count2+=1
    print(count3/(count4+count3))
    accuracies.append(count3/(count4+count3))
    print()
    count+=1
with open('accuracies1.txt', 'wb') as fp:
    pickle.dump(accuracies,fp)

rmse = []

for i,j in user_item_rating.items():
    print('new user')
    tuple = []
    for k in j:
        weight = 0
        rating = 0
        clusterNo = clusters[ids[k[0]]]
        for l in j:
            if k[0]!=l[0] and clusters[ids[l[0]]]==clusterNo:
                rating += simmat[ids[k[0]]][ids[l[0]]]*l[1]
                weight += simmat[ids[k[0]]][ids[l[0]]]
        if weight != 0:
            predicted_rating = rating/weight
            print('predicted: ',predicted_rating)
            print('true: ',k[1])
            tuple.append((k[1]-predicted_rating)**2)
        else:
            print("no similar movies are watched")
            print('true: ', k[1])
    rmse.append(tuple)

with open('accuracies2.txt', 'wb') as fp:
    pickle.dump(accuracies,fp)
