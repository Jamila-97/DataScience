import random 
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import davies_bouldin_score
from collections import defaultdict


def numweekdays(day):
    weekdays = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return weekdays.index(day)+1

def nummonths(month):
    months = ['jan','feb','mar','may','apr','jun','jul','aug','sep','oct','nov','dec']
    return months.index(month)+1


def get_data(data, settype):
    
    X = pd.read_csv(data, header = 0)

    print('The data FORMAT is shown as below\n')
    print(X.head())
    X = X.values.tolist()
    if settype == 'Forest Fires':
        for entry in X: 
            entry[2] = nummonths(entry[2])
            entry[3] = numweekdays(entry[3])

    return X


def is_converged(centroids, old_centroids):
    return set([tuple(a) for a in centroids]) == set([tuple(b) for b in old_centroids])

# return int: euclidean distance
def get_distance(x, c, measuretype):
    """Depending on which distance is given return distance """
 
    if measuretype == 'Manhatten Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=1)
    
    elif measuretype == 'Euclidean Distance': 
        return np.linalg.norm(np.array(x)-np.array(c), ord=2)
    
    elif measuretype == 'L5 Distance':
        return np.linalg.norm(np.array(x)-np.array(c), ord=5)
    
    elif measuretype == 'Maximum Distance':
        list_distance = []
        for i in range(len(x)):
            list_distance += [abs(x[i]-c[i])]
        return max(list_distance)

# return: dictionary of lists 
def get_clusters(X, centroids, measuretype):
    clusters = defaultdict(list)
    for x in X:
        # cluster is a num to indicate the # of centroids
        cluster = np.argsort([get_distance(x[:-1], c[:-1], measuretype) for c in centroids])[0]

        clusters[cluster].append(x)
    return clusters

# return: list of np.lists 
def get_centeroids(old_centroids, clusters):
    new_centroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        if clusters[k]:
            new_centroid = np.mean(clusters[k], axis=0)
            # the class label is determined by majortity vote inside the cluster
            new_centroid[len(clusters[0][0])-1] = Counter([clusters[k][i][-1] for i in range(len(clusters[k]))]).most_common(1)[0][0]
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(old_centroids[k])
    return new_centroids

# return: tuple (centroids, clusters, iteration)
def find_centers(X, K, measuretype):
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)
    iteration = 0
    while not is_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = get_clusters(X, centroids, measuretype)
        centroids = get_centeroids(old_centroids, clusters)
        iteration += 1
    return (centroids, clusters, iteration)

# return purity score 
def get_purity(clusters, centroids, num_instances):
    counts = 0
    for k in clusters.keys():
        labels = np.array(clusters[k])[:, -1]
        counts += Counter(labels).most_common(1)[0][1]
    return float(counts)/num_instances


def kmeans(data, k, distance, output, settype):
   
    X = get_data(data, settype)
    num_instances = len(X)
    centroids, clusters , iteration= find_centers(X, k, distance)
    # store the best records in 5 iterations 
    best_score = 0
    best_centroids = []
    best_clusters =[]
    best_iteratoin = 0
    for i in range(5):
        centroids, clusters , iteration= find_centers(X, k, distance)
        purity = get_purity(clusters, centroids, num_instances)
        if purity > best_score:
            best_centroids = centroids
            best_clusters = clusters
            best_score = purity
            best_iteratoin = iteration
            

    centroids = []
    for c in best_centroids:
        c = c.tolist()
        centroids.append(c)
    best_centroids = centroids
    print('The best purity score is %f' % best_score)
    print('It takes %d number of iterations' % best_iteratoin)
    with open(output, 'w') as out:
        for k in best_clusters.keys():
            out.write('The %d centroid is \n%s\n\n' % (k, best_centroids[k]))
            out.write('It has following points: \n')
            for pt in clusters[k]:
                out.write('%s\n' % pt)
            out.write('\n\n\n\n')
    get_labels(X, clusters)
    
def get_labels(X, clusters):
    labels = []
    for x in X:
        # find in which cluster x is
        for keys in list(clusters.keys()):
            if x in clusters[keys]:
                labels += [keys]
                break
    return labels

def davis_bouldin(X, labels):
    return sklearn.metrics.davies_bouldin_score(X, labels)

dataset = input("Chose Dataset by number: \n 1. Wholesale customers, 2. Wine, 3. Forest Fires, 4. Heart failure clinical records \n")
distance = input("Choose distancemeasure by number:\n 1. Manhatten Distance, 2.Euclidean Distance, 3.L5 Distance, 4.Maximum Distance \n")
k = input("Choose k \n")

if distance == '1':
    distancemeasure = 'Manhatten Distance'
elif distance == '2':
    distancemeasure = 'Euclidean Distance'
elif distance == '3':
    distancemeasure = 'L5 Distance'
elif distance == '4':
    distancemeasure = 'Maximum Distance'
    
if dataset == '1':
    dataset = 'Wholesale customers'
elif dataset == '2':
    dataset = 'Wine'
elif dataset == '3':
    dataset = 'Forest Fires'
elif dataset == '4':
    dataset = 'Heart failure clinical records'
print(dataset)

k = int(k)
if dataset == 'Wholesale customers':
    thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wholesale.csv'
    kmeans(thedata,k,distancemeasure, 'wholesale.out', dataset)
elif dataset == 'Wine':
    thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/wine_data.csv'
    kmeans(thedata,k,distancemeasure, 'wine_data.out', dataset)
elif dataset == 'Forest Fires':
    thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/forestfires.csv'
    kmeans(thedata,k,distancemeasure, 'forestfire.out', dataset)
elif dataset == 'Heart failure clinical records':
    thedata = 'https://raw.githubusercontent.com/Chantalkle/DataScience/main/heart_failure.csv'
    kmeans(thedata,k,distancemeasure, 'heart_failure.out', dataset)
