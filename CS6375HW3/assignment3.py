import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import re
from nltk.tokenize import word_tokenize, RegexpTokenizer

import urllib.request

rdseed=8466 #random seed to replicate results

#Class to carry out the Kmeans Cluster Algorithm.
#It receives a list of preprocessed Tweets
class KmeansCluster:

    def __init__(self, tweets):

        self.tweets = tweets
        self.sample_size = len(tweets)

        #Calculating distances using Jaccard Distance
        self.distances = self.calculateDistances()  

        #Structures to store Kmeans results
        self.centroids=[]
        self.clusters=[]
        self.clusters_sizes=0


    #Function to calculate the Jaccard Distance between two tweets
    def jaccardDist(self, t1, t2):
        A = set(t1)
        B = set(t2)
        return 1 - len(A & B) / len(A | B)

    #Function to find the Jaccard Distance between all tweets.
    #It return a list of list where the i'th list stores the distance between tweet i and all tweets.
    def calculateDistances(self):
        dists = []
        #To save space we stored d(tweet i, tweet j) only on the i'th list with index i<= index j
        for i in range(self.sample_size):
            dij = []
            ti = self.tweets[i]
            for j in range(i, self.sample_size):
                dij.append(self.jaccardDist(ti, self.tweets[j]))
            dists.append(dij)
        return dists
    
    #Function to read the pre-calculated Jaccard Distance from the distance list
    def findDist(self, t1, t2):
        if t1 > t2: #we can find d(j,i) on d(i,j)
            return self.findDist(t2, t1)
        t2-=t1
        return self.distances[t1][t2]

    #Function to assign tweets to the closer centroid
    def assign(self,centroids, clusters):
        #note that in the case the current tweet is disimilar to all centroids, we firt will assign it to a random cluster
        np.random.seed(rdseed) #fixing seed to replicate results
        center = np.random.choice(centroids, 1)[0]

        #looping over all tweets
        for t in range(self.sample_size):
            distCenter = float('inf')           
            
            #looping over the centroids
            for centroid in centroids:
                d_tweet_centroid = self.findDist(t, centroid)

                #If the tweet is not disimilar to the current centroid (d<1) and the current distance is less than the least distance so far
                #then we update the closest centroid
                if d_tweet_centroid<1 and d_tweet_centroid < distCenter:
                    distCenter = d_tweet_centroid
                    center = centroid #updating the closer centroid so far

            clusters[center].append(t) #assigning tweet t to the closer centroid's cluster

    #K means algorithm, it receives the number of desired clusters k
    def KMeans(self, k):
        
        np.random.seed(rdseed) #fixing seed to replicate results
        centroids = np.random.choice(self.sample_size, k, replace=False) #initializing k random centroids
        changes = True

        while (changes):
            changes = False
            clusters = defaultdict(list)
            self.assign(centroids, clusters) #assigning tweets to the cluster of the closer centroid

            #Update process
            for centroid in clusters:
                currCentroid = centroid
                minDist = float('inf')
                for posibleCenter in clusters[centroid]:
                    dist = 0
                    
                    #finding the element of a cluster with minimum sum of distances, in order to update the centroid 
                    for element in clusters[centroid]:
                        dist +=self.findDist(posibleCenter,element)
                    if dist < minDist:
                        currCentroid = posibleCenter
                        minDist=dist
                if currCentroid != centroid:
                    changes = True #if we updated the centroid we continue

                    #updating centroids array
                    centroids=np.delete(centroids, np.where(centroids==centroid))
                    centroids=np.append(centroids,currCentroid)

        self.centroids=centroids
        self.clusters=clusters
        self.clusters_sizes=[len(c) for c in clusters.values()]
        return centroids, clusters, self.clusters_sizes

    #Function to find the sum of squared errors
    def SSE(self):
        err=0
        for centroid in self.clusters:
            for member in self.clusters[centroid]:
                d=self.findDist(centroid,member)
                err+=d*d
        return err

#Function to preprocess tweets as pointed on the assignment instructions
def pre_processing(path):
    target_url = path
    data = urllib.request.urlopen(target_url)
    tweets = list()

    for line in data:
        try:
                line = line.decode()
                line = line.split('|')[2]
                line = re.sub(r'http[\S]*\s', '', line)

                tokenizer = RegexpTokenizer(r'\b(?<!@)\S+\b')
                line = tokenizer.tokenize(line.lower())

                tweets.append(line)
        except:
                continue

    return tweets

#Function to generate an elbow plot given a max number of k
def elbow_plot(Kclust_tweets,max_k):
    errors=[]
    for i in range(max_k):
        Kclust_tweets.KMeans(i+1)
        errors.append(Kclust_tweets.SSE())

    #Elbow Plot
    fig = plt.figure(figsize=(10, 10))
    x=range(1,max_k+1)
    plt.plot(x,errors,'-',x,errors,'ro')
    plt.title('SSE vs Number of Clusters')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.show()
    
#Function to get user options
def getOptions(Kclust_tweets):
    print("Type 'Y' if you would like to generate an elbow plot to select an appropiate range for the number of clusters")

    if input() == 'Y':
        print("Enter a maximum number of clusters for the elbow plot.")
        try:
            elbow_k = int(input())
            print('Generating elbow plot, please wait. Close the graph afterwards to proceed.')
            elbow_plot(Kclust_tweets,elbow_k)
        except ValueError:
            print('The input value is not valid, the elbow plot will not be generated.')
    
    print()
    print("Type 'Y' if you want to enter custom values for the range of the Kmeans table, in any other case the algorithm will be executed with the default parameters.")
    print("The default range of K is [10-15].")

    if input() != 'Y':
        return 10, 15
    else:
        print("Enter a minimum number of clusters.")
        try:
            min_k = int(input())            
        except ValueError:
            print('The input value is not valid, minimum number of clusters set to 12.')
            min_k=10

        print('Enter a maximum number of clusters.')
        try:
            max_k = int(input()) 
            if min_k>max_k:
                raise ValueError           
        except ValueError:
            print('The input value is not valid, maximum number of clusters set to have 5 consecutive number of K(s).')
            max_k=min_k+5


        return min_k,max_k

#Function to extract a desired number of tweets from the clusters
def extract_tweets(Kclust_tweets, k, num_tweets_per_clust):
    tweets = list()
    clust_tweets = list()
    Kmeans = Kclust_tweets.KMeans(k)
    for centroid in Kmeans[0]:
        centroid_tweet = Kclust_tweets.tweets[centroid]
        clust_tweets.append(centroid_tweet)
        print('Centroid Tweet: ')
        print(centroid_tweet)
        print('Tweets in cluster:')
        for i in range(0, num_tweets_per_clust):
            if len(Kmeans[1][centroid]) >= num_tweets_per_clust:
                extracted_tweet = Kclust_tweets.tweets[Kmeans[1][centroid][i]]
                clust_tweets.append(extracted_tweet)
                print(extracted_tweet)
        tweets.append(clust_tweets.copy())
        clust_tweets.clear()
        print('\n')
    return tweets




path='https://dxc190017cs6375.s3.amazonaws.com/msnhealthnews.txt?versionId=2gcJLPW8JUQ3O9wTNu5LSRKFWNsr825u'
print('Preprocessing Tweets, please wait.')
tweets=pre_processing(path)

Kclust_tweets=KmeansCluster(tweets)


min_k,max_k=getOptions(Kclust_tweets)

print()
print('Generating table of different K(s), please wait.')

num_k=range(min_k,max_k+1)
cluster_sizes=[]
SSEs=[]

for k in num_k:
    centroids, clusters, sizes=Kclust_tweets.KMeans(k)
    SSEs.append(round(Kclust_tweets.SSE(),3))
    cluster_sizes.append([str(i)+': '+str(sizes[i-1]) for i in range(1,k+1)])


d={'Value of K':num_k,'SSE':SSEs,'Number of Tweets per Cluster (Cluster: Tweets)':cluster_sizes}

df = pd.DataFrame(data=d)

#Displaying table of different k values
fig = plt.figure(figsize=(20, 20))
ax = plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=df.values, colColours=['grey'] * df.shape[1], bbox=[0, 0, 1, 1],
                colWidths = [.06, .06, .88],colLabels=df.columns, cellLoc='left',rowLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.show()

print()
print("Type 'Y' if you would like to extract tweets within same clusters")

if input() == 'Y':
    print('Enter number of clusters')
    try:
        k = int(input())

        print('Enter number of tweets to extract per cluster')
        try:
            t = int(input())
            extract_tweets(Kclust_tweets, k, t)
        except ValueError:
            print('The input values are not valid')
    except ValueError:
        print('The input values are not valid')

print('Press enter to exit')
input()