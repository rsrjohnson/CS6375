Assingment 3 read me file.
Derek Carpenter / dxc190017
Randy Suarez Rodes / rxs179030


-python version 3.7

-All the code is on one file. You need the following libraries:

numpy, pandas, matplotlib.pyplot, nltk, urllib.request

-Please, read the comments and the report for better understanding.

-Data used: msnhealthnews.txt Hosted on: https://dxc190017cs6375.s3.amazonaws.com/msnhealthnews.txt?versionId=2gcJLPW8JUQ3O9wTNu5LSRKFWNsr825u

-A user can input desired options to generate an elbow plot and a table of different K values. Default values are suggested. If any unsupported value is entered, the value of that parameter will be set to a default. Also, an option to extract tweets from the clusters is provided


NOTE: The algorithm execution time depends on the number on the maximum number of clusters to select, be aware that for big values of K the execution time could be high.


--------------------------------------------------------------------------------------------------------------------
EXAMPLE RUN:
(base) c:\temp>python Assignment3_Part2.py
Preprocessing Tweets, please wait.
Type 'Y' if you would like to generate an elbow plot to select an appropiate range for the number of clusters
Y
Enter a maximum number of clusters for the elbow plot.
15
Generating elbow plot, please wait. Close the graph afterwards to proceed.

Type 'Y' if you want to enter custom values for the range of the Kmeans table, in any other case the algorithm will be executed with the default parameters.
The default range of K is [10-15].
Y
Enter a minimum number of clusters.
10
Enter a maximum number of clusters.
13

Generating table of different K(s), please wait.

Type 'Y' if you would like to extract tweets within same clusters
Y
Enter number of clusters
5
Enter number of tweets to extract per cluster
3
Centroid Tweet:
['iuds', 'may', 'lower', "women's", 'risk', 'for', 'cervical', 'cancer', 'study']
Tweets in cluster:
['heavy', 'coffee', 'intake', 'may', 'affect', 'fertility', 'treatments', 'study']
['no', 'health', 'risk', 'when', "jehovah's", 'witnesses', 'refuse', 'blood', 'study']
['coffee', 'may', 'cut', 'your', 'risk', 'for', 'common', 'form', 'of', 'skin', 'cancer']


Centroid Tweet:
['type', 'of', 'bacteria', 'may', 'be', 'linked', 'to', 'diabetes']
Tweets in cluster:
['summer', 'is', 'peak', 'time', 'for', 'teens', 'to', 'try', 'drugs', 'alcohol', 'report']
['more', 'genes', 'linked', 'to', 'osteoarthritis', 'identified']
['medicare', 'coverage', 'gap', 'may', 'cause', 'seniors', 'to', 'forgo', 'antidepressants']


Centroid Tweet:
['melanoma', 'may', 'be', 'more', 'aggressive', 'in', 'kids']
Tweets in cluster:
['some', 'crash-avoidance', 'systems', 'may', 'work', 'better', 'than', 'others']
['dangerous', 'rage', 'may', 'be', 'common', 'among', 'u.s', 'teens']
['botox', 'may', 'ease', 'tremors', 'in', 'multiple', 'sclerosis', 'patients']


Centroid Tweet:
['poor', 'sleep', 'may', 'impact', 'stress', 'response', 'in', 'older', 'adults']
Tweets in cluster:
['poor', 'sleep', 'affects', 'immune', 'system', 'much', 'like', 'physical', 'stress']
['sleep', 'can', 'sharpen', 'your', 'memory']
['violence', 'takes', 'a', 'toll', 'on', "children's", 'sleep']


Centroid Tweet:
['spouses', 'of', 'cancer', 'patients', 'may', 'have', 'raised', 'risk', 'of', 'heart', 'disease', 'stroke']
Tweets in cluster:
['maintain', 'heart', 'health', 'during', 'summer']
['smallest', 'largest', 'fetuses', 'at', 'higher', 'risk', 'of', 'stillbirth']
['early', 'surgery', 'may', 'benefit', 'some', 'with', 'heart', 'infection']


Press enter to exit


(base) c:\temp>
----------------------------------------------------------------------------------------------------------------------------------