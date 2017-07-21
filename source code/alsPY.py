
# Implementation of Movie recommendation system on spark
from __future__ import print_function

import sys
import itertools
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkConf, SparkContext
from math import sqrt
from operator import add
from os.path import join, isfile, dirname


def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        # print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print ("No ratings provided.")
        sys.exit(1)
    else:
        return ratings

LAMBDA = 0.001   # regularization
np.random.seed(42)

# function for Calculating the rmse
def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

# function for computing the values and updating them
def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]
    #  print (ratings)
    #print ((mat.T).shape)
    #print ((ratings[i, :].T).shape)

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


	
	
	
if __name__ == "__main__":

    print("Running Movie Recommendation system")

    conf = SparkConf().setAppName("MovieLensALS").set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)
    
    # load ratings and movie titles

    movieLensHomeDir = "/movie1"

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.txt")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.txt")).map(parseMovie).collect())
    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    numPartitions = 4
    training = ratings.values().repartition(numPartitions).cache()
    a=[]
    mat=np.zeros(22384240)
    
    a_list =training.collect()
    a_array = np.array(a_list)
    rows, row_pos = np.unique(a_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(a_array[:, 1], return_inverse=True)
    pivot_table = np.zeros((len(rows), len(cols)), dtype=a_array.dtype)
    pivot_table[row_pos, col_pos] = a_array[:, 2]

    t_matrix = np.matrix(pivot_table)
		
    M =  6040 # number of users
    U =  3706 # number of movies
    F =  20
    ITERATIONS = 20
    partitions = 5

    R = t_matrix # Rating matrix
    W = R>0.5 # Initializing the weighted Matrix
    W[W == True] = 1
    W[W == False]= 0
    W = W.astype(np.float64, copy=False)
    # Initializing the Factors
    ms = matrix(rand(M, F)) 
    us = matrix(rand(U, F))
    # Broadcasting the Matrices
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        # parallelizing the computation
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect()
        
        # arranging it into a matrix 
        ms = matrix(np.array(ms)[:, :, 0])
        # Broadcasting the matrix
        msb = sc.broadcast(ms)
        # parallelizing the computation
        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        # arranging into a matrix form
        us = matrix(np.array(us)[:, :, 0])
        # Broadcasting the matrix
        usb = sc.broadcast(us)
        # getting the error rate
        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)
    #print (R)
    #spark.stop()
    recommendation=np.dot(ms,us.T)
    #print(recommendation)
    #print(recommendation.shape)
    output=open("output.txt",'w')
    check=0
    #Giving Recommendations 
    for i in range(M):
        check=0
        string_recommend="-------------------Recommendations for user"+"  "+str(i)+"--------------------------"+"\n"
        for j in range(U):
            if((W[i,j]==0 and recommendation[i,j]>3 and recommendation[i,j]<=5)):
		try:
                   a=movies[j]
                   #output.write("-----------------------Recommendations for user"+" "+str(i)+"------------------------"+"\n")
		   check=1
                   string_recommend=string_recommend+"Movie title"+" :  "+movies[j]+","+"Movie ID"+" :  "+str(j)+","+"Predicted Rating: "+str(recommendation[i,j])+"\n"
                except:
                   print("Exception")
	final_string=string_recommend.encode('utf-8')
        if check==1:
            output.write(final_string)
