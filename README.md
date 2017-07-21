# Movie-Recommendation-system
A Simple Movie Recommendation system implementation without using any Machine learning Library.
This is a implementation of a Movie recommendation system without the use of the machine learning libraries.
The algorithm used to build the system is Alternating Least square Algorithm with weighted-lambda-regularization.
Movie Recommendations to each user is generated in results.


create a folder named "movie1" in the hadoop file system.
place the input files into them.
command to place the files in the HDFS


hadoop fs - put filename hdfs-file-path



Then run the alsPY.py file

command

spark-submit alsPY.py

after finishing executing the output.txt is generated where the recommendations are written.
