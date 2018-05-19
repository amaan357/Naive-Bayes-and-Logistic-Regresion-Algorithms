# Naive-Bayes-and-Logistic-Regresion-Algorithms
Implemented Naive-Bayes-and-Logistic-Regresion-Algorithms from scratch using pythom for email spam-ham classification

The modules used are re, numpy and pandas

Make sure to place the train and test folders in the same directory as the program before running it.
The input folder names have been hard coded in the program so keep them as train and test only.

To run program, type below lines in terminal with appropriate inputs:
naive bayes : python nb.py
logistic regression : python -W ignore lr.py <lambda> <learning_rate> <number_of_iterations> 
#(-W ignore) ignores the runtime warning produced by the sigmoid function

LR program can be run with or without the arguments, if no arguments are given then default values will be used
lambda=1, learning rate=0.05, n=50

eg:python lr.py

   python lr.py 1
   
   python lr.py 2 0.01
   
   python lr.py 3 0.04 100
