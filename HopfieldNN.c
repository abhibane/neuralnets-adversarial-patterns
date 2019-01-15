
import numpy as np

MAX = 300 
SIZE = 784 
MAXIT = 1000 
BLACK = 0 
WHITE = 1 
error = np.e**(-3)
lamda = 0 
activityLevel = 0 
N = 784;
p = 50;

w=[SIZE][SIZE]
weight=[SIZE][SIZE]
test_pattern=[MAX][SIZE]
recalloutput=[MAX][SIZE]
trg_pattern=[MAX][SIZE]
pat=[SIZE]
change=[SIZE]

def read_trg_set():
    trg_pattern = np.loadtxt('patternset1to5.txt')
    for j in range(p):
        for i in range (N):
            if (trg_pattern[j][i]==1):
                activityLevel = activityLevel + 1
    activityLevel = activityLevel / (N*p)

def initialise_wts():
    for i in range(N):
        w[i][i]=0
        change[i=0]
        for j in range(i+1, N):
            w[i][j]=0
            for k in range (p):
                w[i][j]  = w[i][j] + trg_pattern[k][i] * trg_pattern[k][j]
            w[i][j]=w[i][j]/N;
            w[j][i]=w[i][j];
            w[i][i]=0;

def correlate()
    tot_corr=0.0 
    for i in range(p):
        for j in range(i+1,p):
            corr = 0.0 
            for k in range(N):
                corr=corr + trg_pattern[i][k] * trg_pattern[j][k];
            tot_corr=tot_corr + N;
    tot_corr=tot_corr / (1.0 * N * p * p);
	 print("\n The average overlap is",tot_corr);
	 print("\n");        
	




