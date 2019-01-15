
import numpy as np
import random 
import matplotlib.pyplot as plt
import math

MAX = 10 
SIZE = 784 
MAXIT = 1000
BLACK = 0 
WHITE = 1 
error = np.e**(-3)
lamda  
activityLevel = 0 
N = 784;
p =10 ;

w=np.zeros((SIZE,SIZE))
weight=np.zeros((SIZE,SIZE))
test_pattern=np.zeros((MAX,SIZE))
recalloutput=np.zeros((MAX,SIZE))
trg_pattern=np.zeros((MAX,SIZE))
pat=np.zeros(SIZE)
change=np.zeros(SIZE)

def read_trg_set():
    global activityLevel
    global trg_pattern
    trg_pattern = np.loadtxt('patternset1to5.txt')
    for j in range(p):
        for i in range (N):
            if (trg_pattern[j][i]==1):
                activityLevel = activityLevel + 1
    activityLevel = activityLevel / (N*p)
    print("The activity level is", activityLevel)
    
def initialise_wts():
    global w
    global trg_pattern
    for i in range(N):
        w[i][i] = 0
        change[i] = 0
        for j in range(i+1, N):
            w[i][j] = 0
            for k in range (p):
                w[i][j]  = w[i][j] + trg_pattern[k][i] * trg_pattern[k][j]
            w[i][j]=w[i][j]/N
            w[j][i]=w[i][j]
            w[i][i]=0
    


def correlate():
    global trg_pattern
    tot_corr=0.0 
    for i in range(p):
        for j in range(i+1,p):
            corr = 0.0 
            for k in range(N):
                corr=corr + trg_pattern[i][k] * trg_pattern[j][k]
            tot_corr = tot_corr + 2 * corr
        tot_corr = tot_corr + N 
    tot_corr=tot_corr / (1.0 * N * p * p)
    print("\n The average overlap is",tot_corr)



def write_hebb_wt():
    global w 
    #mat = np.matrix(w) 
    np.savetxt("nistpatterhebb.txt",w,fmt='%.2f')
    
       
def display():
    global trg_pattern
    for i in range(p) : 
        img = trg_pattern[i].reshape(28,28)
        plt.imshow(img, cmap="Greys")
        plt.show()
    
def phase1(output , N , change , energy_val) : 
    already_minima = True 
    changes = 0
    nitr = list(range(N)) 
    random.shuffle(nitr)
    for j in range(N):
        index = random.randint(0,32767) % N  
        difference = delta (output, index , N )
        if (difference < 0):
            change[index] = change[index] + 1 
            already_minima = False 
            energy_val = difference 
            output [index] = output[index] * -1 
            changes = changes + 1 
    return (already_minima,energy_val,change,output)
        
def phase2(output , N , change , energy_val) : 
    already_minima = True 
    changes = 0 
    
    for j in range(N):        
        difference = delta (output, j , N )
        if (difference < 0):
            change[j] = change[j] + 1 
            output [j] = output[j] * -1
            already_minima = False 
            energy_val = difference 
            changes = changes + 1 
    return (already_minima,energy_val,change,output)
              
def delta(states, i, N ) : 
    add = 0.0 
    global w
    for j in range(N):
        add = add + w[i][j] * states[j]
    return (2 * add * states[i])

def energy(states,N):
    global w
    add = 0.0 
    itr = 0 
    for i in range (N):
        for j in range (i+1,N):
            add = add - w[i][j] * states[i] * states[j]
            itr = itr + 1
        #print(add)
    return add 

def relax(output , N , change):
    global error 
    flag = False
    iterations = 0 
    start = energy(output, N)
    print("Starting Energy Value =" , start)
    energy_val = start 
    while(flag == False and iterations < MAXIT):
        flag1,energy_val,change,output = phase1(output, N, change, energy_val)
        flag2,energy_val,change,output = phase2(output, N, change, energy_val)
        flag3,energy_val,change,output = phase1(output, N, change, energy_val)
        flag = flag1 and flag2 and flag3 
        iterations = iterations + 1 
    difference = start - energy_val 
    if (math.fabs(difference) < error) : 
        return True 
    else:
        return False 

def change_wts(old, new, N , lamda) : 
    global w
    for i in range(N):
        for j in range(i+1 , N ):
            w[i][j] = w[i][j] - lamda * (new[i]*new[j]-old[i]*old[j])
            w[j][i] = w[i][j]
    
def Train_SG():
    i=0
    global lamda
    global trg_pattern
    convergence = False 
    pat_conv = True 
    sweep = 0 
    carry = 0 
    new = [SIZE]


    while (convergence == False  and  sweep < MAXIT):
        new = np.copy(trg_pattern[i])
        pat_conv = relax(new,N,change)
        carry = carry and pat_conv
        if (pat_conv == False ):
            change_wts(trg_pattern[i],new,N, lamda)
        i = (i + 1) % p
        if (i==0):
            convergence = carry 
            carry = True 
            sweep = sweep + 1 
            #print("THE SWEEP # ", sweep)
            if (sweep % 100 == 0) : 
                lamda = lamda / 2 
            print("THE SWEEP # ", sweep)
            if(sweep == 7 and sweep ==9 and sweep ==11):
                write_hebb_wt(N,p,change, lamda)
 
      
def read_test_set():
    global test_pattern
    activity = 0 
    test_pattern = np.loadtxt('patternset1to5.txt')
    for j in range(p):
        for i in range (N):
            if (test_pattern[j][i]==1):
                activity = activity + 1
    activity = activity / (N*p) 
    print("The acitivity level of test pattern is ", activity)       

def recallnew():
    global recalloutput 
    global test_pattern
    global w
    temp = 0.0
    for k in range (p):
        for j in range(N):
            for i in range(N):
                temp = temp + test_pattern[k][i]*w[i][j]
            if(temp > 0 ):
                recalloutput[k][j]=1
            else:
                recalloutput[k][j]=-1
            temp = 0.0 

def overlap():
    corr= 0.0
    global recalloutput 
    for k in range(p):
        corr = 0.0
        for i in range(N):
            if(test_pattern[k][i]==recalloutput[k][i]):
                corr = corr + 1 
        print("Overlap of pattern Number", k+1, " is : " ,  corr/N)
        
def print_output(): 
    global recalloutput 
    for i in range(p) : 
        img = recalloutput[i].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()

if __name__ == "__main__" :
    
    lamda = 0.9 
    lamda = lamda / N 
    
    read_trg_set()	
    initialise_wts()
    correlate()
    write_hebb_wt()
    Train_SG()
    display()	
    read_test_set()   	
    recallnew()
    overlap()
    print_output();