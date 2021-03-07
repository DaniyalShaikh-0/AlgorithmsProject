
# Function to find length of Longest Common Subsequence of substring
# X[0..m-1] and Y[0..n-1]
from tkinter import *
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from generateData import *
import os
import time
import plotly.graph_objects as go
global filename
global owninput
def LCSLength(X, Y, m, n, lookup):
    # return if we have reached the end of either string
    if m == 0 or n == 0:
        return 0
    # construct a unique dict key from dynamic elements of the input
    key = (m, n)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
 
        # if last character of X and Y matches
        if X[m - 1] == Y[n - 1]:
            lookup[key] = LCSLength(X, Y, m - 1, n - 1, lookup) + 1
 
        else:
            # else if last character of X and Y don't match
            lookup[key] = max(LCSLength(X, Y, m, n - 1, lookup),
                              LCSLength(X, Y, m - 1, n, lookup))
 
    # return the sub-problem solution from the dictionary
    # print(lookup)
    # makeGraphTBL(lookup)
    if len(X)==m and len(Y)==n:
        # print(m,n)
        tbl=[[0 for _ in range(n+1)] for _s in range(m)]
        max_=0
        for i in range(1,m):
            for j in range(1,n+1):
                try:
                    max_=tbl[i][j]=lookup[(i,j)]
                except KeyError:
                    tbl[i][j]=max_

        # for x in tbl:
        #     print(x)
    # if len(X) <=15 or len(Y) <= 15:
        maakeLCSTBL(tbl,X,Y)
    return lookup[key]
 
def lcs(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='a' and fname[0]!='b' and fname[0]!='c':
        return 'Chosen file can not be run with Longest common Subsequence problem'
    a=open(fname,'r')
    X=a.readline()
    Y=a.readline()
    print(f'X is : {X} & Y is : {Y} and lenghts are {len(X)} & {len(Y)}')
    a.close()
    return LCSLength(X,Y,len(X),len(Y),{})

# Function to find length of shortest Common supersequence of
# sequences X[0..m-1] and Y[0..n-1]
def SCSLength(X, Y, m, n, lookup):
 
    # if we have reached the end of either sequence, return
    # length of other sequence
    if m == 0 or n == 0:
        return n + m
 
    # construct an unique dict key from dynamic elements of the input
    key = (m, n)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
 
        # if last character of X and Y matches
        if X[m - 1] == Y[n - 1]:
            lookup[key] = SCSLength(X, Y, m - 1, n - 1, lookup) + 1
 
        # else if last character of X and Y don't match
        else:
            lookup[key] = min(SCSLength(X, Y, m, n - 1, lookup),
                              SCSLength(X, Y, m - 1, n, lookup)) + 1
    if len(X)==m and len(Y)==n:
        # print(m,n)
        tbl=[[0 for _ in range(n+1)] for _s in range(m)]
        max_=0
        for i in range(1,m):
            for j in range(1,n+1):
                try:
                    max_=tbl[i][j]=lookup[(i,j)]
                except KeyError:
                    tbl[i][j]=max_
        # for x in tbl:
        #     print(x)
        maakeLCSTBL(tbl,X,Y,1)
    # return the sub-problem solution from the dict
    return lookup[key]

def scs(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='a' and fname[0]!='b' and fname[0]!='c':
        return 'Chosen file can not be run with Shortest common subsequence problem'
    a=open(fname,'r')
    X=a.readline()
    Y=a.readline()
    a.close()
    return SCSLength(X,Y,len(X),len(Y),{})-1
# Function to find Levenshtein Distance between X and Y
# m and n are the number of characters in X and Y respectively
def dist(X, Y):
 
    (m, n) = (len(X), len(Y))
 
    # for all i and j, T[i,j] will hold the Levenshtein distance between
    # the first i characters of X and the first j characters of Y
    # note that T has (m+1)*(n+1) values
    T = [[0 for x in range(n + 1)] for y in range(m + 1)]
 
    # source prefixes can be transformed into empty by
    # dropping all characters
    for i in range(1, m + 1):
        T[i][0] = i                     # (case 1)
 
    # target prefixes can be reached from empty source prefix
    # by inserting every character
    for j in range(1, n + 1):
        T[0][j] = j                     # (case 1)
 
    # fill the lookup table in bottom-up manner
    for i in range(1, m + 1):
 
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:    # (case 2)
                cost = 0                # (case 2)
            else:
                cost = 1                # (case 3c)
 
            T[i][j] = min(T[i - 1][j] + 1,          # deletion (case 3b)
                          T[i][j - 1] + 1,          # insertion (case 3a)
                          T[i - 1][j - 1] + cost)   # replace (case 2 + 3c)
    maakeLCSTBL(T,X,Y,2)
 
    return T[m][n]
 
def edit(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='a' and fname[0]!='b' and fname[0]!='c':
        return 'Chosen file can not be run with Levensthien distance problem'
    a=open(fname,'r')
    X=a.readline()
    Y=a.readline()
    a.close()
    return dist(X,Y)

    

# Iterative function to find length of longest increasing sub-sequence
# of given list
def LIS(A):
 
    # list to store sub-problem solution. L[i] stores the length
    # of the longest increasing sub-sequence ends with A[i]
    L = [0] * len(A)
 
    # longest increasing sub-sequence ending with A[0] has length 1
    L[0] = 1
 
    # start from second element in the list
    for i in range(1, len(A)):
        # do for each element in sublist A[0..i-1]
        for j in range(i):
            # find longest increasing sub-sequence that ends with A[j]
            # where A[j] is less than the current element A[i]
            if A[j] < A[i] and L[j] > L[i]:
                L[i] = L[j]
 
        # include A[i] in LIS
        L[i] = L[i] + 1
    
    makelisTbl(L,A,1)
 
    # return longest increasing sub-sequence (having maximum length)
    return max(L)
 
def lisp(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='d' and fname[0]!='e' and fname[0]!='g' and fname[0]!='i':
        return 'Chosen file can not be run with Longest increasing subsequence problem'
    a=open(fname,'r')
    A=list(map(int,a.readline().split()))
    a.close()
    return LIS(A)

# Function to find the most efficient way to multiply
# given sequence of matrices
def MatrixChainMultiplication(dims, i, j, T):
 
    # base case: one matrix
    if j <= i + 1:
        return 0
 
    # stores minimum number of scalar multiplications (i.e., cost)
    # needed to compute the matrix M[i+1]...M[j] = M[i..j]
    min_ = float('inf')
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a lookup table
    if T[i][j] == 0:
 
        # take the min_imum over each possible position at which the
        # sequence of matrices can be split
 
        """
            (M[i+1]) x (M[i+2]..................M[j])
            (M[i+1]M[i+2]) x (M[i+3.............M[j])
            ...
            ...
            (M[i+1]M[i+2]............M[j-1]) x (M[j])
        """
 
        for k in range(i + 1, j):
 
            # recur for M[i+1]..M[k] to get an i x k matrix
            cost = MatrixChainMultiplication(dims, i, k, T)
 
            # recur for M[k+1]..M[j] to get a k x j matrix
            cost += MatrixChainMultiplication(dims, k, j, T)
 
            # cost to multiply two (i x k) and (k x j) matrix
            cost += dims[i] * dims[k] * dims[j]
 
            if cost < min_:
                min_ = cost
 
        T[i][j] = min_
 
    # return min cost to multiply M[j+1]..M[j]
    return T[i][j]

def mcm(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='d' and fname[0]!='e' and fname[0]!='g' and fname[0]!='i':
        return 'Chosen file can not be run with Matrix Chain Multiplication problem'
    a=open(fname,'r')
    A=list(map(int,a.readline().split()))
    # print(f'Lenght of elements : {len(A)} and array:\n {A}')
    a.close()
    return MatrixChainMultiplication(A,0,len(A)-1,T = [[0 for x in range(len(A))] for y in range(len(A))])




# Values (stored in list v)
# Weights (stored in list w)
# Number of distinct items (n)
# Knapsack capacity W
# def knapSack(W, wt, val, n): 
#     K = [[0 for x in range(W + 1)] for x in range(n + 1)] 
  
#     # Build table K[][] in bottom up manner 
#     for i in range(n + 1): 
#         for w in range(W + 1): 
#             if i == 0 or w == 0: 
#                 K[i][w] = 0
#             elif wt[i-1] <= w: 
#                 K[i][w] = max(val[i-1] 
#                           + K[i-1][w-wt[i-1]],   
#                               K[i-1][w]) 
#             else: 
#                 K[i][w] = K[i-1][w] 
#     for i in range(n+1):
#         for a in range(W+1):
#             print(K[i][a],end=' ')
#         print()
#     return K[n][W]
def knapSack(v, w, n, W, lookup):
    # base case: Negative capacity
    if W < 0:
        return float('-inf')
 
    # base case: no items left or capacity becomes 0
    if n < 0 or W == 0:
        return 0
 
    # construct an unique dict key from dynamic elements of the input
    key = (n, W)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
        # Case 1. include current item n in knapSack (v[n]) & recur for
        # remaining items (n - 1) with decreased capacity (W - w[n])
        include = v[n] + knapSack(v, w, n - 1, W - w[n], lookup)
        
 
        # Case 2. exclude current item n from knapSack and recur for
        # remaining items (n - 1)
        exclude = knapSack(v, w, n - 1, W, lookup)
 
        # assign max value we get by including or excluding current item
        lookup[key] = max(include, exclude)
 
    # return solution to current sub-problem

    return lookup[key]

def knap01(fname):
    lookup={}
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='f' and fname[0]!='h':
        return 'Chosen file can not be run with 0-1 knapsack problem'
    f=open(fname,'r')
    v=list(map(int,f.readline().split()))
    w=list(map(int,f.readline().split()))
    W=int(f.readline())
    n=len(v)
    f.close()
    ans=knapSack(v, w, n-1, W,lookup)
    makeknapTbl(lookup,1)
    # for i in range(1,n+1):
    #     for j in range(1,n):
    #         try:
    #             tbl[i][j]=lookup[(i,j)]
        # for x in tbl:
        #     print(x)
        # maakeLCSTBL()
    return ans

# 0-1 Knapsack problem
# PASS TO ALGORITHM FROM HERE OR PUT THESE IN FUCNTION
    
 

# Return true if there exists a sublist of list A[0..n) with given sum
def subsetSum(A, n, sum):
 
    # T[i][j] stores true if subset with sum j can be attained with
    # using items up to first i items
    T = [[False for x in range(sum + 1)] for y in range(n + 1)]
 
    # if sum is zero
    for i in range(n + 1):
        T[i][0] = True
 
    # do for ith item
    for i in range(1, n + 1):
 
        # consider all sum from 1 to sum
        for j in range(1, sum + 1):
 
            # don't include ith element if j-A[i-1] is negative
            if A[i - 1] > j:
                T[i][j] = T[i - 1][j]
            else:
                # find subset with sum j by excluding or including the ith item
                T[i][j] = T[i - 1][j] or T[i - 1][j - A[i - 1]]
 
    # return maximum value
    return T[n][sum]
 
 
# Return true if given list A[0..n-1] can be divided into two
# sublists with equal sum
def partition(A):
 
    total = sum(A)
 
    # return true if sum is even and list can can be divided into
    # two sublists with equal sum
    return (total & 1) == 0 and subsetSum(A, len(A), total // 2)
 
 

def part(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='d' and fname[0]!='e' and fname[0]!='g' and fname[0]!='i':
        return 'Chosen file can not be run with Partition problem'
    a=open(fname,'r')
    A=list(map(int,a.readline().split()))
    a.close()
    return partition(A)
# Function to find best way to cut a rod of length n
# where rod of length i has a cost price[i-1]
def rodCut(price, n,length,T):
    # T[i] stores maximum profit achieved from rod of length i
 
    # consider rod of length i
    for i in range(1, n + 1):
        # divide the rod of length i into two rods of length j
        # and i-j each and take maximum
        for j in range(1, i + 1):
            if j<=len(price):
                T[i] = max(T[i], price[j - 1] + T[i - j])
 
    # T[n] stores maximum profit achieved from rod of length n
    return T[n]
 
def rcutting(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='f' and fname[0]!='h':
        return 'Chosen file can not be run with rod cutting problem'
    f=open(fname,'r')
    pric=list(map(int,f.readline().split()))
    lens=list(map(int,f.readline().split()))
    W=int(f.readline())
    ar=min(len(lens),W)
    # T[i] stores maximum profit achieved from rod of length i
    T = [0] * (ar + 1)
    f.close()
    ans=rodCut(pric,ar,lens,T)
    T[0]='max vals : '
    makerodtbl(T,pric)
    return ans


# Function to find the minimum number of coins required
# to get total of N from set S
import random
def findMinCoins(S, N):
 
    # T[i] stores minimum number of coins needed to get total of i
    T = [0] * (N + 1)
 
    for i in range(1, N + 1):
 
        # initialize minimum number of coins needed to infinity
        T[i] = float('inf')
 
        # do for each coin
        for c in range(len(S)):
            # check if index doesn't become negative by including
            # current coin c
            if i - S[c] >= 0:
                res = T[i - S[c]]
 
                # if total can be reached by including current coin c,
                # update minimum number of coins needed T[i]
                if res != float('inf'):
                    T[i] = min(T[i], res + 1)
 
    # T[N] stores the minimum number of coins needed to get total of N
    return T[N]
 
 

def coinchange(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='d' and fname[0]!='e' and fname[0]!='g' and fname[0]!='i':
        return 'Chosen file can not be run with Coin change problem'
    a=open(fname,'r')
    A=list(map(int,a.readline().split()))
    if fname[0]=='j':
        dChange=int(a.readline())
    else:
        x=[55,64]
        dChange=random.choice(x)
    a.close()
    return findMinCoins(A,dChange)

# Function to determine if can be segmented into a space-separated
# sequence of one or more dictionary words
def wordBreak(dict, str, lookup):
 
    # n stores length of current substring
    n = len(str)
 
    # return true if we have reached the end of the String
    if n == 0:
        return True
 
    # if sub-problem is seen for the first time
    if lookup[n] == -1:
 
        # mark sub-problem as seen (0 initially assuming String
        # can't be segmented)
        lookup[n] = 0
 
        for i in range(1, n + 1):
            # consider all prefixes of current String
            prefix = str[:i]
 
            # if prefix is found in dictionary, then recur for suffix
            if prefix in dict and wordBreak(dict, str[i:], lookup):
                # return true if the can be segmented
                lookup[n] = 1
                return True
 
    # return solution to current sub-problem
    return lookup[n] == 1
def wordb(fname):
    if not len(fname):
        return 'Please choose a file'
    if fname[0]!='j':
        return 'chosen file cannot be run with Work break problem'
    a=open(fname,'r')
    dct = a.readline().split()
    # print(f'dt is : {dct}')
    wrd = a.readline()
    # print(wrd)
    a.close()
    return wordBreak(dct,wrd,lookup = [-1] * (len(wrd) + 1))
def btn1():
    
    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    global filename
    filename = askopenfilename()
    # print(filename)
    filename = filename.split('/')[-1]
    l0=Label(wins,text='showing file ' + filename,bg='black',fg='yellow').grid(column=2,row=1)
    os.system('notepad.exe '+filename)
    
    
def windd(txt):
    window2= Tk()
    window2.geometry("1500x100")
    window2.title("Algorithm")
    if len(txt) > 20:
        siz=24
    else:
        siz=30
    l=Label(window2,text=txt,bg='black',fg='yellow',font=("Arial Bold", siz)).place(height=100,width=1500)
    window2.mainloop()


def b1():
    global filename
    # print(f'file name is : {filename}')
    userinput=lcs(filename)
    # lcs1=Label(wins,text=userinput).grid(column=1,row=4)
    userinput=str(userinput)
    # l=Label(wins,text=userinput,bg='black',fg='yellow',font=("Arial Bold", 24)).place(height=100,width=100,relx=0.8,rely=0.1)
    if len(userinput) < 19:

        windd('Answer of Longest Common Subsequence on file : '+ filename + ' = ' + str(userinput))
    else:
        windd(str(userinput))
def b2():

    
    userinput=scs(filename)
    # l2=Label(wins,text=userinput).grid(column=1,row=5)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of Shortest Common Supersequence on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
def b3():
    userinput=edit(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of Levinthean edit distance on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
   
    # l3=Label(wins,text=userinput).grid(column=1,row=6)
def b4():

    
    userinput=lisp(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of longest increasing supersequence on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
    
    # l4=Label(wins,text=userinput).grid(column=1,row=7)
def b5():

    
    userinput=mcm(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of Matrix chain multiply on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
    
    # l5=Label(wins,text=userinput).grid(column=1,row=8)

def b6():
    userinput=knap01(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of 0-1 KnapSack on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
    # l6=Label(wins,text=userinput).grid(column=1,row=9)

def b7():
    userinput=part(filename)
    userinput=str(userinput)
    if userinput==True:
        write='Subset Partition is possible for chosen input'
    else:
        write='Subset Partition not possible for chosen input'
    windd(str(write))
    # l7=Label(wins,text=write).grid(column=1,row=10)

def b8():
    userinput=rcutting(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of Rod Cutting on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
   
    # l8=Label(wins,text=userinput).grid(column=1,row=11)

def b9():
    userinput=coinchange(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of Coin Change on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
    # l9=Label(wins,text=userinput).grid(column=1,row=12)

def b10():
    userinput=wordb(filename)
    userinput=str(userinput)
    if len(userinput) < 19:

        windd('Answer of WorkBreak on file : '+ filename + ' = '+ str(userinput))
    else:
        windd(str(userinput))
    # l10=Label(wins,text=userinput).grid(column=1,row=13)
 
def genall():
    genABC()
    genDEGI()
    genFH()
    genJ()
    lbl2 = Label(wins,text='DATA GENERATED SUCCESSFULLY',bg='#98EDED',font=("Arial Bold Underlined", 10))
    lbl2.grid(column=2,row=0)
    # time.sleep(1)
    # lbl2.destroy()

def Transpose(A):
    cols=len(A)
    rows=(len(A[0]))
    # print(f'Matrix = {rows} x {cols}')
    AT=[[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            AT[i][j]=A[j][i]
    return AT


def makerodtbl(T,pric):
    pric.insert(0,'Prices : ')
    h=600
    w=1920+(len(T)*30)
    txt='Rod cutting Problem'
    fig = go.Figure(data=[go.Table(
    header=dict(values=pric,
                height=30,
                line_color='white',
                fill_color='#66ffe0',
                align='left'),
    cells=dict(values=T, # 2nd column
                height=30,
                line_color='white',
                fill_color='#e6fffa',
                align='left'))
    ],
    layout=dict(title_text=txt,width=w, height=h,titlefont=dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )))
    fig.show()


def makeknapTbl(lookup,n):
    if len(lookup)>15:
        h=600
        w=1920+(len(lookup)*30)
    else:
        h=600
        w=1200
    if n==1:
        txt='0-1 KnapSack\nn\nW\nVal'
    elif n==0:
        txt='Longest Common Subsequence'
    else:
        txt='Levenshtein edit Distance'
    keys=list(lookup.keys())
    vals=list(lookup.values())
    keys.insert(0,('n','w'))
    vals.insert(0,'Value : ')
    print(keys)
    fig = go.Figure(data=[go.Table(
    header=dict(values=keys,
                height=30,
                line_color='white',
                fill_color='#66ffe0',
                align='left'),
    cells=dict(values=vals, # 2nd column
                height=30,
                line_color='white',
                fill_color='#e6fffa',
                align='left'))
    ],
    layout=dict(title_text=txt,width=w, height=h,titlefont=dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )))
    fig.show()

def makelisTbl(L,A,n=0):
    if len(A)>15:
        h=1600+(len(A)*10)
        w=1920+(len(A)*10)
    else:
        h=800
        w=900
    if n==1:
        txt='Longest Increasing SubSequence'
    elif n==0:
        txt='Partition Problem'
    else:
        txt='Coin Change Problem'
    fig = go.Figure(data=[go.Table(
    header=dict(values=A,
                height=30,
                line_color='white',
                fill_color='#66ffe0',
                align='left'),
    cells=dict(values=L, # 2nd column
                height=30,
                line_color='white',
                fill_color='#e6fffa',
                align='left'))
    ],
    layout=dict(title_text=txt,width=w, height=h,titlefont=dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )))
    fig.show()

def maakeLCSTBL(table,X,Y,n=0):
    X=' '+X
    Y=' '+Y
    l=0
    dec=max(len(X),len(Y))
    if dec>15:
        h=1600+(dec*10)
        w=1920+(dec*10)
    else:
        h=800
        w=900
    # /print(vals)
    for i in range(len(table[0])):
        try:
            table[0][i]=Y[i]
        except IndexError:
            x=0
            # print('wrong at : ',i)

    # for l in range(1,len(Y)+1):
    #     for j in range(1,len(X)-1):
    #         try:
    #             max_=vals[l][j]=table[l-1][j-1]
    #         except IndexError:
    #             vals[l][j]=max_
    # for i in Transpose(vals):
    #     print(i)
    # for j in table:
    #     print(j)
    if n==1:
        txt='Shortest Common SuperSequence'
    elif n==0:
        txt='Longest Common Subsequence'
    else:
        txt='Levenshtein edit Distance'

    fig = go.Figure(data=[go.Table(
    header=dict(values=list(X),
                height=30,
                line_color='white',
                fill_color='#66ffe0',
                align='left'),
    cells=dict(values=table, # 2nd column
                height=30,
                line_color='white',
                fill_color='#e6fffa',
                align='left'))
    ],
    layout=dict(title_text=txt,width=w, height=h,titlefont=dict(
        family="sans serif",
        size=18,
        color="LightSeaGreen"
    )))
    fig.show()


# print(lisp('e.txt'))
def chngScr():
    window.wm_state('iconic')
    window.iconify()
    global wins
    wins=Tk()
    wins.geometry('500x500')
    bttn12=Button(wins,text="Choose file",bg="Green",fg="black",command=btn1)
    bttn12.grid(row=0,column=0)
    # combo=Combobox(wins).grid(column=1,row=2)
    # combo['values']={"File","user input", "Choose option"}
    # combo.current("Choose option")
    lbl1 =Label(wins, text="Algorithms:", font=("Arial Bold", 20),
    bg="black",fg="white")
    lbl1.grid(column=0,row=3)
    bttn2=Button(wins,text="Longest Common Subsequence",bg="Green",fg="black",command=b1)
    bttn2.grid(column=0,row=4)
    bttn3=Button(wins,text="Shortest Common Supersequence",bg="Light Green",fg="black",command=b2).grid(column=0,row=5)
    bttn4=Button(wins,text="Levenshtein edit Distance",bg="Green",fg="black",command=b3).grid(column=0,row=6)
    bttn5=Button(wins,text="Longest Increasing Subsequence",bg="Light Green",fg="black",command=b4).grid(column=0,row=7)
    bttn6=Button(wins,text="Matrix Chain Multiplication",bg="Green",fg="black",command=b5).grid(column=0,row=8)
    bttn7=Button(wins,text="0/1 Knapsack",bg="Light Green",fg="black",command=b6).grid(column=0,row=9)
    bttn8=Button(wins,text="Partition Problem",bg="Green",fg="black",command=b7).grid(column=0,row=10)
    bttn9=Button(wins,text="RodCutting Problem",bg="Light Green",fg="black",command=b8).grid(column=0,row=11)
    bttn10=Button(wins,text="Coin change making",bg="Green",fg="black",command=b9).grid(column=0,row=12)
    bttn11=Button(wins,text="Word Break Problem",bg="Light Green",fg="black",command=b10).grid(column=0,row=13)
    bttn11=Button(wins,text="GENERATE NEW RANDOM DATA",bg="Green",fg="black",command=genall).grid(column=0,row=14)


if __name__ == '__main__':
    filename=''
    scrwid='1920x1080'
    labwids=int(scrwid.split('x')[0])
    window = Tk()
    # scs('a.txt')
    # wins
    window.geometry(scrwid)
    window['bg']='#018d82'
    img = Image.open("D:\\Semester 5\\Algorithms\\Algorithms.jpg")
    # print(img.height,img.width)
    render = ImageTk.PhotoImage(img)
    imgs = Label(window,image=render)
    imgs.image = render
    pl=12
    imgs.place(relx=0.5,rely=0.2,anchor = 'center',height=290,width=735)
    window.title("Running Algorithms")

    lbl =Label(window, text="  A PROJECT OF ALGORITHMS BY:", font=("Arial Bold", labwids//50),
    bg="#018d82",fg="white",anchor='center')
    lbl.place(relx=0.5,rely=0.4,anchor = 'center')
    # USER_INP = simpledialog.askstring(title="Test",
    #          
    #                        prompt="What's your Name?:")
    pl+=1
    lbl2 =Label(window, text="\n Tehreem Ahmed Siddique (18k1064)\n\n Daniyal Abdul Bari (18k1055)", font=("Arial Bold", labwids//55),
    bg="#018d82",fg="white")
    lbl2.place(relx=0.5,rely=0.55,anchor = 'center')
    # bttn1=Label(window,text="Enter File as input",bg="white",fg="black").grid(column=0,row=1)
    buton=Button(window,text="NEXT",bg="black",fg="white",command=chngScr)
    buton.place(relx=0.5,rely=0.8,anchor = 'center',width=100)
    window.mainloop()