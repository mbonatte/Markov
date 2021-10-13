def numberTransitions(df,worst,best):
    nStates = abs(best-worst)+1 #Number of Condition States (CS) an asset can be
    nTransitions = np.zeros(nStates)
    for i in range(len(df.iloc[:,0])): #Read the entire list
        if (worst > best):
            if (int(df.Initial[i])-int(df.Final[i])==-1): #Did it changed one EC?
                nTransitions[int(df.iloc[i,j])-1]+=1
        else:
            if (int(df.Initial.iloc[i])-int(df.Final.iloc[i])==1): #Did it changed one EC?
                nTransitions[int(df.Initial.iloc[i])-worst]+=1
    return nTransitions

def timeTransitions(df,worst,best):
    nStates = abs(best-worst)+1 #Number of Condition States (CS) an asset can be
    tTransitions = np.zeros(nStates)
    for i in range(len(df.iloc[:,0])): #Read the entire list
        if (worst > best):
            if (int(df.Initial[i])<=int(df.Final[i])):
              tTransitions[int(df.Initial[i])-1]+=df.Time[i]
        else:
            if (int(df.Initial.iloc[i])>=int(df.Final.iloc[i])):
              tTransitions[int(df.Initial.iloc[i])-worst]+=int(df.Time.iloc[i])
    return tTransitions

def likelihood(teta, df,worst,best):
    np.seterr(divide = 'ignore')
    nStates = abs(best-worst)+1 #Number of Condition States (CS) an asset can be
    Q = intensityMatrix(teta)
    V = 0
    prob_matrix =[]
    for i in range(max(list(df.Time))+1):
        prob_matrix.append(probability_matrix(i, Q))
    for i in range(len(df.iloc[:,0])): #Read the entire list
        P = prob_matrix[int(df.Time.iloc[i])]
        if (worst > best):
            V += np.log(P[int(df.Initial.iloc[i]-1)][int(df.Final.iloc[i])-1])
        else:
            V += np.log(P[-int(df.Initial.iloc[i])+nStates][-int(df.Final.iloc[i])+nStates])
    return -V

def intensityMatrix(teta):
    teta[-1]=0
    Q = tTransitions = np.zeros((len(teta),len(teta)))
    for i in range(len(teta)-1):
        Q[i][i] = -teta[i]
        Q[i][i+1] = teta[i]
    return Q

def probability_matrix(deltaT, Q):
    np.seterr(all='ignore')
    P = np.zeros((len(Q),len(Q)))
    for i in range(80):
        P +=  np.dot(np.linalg.matrix_power(np.dot(Q,1), i),(1/math.factorial(i)))
    return np.linalg.matrix_power(P, deltaT)

def markov(df,worst,best,optimize=True):
    nStates = abs(best-worst)+1 #Number of Condition States (CS) an asset can be
    nTransitions = numberTransitions(df,worst,best) #Number of Transitions from one CS to another
    tTransitions = timeTransitions(df,worst,best) #Sum of all time the asset stayed in the same CS
    teta = [nTransitions[i]/tTransitions[i] for i in range(nStates)]
    if (worst < best):
        teta.reverse()
    like = likelihood(teta, df,worst,best)
    print('prior likelihood = ',like)
    print('prior relative likelihood = ',np.exp(-like/len(df)))
    if (optimize != True):
        return teta
    bounds = [(0.000001, 1) for n in range(nStates)]
    res = minimize(
        likelihood,
        teta,
        args=(df,worst,best,),
        method = 'SLSQP',
        bounds=bounds,
        #options={'disp': True}
        )
    like = likelihood(res['x'], df,worst,best)
    print('posterior likelihood = ',like)
    print('posterior relative likelihood = ',np.exp(-like/len(df)))
    return res['x']

def meanEC(deltaT, Q,worst,best):
    meanEC = []
    prob_initial= [0 for i in range(len(Q[0]))]
    prob_initial[0]=1
    for i in range(deltaT):
        prob = np.array(prob_initial).dot(probability_matrix(i, Q))
        s = 0
        for j,p in enumerate(prob):
            if (worst > best):
                s += (j+worst)*p
            if (worst < best):
                s += (best-j)*p
        meanEC.append(s)
    return meanEC

def meanEC_MC(deltaT, Q,worst,best,MC=10000):
    import random
    meanEC = [[] for i in range(deltaT)]
    stdEC = [[] for i in range(deltaT)]
    prob_matrix = probability_matrix(1, Q)
    for ativo in range(MC):
        prob= [0 for i in range(len(Q[0]))]
        prob[0]=1
        nota=5
        for i in range(deltaT):
            prob = np.array(prob).dot(prob_matrix)
            nota=random.choices([i for i in range(len(Q[0]))], weights=list(prob), k=1)[0]
            if (i==0):
                nota=0
            if (worst > best):
                meanEC[i].append(best+nota)
            if (worst < best):
                meanEC[i].append(best-nota)
            prob= [0 for i in range(len(Q[0]))]
            prob[nota]=1
        for i in range(deltaT):
            stdEC[i] = np.std(meanEC[i])
            #if (i==20):
                #print(sorted(meanEC[i]))
            #meanEC[i] = np.mean(meanEC[i])
    return np.array(meanEC),np.array(stdEC)   
  
def stdEC(deltaT, Q,worst=1,best=5):
    stdEC = []
    prob_initial= [0 for i in range(len(Q[0]))]
    prob_initial[0]=1
    for i in range(deltaT):
        prob = np.array(prob_initial).dot(probability_matrix(i, Q))
        mean=0
        for j,p in enumerate(prob):
            if (worst > best):
                mean += (j+worst)*p
            if (worst < best):
                mean += (best-j)*p
        var=0
        for j,p in enumerate(prob):
            if (worst > best):
                var += ((best-j)+worst)**2 *p
            if (worst < best):
                var += ((best-j)- mean)**2 *p
        stdEC.append(var**0.5)
  return stdEC
