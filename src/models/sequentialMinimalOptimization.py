# Throughout I will use 1 and 2 instead of a and b like in my explanation in the report
# That is: alpha_a := alpha1, alpha_b := alpha2, etc. 
import numpy as np
import random

X = np.array([
    [2, 5], [1.5, 2.5], [3, 4], [4, 5], [1, 1], [0.5, 4], [0,2],
    [2, 0], [4, 1], [4, 2], [5, 2], [3, -1], [5, -1.5],
])

y = np.array([
    1, 1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1
])

#TODO
C = 10 #THIS IS A PLACEHOLDER VALUE NEED TO FIX AND FIND CORRECT 
#TODO
tol = 10e-2 #this was random value go find real one
#TODO
eps = 10e-4

def kernel(index1, index2):
    #currently our kernel function is just the dot product
    return np.dot(X[index1], X[index2])

def bValue(alpha):
    #TODO
    return 1

def hypothesis(index, alpha):
    total = 0
    for i in range(y.size()):
        subtotal = alpha[i]*y[i]*kernel(X[i], X[index])
        total += subtotal

    total += bValue(alpha)
    return total

def secondChoiceHeuristic(index2, alpha, threshold):
    #TODO
    return 4 

def stepIndex(index, length):
    if index == length -1:
        index == 0
    else:
        index += 1
    return index 

def objectiveFunction(value):
    #TODO
    return 0

def updateWeightVector(alpha):
    #TODO
    return 


def takeStep(index1, index2, alpha):
    if index1 == index2:
        return 0, alpha
    alpha1 = alpha[index1]
    alpha2 = alpha[index2]
    y1 = y[index1]
    y2 = y[index2]
    E1 = hypothesis(index1, alpha) - y1 #THIS SHOULD BE STORED IN ERROR CACHE TODO
    E2 = hypothesis(index2, alpha) - y2 #THIS SHOULD BE STORED IN ERROR CACHE TODO
    s = y1 * y2
     #L = U
     #H = V
    if y1 == y2:
        U = max(0, alpha1 + alpha2 - C)
        V = min(C, alpha1 + alpha2)
    else:
        U = max(0, alpha1 - alpha2)
        V = min(C, C-  alpha1 + alpha2)

    if U == V:
        return 0, alpha
    
    k11 = kernel(X[index1], X[index1])
    k12 = kernel(X[index1], X[index2])
    k22 = kernel(X[index2], X[index2])

    eta = 2*k12 - k11 - k22
    #clip the value
    if eta < 0:
        a2 = alpha2 - (y2 * (E1-E2))/eta
        if a2 < U:
            a2 = U
        elif a2 > V:
            a2 = V
    else:
        Uobj = objectiveFunction(U) #objective function at a2 = U
        Vobj = objectiveFunction(V)

        if Uobj > Vobj+eps:
            a2 = U
        elif Uobj < Vobj-eps:
            a2 = V
        else:
            a2 = alpha2

    if abs(a2 - alpha2) < eps*(a2 + alpha2 + eps):
        return 0, alpha
    
    a1 = alpha1 + s*(alpha2 - a2)
    #TODO update threshold to reflect change in Lagrange multipliers
    #TODO update error cache using new Lagrange multipliers
    
    #update alpha vector
    alpha[index1] = a1
    alpha[index2] = a2
    updateWeightVector(alpha)
    return 1, alpha

        


def examineExample(index2, alpha, threshold):
    y2 = y[index2]
    alpha2 = alpha[index2]
    E2 = hypothesis(index2, alpha) - y2 #THIS SHOULD BE STORED IN ERROR CACHE TODO
    r2 = E2 * y2

    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        if len([a for a in alpha if 0 < a < C]) > 1:
            index1 = secondChoiceHeuristic(index2, alpha, threshold)
            val, alpha = takeStep(index1, index2, alpha)
            if val:
                return 1, alpha, threshold

        #loop through alpha values on boundary starting at random index
        current = random.randint(0, y.size()) #inclusive
        start = current
        #check the first one before the loop
        if alpha[current] != 0 and alpha[current] != C:
            val, alpha = takeStep(index1, index2, alpha)
            if val:
                return 1, alpha, threshold
        else:
            stepIndex(current, y.size())

        while True:
            if current == start:
                break

            if alpha[current] != 0 and alpha[current] != C:
                val, alpha = takeStep(index1, index2, alpha)
                if val:
                    return 1, alpha, threshold
            else:
                stepIndex(current, y.size())

        #otherwise, just loop over all possible valuse starting at a random index
        current = random.randint(0, y.size()) #inclusive
        start = current

        val, alpha = takeStep(current, index2, alpha)
        while not val:
            stepIndex(current, y.size())
            if current == start:
                break
            val, alpha = takeStep(current, index2, alpha)

        if val:
            return 1, alpha, threshold

    return 0, alpha, threshold


def smo():
    threshold = 0 
    pointCount = y.size()
    alpha = [0] * pointCount
    numChanged = 0
    examineAll = 1

    while(numChanged > 0 | examineAll):
        numChanged = 0
        if examineAll:
            for i in range(pointCount):
                num, alpha, threshold = examineExample(i, alpha, threshold)
                numChanged += num
        else:
            for i in range(pointCount):
                if alpha[i] != 0 and alpha[i] != C:
                    num, alpha, threshold = examineExample(i, alpha, threshold)
                    numChanged += num
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1


if __name__ == "__main__":
    #could change values of data and labels here if I wanted
    #but for now we will just launch into the algorithm
    #could also set kernel function????
    smo()