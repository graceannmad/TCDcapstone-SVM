# Throughout I will use 1 and 2 instead of a and b like in my explanation in the report
# That is: alpha_a := alpha1, alpha_b := alpha2, etc. 
import numpy as np
import random

#temporary place-holder values
X = np.array([
    [2, 5], [1.5, 2.5], [3, 4], [4, 5], [1, 1], [0.5, 4], [0,2],
    [2, 0], [4, 1], [4, 2], [5, 2], [3, -1], [5, -1.5],
])
y = np.array([
    1, 1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1
])

#These values can be varied here or set in the main function
#suggested tolerance by Cristianini and Shawe-Taylor (2000)
#suggested epsilon from original SMO paper
C = 1
tol = 10e-2
eps = 10e-3

def kernel(index1, index2):
    #currently our kernel function is just the dot product
    return np.dot(X[index1], X[index2])

def update_b(a1, alpha1, a2, alpha2, y1, y2, E1, E2, k11, k12, k22, oldb):
    b1 = E1 + y1*(a1 - alpha1)*k11 + y2*(a2 - alpha2)*k12 + oldb
    b2 = E2 + y1*(a1 - alpha1)*k12 + y2*(a2 - alpha2)*k22 + oldb

    if 0 < a1 and a1 < C:
        return b1
    elif 0 < a2 and a2 < C:
        return b2
    else:
        return (b1 + b2)/2
        
def secondChoiceHeuristic(E2, index2, E, alpha):
    non_bound = [i for i in range(len(alpha)) if 0 < alpha[i] < C and i != index2]

    if not non_bound:
        return None

    if E2 < 0:
        return max(non_bound, key=lambda i: E[i])
    else:
        return min(non_bound, key=lambda i: E[i])

def stepIndex(index, length):
    if index == length -1:
        index = 0
    else:
        index += 1
    return index 

def objectiveFunction(alpha2, index2, alpha):
    oldVal = alpha[index2] #temporarily alter alpha
    alpha[index2] = alpha2

    otherSum = 0
    for i in range(y.size):
        for j in range(y.size):
            otherSum += alpha[i]*alpha[j]*y[i]*y[j]*kernel(i, j)

    obj = sum(alpha) - (1/2)*otherSum

    alpha[index2] = oldVal #revert alpha
    return obj

def takeStep(index1, index2, alpha, b, E):
    if index1 == index2:
        return 0, alpha, b, E
    alpha1 = alpha[index1]
    alpha2 = alpha[index2]
    y1 = y[index1]
    y2 = y[index2]
    E1 = E[index1]
    E2 = E[index2] 
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
        return 0, alpha, b, E
    
    k11 = kernel(index1, index1)
    k12 = kernel(index1, index2)
    k22 = kernel(index2, index2)

    eta = 2*k12 - k11 - k22
    #clip the value
    if eta < 0:
        a2 = alpha2 - (y2 * (E1-E2))/eta
        if a2 < U:
            a2 = U
        elif a2 > V:
            a2 = V
    else:
        Uobj = objectiveFunction(U, index2, alpha) #objective function at a2 = U
        Vobj = objectiveFunction(V, index2, alpha) #at a2 = V

        if Uobj > Vobj+eps:
            a2 = U
        elif Uobj < Vobj-eps:
            a2 = V
        else:
            a2 = alpha2

    if abs(a2 - alpha2) < eps*(a2 + alpha2 + eps):
        return 0, alpha, b, E
    
    a1 = alpha1 + s*(alpha2 - a2)

    #same condition down here for a1
    if abs(a1 - alpha1) < eps*(a1 + alpha1 + eps):
        return 0, alpha, b, E
    
    #we may need to also clip a1
    if a1 < 0:
        a1 = 0
    elif a1 > C:
        a1 = C
    
    #update alpha vector
    alpha[index1] = a1
    alpha[index2] = a2
    b_old = b
    b = update_b(a1, alpha1, a2, alpha2, y1, y2, E1, E2, k11, k12, k22, b)
    #Update error cache using new Lagrange multipliers and b value
    for i in range(len(E)):
        E[i] += (
            y1 * (a1 - alpha1) * kernel(index1, i)
            + y2 * (a2 - alpha2) * kernel(index2, i)
            + (b - b_old)
        )
    #Force exact zeros for updated points
    E[index1] = 0.0
    E[index2] = 0.0
    return 1, alpha, b, E

def examineExample(index2, alpha, b, E):
    y2 = y[index2]
    alpha2 = alpha[index2]
    E2 = E[index2]
    r2 = E2 * y2

    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        if len([a for a in alpha if 0 < a < C]) > 1:
            index1 = secondChoiceHeuristic(E2, index2, E, alpha)
            if index1 is not None:
                val, alpha, b, E = takeStep(index1, index2, alpha, b, E)
                if val:
                    return 1, alpha, b, E

        #loop through alpha values on boundary starting at random index
        current = random.randint(0, y.size-1) #inclusive
        start = current
        #check the first one before the loop
        if alpha[current] != 0 and alpha[current] != C:
            val, alpha, b, E = takeStep(current, index2, alpha, b, E)
            if val:
                return 1, alpha, b, E
        else:
            current = stepIndex(current, y.size)

        while True:
            if current == start:
                break

            if alpha[current] != 0 and alpha[current] != C:
                val, alpha, b, E = takeStep(current, index2, alpha, b, E)
                if val:
                    return 1, alpha, b, E
            else:
                current = stepIndex(current, y.size)

        #otherwise, just loop over all possible valuse starting at a random index
        current = random.randint(0, y.size-1) #inclusive
        start = current

        val, alpha, b, E = takeStep(current, index2, alpha, b, E)
        while not val:
            current = stepIndex(current, y.size)
            if current == start:
                break
            val, alpha, b, E = takeStep(current, index2, alpha, b, E)

        if val:
            return 1, alpha, b, E

    return 0, alpha, b, E

def smo():
    pointCount = y.size
    alpha = np.zeros(pointCount)
    numChanged = 0
    examineAll = 1
    b = 0
    #error cache
    E = -y.astype(float) #because alpha and b are all zeroes

    while numChanged > 0 or examineAll:
        numChanged = 0
        if examineAll:
            for i in range(pointCount):
                num, alpha, b, E = examineExample(i, alpha, b, E)
                numChanged += num
        else:
            for i in range(pointCount):
                if alpha[i] != 0 and alpha[i] != C:
                    num, alpha, b, E = examineExample(i, alpha, b, E)
                    numChanged += num
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
    
    return alpha, b

if __name__ == "__main__":
    #could change values of data and labels, value of C
    #tolerances, and epsilon here if wanted
    #but for now we will just launch into the algorithm
    alpha, b = smo()
    print(f"The Lagrange multiplers are: {alpha}")
    print(f"The b value is: {b}")