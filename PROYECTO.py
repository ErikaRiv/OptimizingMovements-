# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:29:53 2021

@author: Erika
"""

import numpy as np
import cvxopt
import plotly

def nuevoPaso(N, T, uk):
    #Constantes
    x=1
    x1=1
    x2=1
    y=1
    y1=1
    y2=1
    h=1 
    g=1
    alpha=0.1
    #beta=0.1
    gamma=0.1
    
    #Matrices de X^{...} y Y^{...}
    x_3=uk[0:N]
    y_3=uk[N:int(2*N)]
    
    #Tiempo discreto 
    t=np.empty(shape=[N, 1]) 
    for i in range (N): #creación de vector de tiempo discreto
        t[i]=i*T 
    #print("t: ", t)
    
    #Posiciones, velocidaad y aceleracióm en X y Y
    x_hat=np.zeros((3,N))
    y_hat=np.zeros((3,N))
    
    #Creación de matriz A Formula(4)
    A=np.eye(3)
    A[0,1]=A[1,2]=T
    A[0,2]=T**2 /2
    #print("A: ", A)
    #Creación de vector B Formula(4)
    B=np.zeros((3,1))
    B[0,0]=(T**3)/6
    B[1,0]=(T**2)/2
    B[2,0]=T
    #print("B: ", B)
    
    #Inicializando  las matrices de Xhat y Yhat
    x_hat[:,0] = np.matrix([x, x1, x2])
    y_hat[:,0] = np.matrix([y, y1, y2])
    #print("x_hat: ", x_hat)
    #print("y_hat: ", y_hat)
    
    #Nuevo paso 
    for i in range(N-1): 
        x_hat[:, i+1] = (A * (np.asmatrix(x_hat[:,i]).T) + B*x_3[i]).T #Ax_hat+Bx_3 Formula(1)
        y_hat[:, i+1] = (A * (np.asmatrix(x_hat[:,i]).T) + B*y_3[i]).T #Ay_hat+By_3 Formula(2)
    #print("x_hat: ", x_hat)
    #print("y_hat: ", y_hat)
    
    vect = [1,0,-h/g] #vector para Formula(5)
    
    #Aproximación de la posición
    z_x = np.transpose(vect)@x_hat #Formula(5)
    z_y	= np.transpose(vect)@y_hat #Formula(6)
    
    #Contrucción Pps(state) posición y Ppu (jerk) posición
    Pps = np.zeros((N,3))
    Ppu = np.zeros((N,N))
    
    #Entradas de Ppu
    b = np.zeros(N)
    for j in range(N):
        Pps[j] = np.array([1, j*T, (j**2 * T**2)/2])
        b[j] = (1/6 + j/2 + (j**2)/2) * (T**3)
        for l in range(N):
            if j < l:
                Ppu[j,l] = 0
            else:
                Ppu[j,j-l] = b[l]       
    #print("Pps:",Pps)
    #print("\nPpu:",Ppu)
    
    #Construcción Pvs y Pvu (velocidades)
    Pvs = np.zeros((N,3))
    Pvu = np.eye(N)
    #entradas de Pvu
    c = np.zeros((N,1))
    for j in range (N):#pilas
        Pvs[j] = np.array([0, 1, j*T])    
        c[j] = (1/2 + j) * (T**2)    
        for l in range(N):#pilas
            if j < l:
                Pvu[j,l] = 0
            else:
                Pvu[j,j-l] = c[l]
    #print("Pvs:",Pvs)
    #print("\nPvu:",Pvu)
    
    #Construcción Pzs y Pzu (Aceleraciones?)(Centroide)
    Pzs = np.zeros((N,3))
    Pzu = np.eye(N)
    # d: entradas de Pzu
    d = np.empty(shape=[N,1])
    for j in range (N-1):
        Pzs[j] = np.array([1, j*T, (j**2 * T**2)/2 - h/g])    
        d[j] = (T**3)/6 + (j*T**3)/2 + ((j**2)*(T**3))/2 - (h*T)/g    
        for l in range(N-1):
            if j < l:
                Pzu[j,l] = 0
            else:
                Pzu[j,j-l] = d[l]   
    #print("Pzs:",Pzs)
    #print("\nPzu:",Pzu)
      
    #Formula (7)
    #X = np.zeros((N,1))
    #Y = np.zeros((N,1))
    X = np.zeros(N)
    Y = np.zeros(N)
    for i in range(N-1): #Formula (7)
        X[i] = np.dot(Pps[i], x_hat[:,i]) + np.dot(Ppu[i], x_3)
        Y[i] = np.dot(Pps[i], y_hat[:,i]) + np.dot(Ppu[i], y_3)
    #print("\nX:",X)
    #print("\nY:",Y) 
    
    #Formula (8)
    #X_1=np.zeros((N,1))
    #Y_1=np.zeros((N,1))
    X_1 = np.zeros(N)
    Y_1 = np.zeros(N)
    for i in range(N-1):
        X_1[i]=np.dot(Pvs[i], x_hat[:,i]) + np.dot(Pvu[i], x_3)
        Y_1[i]=np.dot(Pvs[i], y_hat[:,i]) + np.dot(Pvu[i], y_3)
    #print("X_1: ", X_1)
    #print("Y_1: ", Y_1)
    
    #Formula (9)
    Zx = np.zeros((N*1))
    Zy = np.zeros((N*1))
    for i in range(N-1): 
        Zx[i] = np.dot(Pzs[i], x_hat[:,i]) + np.dot(Pzu[i], x_3)
        Zy[i] = np.dot(Pzs[i], y_hat[:,i]) + np.dot(Pzu[i], y_3)
    #print("\nZx:",Zx)
    #print("\nZy:",Zy)
    
    # X^{...}Y^{...} aceleraciones del zmp 
    uk =np.append(x_3,y_3) #Formula(13)
    print("uk: ", uk)
    I = np.eye(N)
    
    Qprime = alpha*I + gamma*Pzu.T@Pzu #Formula(15)
    
    #Matriz Q Formula(14)
    Q=np.zeros((2*N,2*N))
    print("here ", Q)
    half = int(N/2)
    for i in range(half):
    	for j in range(half):
    		Q[i,j]=Qprime[i,j]
    for k in range(15,32):
    	for l in range(15,32):
    		Q[k,l]=Qprime[k-16,l-16]
    print("Q: ", Q)
    
    #pk Formula (16)
    pkx=np.dot(gamma*Pzu, (np.dot(Pzs,x_hat[:,i])-Zx))
    #print("pkx: ", pkx)
    #pkx=pkx.reshape(-1, 1)
    #print("pkxre: ", pkx)
    pky=np.dot(gamma*Pzu, (np.dot(Pzs,y_hat[:,i])-Zy))
    #pky=pky.reshape(-1, 1)
    #pk=np.vstack((pkx, pky))
    pk=np.zeros(2*N)
    for i in range (2*N):
        if i < N:
            pk[i]=pkx[i]
        else:
            pk[i]=pky[i-N]
    #print("pk", pk)
    return(uk, Q, pk)

def nuevoPaso2(N, T, uk):
    #Constantes
    x=1
    x1=1
    x2=1
    y=1
    y1=1
    y2=1
    h=1 
    g=1
    alpha=0.1
    beta=0.1
    gamma=0.1
    m=2
    n = N/(m+1) 
    n = int(n)
    
    #Matrices de X^{...} y Y^{...} Aceleración CoM
    x_3=uk[0:N]
    y_3=uk[N:int(2*N)]
    
    #Tiempo discreto 
    t=np.empty(shape=[N, 1]) 
    for i in range (N): #creación de vector de tiempo discreto
        t[i]=i*T 
    #print("t: ", t)
    
    #Posiciones, velocidad y acelaración en X y Y
    x_hat=np.zeros((3,N))
    y_hat=np.zeros((3,N))
    
    #Creación de matriz A Formula(4)
    A=np.eye(3)
    A[0,1]=A[1,2]=T
    A[0,2]=T**2 /2
    #print("A: ", A)
    #Creación de vector B Formula(4)
    B=np.zeros((3,1))
    B[0,0]=(T**3)/6
    B[1,0]=(T**2)/2
    B[2,0]=T
    #print("B: ", B)
    
    #Inicializando  las matrices de Xhat y Yhat
    x_hat[:,0] = np.matrix([x, x1, x2])
    y_hat[:,0] = np.matrix([y, y1, y2])
    #print("x_hat: ", x_hat)
    #print("y_hat: ", y_hat)
    
    #Nuevo paso 
    for i in range(N-1): 
        x_hat[:, i+1] = (A * (np.asmatrix(x_hat[:,i]).T) + B*x_3[i]).T #Ax_hat+Bx_3 Formula(1)
        y_hat[:, i+1] = (A * (np.asmatrix(x_hat[:,i]).T) + B*y_3[i]).T #Ay_hat+By_3 Formula(2)
    #print("x_hat: ", x_hat)
    #print("y_hat: ", y_hat)
    
    vect = [1,0,-h/g] #vector para Formula(5)
    
    #Aproximación de la posición
    z_x = np.transpose(vect)@x_hat #Formula(5)
    z_y	= np.transpose(vect)@y_hat #Formula(6)
    
    #Contrucción Pps(state) posición y Ppu (jerk) posición
    Pps = np.zeros((N,3))
    Ppu = np.zeros((N,N))
    
    #Entradas de Ppu
    b = np.zeros(N)
    for j in range(N):
        Pps[j] = np.array([1, j*T, (j**2 * T**2)/2])
        b[j] = (1/6 + j/2 + (j**2)/2) * (T**3)
        for l in range(N):
            if j < l:
                Ppu[j,l] = 0
            else:
                Ppu[j,j-l] = b[l]       
    #print("Pps:",Pps)
    #print("\nPpu:",Ppu)
    
    #Construcción Pvs y Pvu (velocidades)
    Pvs = np.zeros((N,3))
    Pvu = np.eye(N)
    #entradas de Pvu
    c = np.zeros((N,1))
    for j in range (N):#pilas
        Pvs[j] = np.array([0, 1, j*T])    
        c[j] = (1/2 + j) * (T**2)    
        for l in range(N):#pilas
            if j < l:
                Pvu[j,l] = 0
            else:
                Pvu[j,j-l] = c[l]
    #print("Pvs:",Pvs)
    #print("\nPvu:",Pvu)
    
    #Construcción Pzs y Pzu (Aceleraciones?)(Centroide)
    Pzs = np.zeros((N,3))
    Pzu = np.eye(N)
    # d: entradas de Pzu
    d = np.empty(shape=[N,1])
    for j in range (N-1):
        Pzs[j] = np.array([1, j*T, (j**2 * T**2)/2 - h/g])    
        d[j] = (T**3)/6 + (j*T**3)/2 + ((j**2)*(T**3))/2 - (h*T)/g    
        for l in range(N-1):
            if j < l:
                Pzu[j,l] = 0
            else:
                Pzu[j,j-l] = d[l]   
    #print("Pzs:",Pzs)
    #print("\nPzu:",Pzu)
    
    #posición del pie en el plano
    x_fc=0
    y_fc=0
    
    #Velocidad de referencia
    x_ref = np.zeros(N)
    y_ref = np.zeros(N)
    
    x_f = np.zeros((3*m))
    y_f = np.zeros((3*m))
    
    '''
    #Formula(18)
    ux =np.append(x_3, x_f)
    uy=np.append(y_3, y_f)
    uk=np.append(ux, uy)
    #print("uk: ", uk)
    '''
    ux =np.append(x_3, x_fc)
    uy=np.append(y_3, y_fc)
    uk=np.append(ux, uy)

    
    for i in range(n*m+1): 
            #Formula (22)
            #u_c, U inicial 
            u_c=np.zeros(N)
            U=np.ones((N, m))
            # cosntrucción de u_c (vector)
            u_c[i:(i+n)]=np.ones(n)
            if i in range(n):
                #construcción de U, columnas de u_c
                U[0:(n+i), 0]=np.zeros(n+i)
                U[(i+n*m):(n*(m+1)), 0] =np.zeros(n*(m+1)-(i+n*m))
                # segunda columna de U
                U[i:(i+2*n), 1] = np.zeros(2*n)
            if i in range(n,2*n+1):
                #construct U_c column 1
                U[(i-n):(i+n), 0] = np.zeros(2*n)
                # construct U_c column 2
                U[0:(i-n), 1] = np.zeros(i-n)
                U[i:(n*(m+1)), 1] = np.zeros((m+1)*n -i)
            #print("UC: ", u_c)
            #print("U: ", U)
            
            I=np.identity(N)
            #Formula (24)
            Q_kprime11 = alpha *I + beta * np.dot(Pvu.T, Pvu) + gamma * np.dot(Pzu.T, Pzu)
            Q_kprime12 = - gamma *Pzu.T@U
            Q_kprime21 = - gamma *U.T@Pzu
            Q_kprime22 = - gamma *U.T@U
            Q_kprime1 = np.hstack((Q_kprime11, Q_kprime12))
            Q_kprime2 = np.hstack((Q_kprime21, Q_kprime22))
            Q_kprime = np.vstack((Q_kprime1, Q_kprime2))
            print("Qprime: ", Q_kprime.shape)
            
            Qk = np.kron(np.eye(2),Q_kprime)  #Formula (23)
            print("Qk: ", Qk.shape)
            
            #print("a: ", np.dot(u_c, x_fc) )
            #print("a: ", np.dot(Pzu, (np.dot(Pzs, x_hat[:,i])-np.dot(u_c, x_fc))))
            # formula (25)
            pk1=beta*np.dot(Pvu, np.dot(Pvs, x_hat[:,i])-x_ref)+gamma*np.dot(Pzu.T, (np.dot(Pvs, x_hat[:,i])-np.dot(u_c, x_fc)))
            #pk1 = Pk1.reshape(-1, 1)
            pk2 = - gamma* np.dot(U.T,np.dot(Pzs, x_hat[:,i]) - np.dot(Pzs, x_hat[:,i]))
            #Pk2 = Pk2.reshape(-1, 1)
            pk3=beta*np.dot(Pvu, np.dot(Pvs, y_hat[:,i])-y_ref)+gamma*np.dot(Pzu.T, (np.dot(Pvs, x_hat[:,i])-np.dot(u_c, y_fc)))
            #pk3 = pk3.reshape(-1, 1)
            pk4 = - gamma* np.dot(U.T,np.dot(Pzs, y_hat[:,i]) - np.dot(Pzs, y_hat[:,i]))
            #pk4 = pk4.reshape(-1, 1)
            #pk  = np.vstack((pk1, pk2, pk3, pk4))
            pkx=np.append(pk1, pk2)
            pky=np.append(pk3, pk4)
            pk=np.append(pkx, pky)
            #print("pk: ", pk)
    print("Q: ", Qk)
    print("uk: ", U.shape)
    return(u_c, Qk, pk)

#Función objetivo
def funcion(uk, Q, pk):
    f=0.5*np.dot(uk, np.dot(Q, uk))+np.dot(pk.T, uk)
    return(f)

def gradiente (uk, Q, pk):
    Qx=np.dot(Q, uk)
    bt=np.transpose(pk)
    g=Qx-bt
    #print("Q: ", Q)
    #print("\n bt; ", bt)
    return(g)

def sdescent(uk, funcion, gradiente, Q, pk):
    tol = 1e-7
    x=uk
    b=pk
    #print("here001 ", pk)
    gk=gradiente(x, Q, b)
    #print("g:" ,gk)
    gn=np.linalg.norm(gk) #norma del gradiente
    k=0
    #print("x: ", x)
    while gn>tol and k<100000: 
        alpha=0.01 #tamaño de paso fijo
        #print("xh: ", x)
        x=x-(alpha*gk) #calculo del nuevo x
        #print("x1: ", x)
        gnew=gradiente(x, Q, b) #calculo del gradiente con la nueva x
        #gn=np.linalg.norm(gnew) #norma del gradiente
        k+=1 #contador 
        print("||g(x*)||: ", gn)
        if np.linalg.norm(gnew-gk)<tol:
            break
        gk = gnew
        gn = np.linalg.norm(gk)
    print("||g(x*)||:", np.linalg.norm(gradiente(x, Q, b)))
    print("Iteraciones:", k)
    #print("x: ", x)
    return(x)

def barzilai(uk,Q,b,g):
    max_iter = 100
    tol = 1e-4
    xk=uk
    yk=uk
    gk=g(xk,Q,b)
    k = 0
    alpha0 = 0.5
    xnew = xk - alpha0 * gk
    gnew = g(xnew,Q,b)
    print(gnew)
    sk = xnew-xk
    yk = gnew-gk   
    gkv = []
    aux1 = 0
    aux2 = 0
    aux3 = 0
    while np.linalg.norm(gk)>tol and k<max_iter:  
        alphak = (np.dot(sk,yk))/(yk.T@yk) 
        if np.isnan(alphak):
            alphak = 9
        print("alpha:",alphak)
        #print(alphak)
        #print(alphak)
        xnew = xk - alphak*gnew
        #Actualizo
        gnew = g(xnew,Q,b)
        sk = xnew-xk
        yk = gnew-gk
        xk = xnew  
        if np.linalg.norm(gnew-gk)<tol:
            break
        gk = g(xk,Q,b)
        gkn = np.linalg.norm(gk)
        print(gkn)
        k+=1
    #print("||g(x*)||: ",la.norm(gk))
    #print("Iter: ", k)
    print("||gk(x*)|| = ",np.linalg.norm(g(xk,Q,b)))
    return xk

#MAIN
#Valores iniciales
N=16
T=0.1
m=2
#Matrices de X^{...} y Y^{...}
x_3=np.zeros((N,1))
y_3=np.ones((N,1))
uk =np.append(x_3,y_3) #Formula(13)
#uk, Q, pk=nuevoPaso(N, T, uk)

uk, Q, pk=nuevoPaso2(N, T, uk)
#uk=conjugado(Q, uk, pk)
#uk=barzilai(uk, Q, pk, gradiente)
#uk=sdescent(uk, funcion, gradiente, Q, pk)
#print(np.isnan(8))
print("uk: ", uk)






#'''