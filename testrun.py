import numpy as np
import matplotlib.pyplot as plt
import math,copy

x_train=np.array([1.0,1.4,1.8,2.0])
y_train=np.array([300.0,400.0,450.0,500.0])

def compute_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=cost+(f_wb-y[i])**2
    total_cost=1/(2*m)*cost
    return total_cost

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw,dj_db=0,0

    for i in range(m):
        f_wb=w*x[i]+b
        dj_dw_i=(f_wb-y[i])*x[i]
        dj_db_i=f_wb-y[i]
        dj_db+=dj_db_i
        dj_dw+=dj_dw_i

    dj_db=dj_db/m
    dj_dw=dj_dw/m

    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    J_history=[]
    p_history=[]
    b=b_in
    w=w_in

    for i in range(num_iters):
        dj_dw,dj_db=gradient_function(x,y,w,b)
        b=b-alpha*dj_db
        w=w-alpha*dj_dw

        if i<100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])

        return w,b,J_history,p_history


w_in=0
b_in=0
iter=10000
tmp_alpha=1.0e-2

w_final,b_final,J_his,P_his=gradient_descent(x_train,y_train,w_in,b_in,tmp_alpha,iter,compute_cost,compute_gradient)

print(f"(w,b) found as ({w_final:8.4f}, {b_final:8.4f})")

print(f"1000 sq.ft house prediction {w_final*1.0+b_final:0.1f} thousand dollars")

plt.scatter(w_final,b_final,c='r',label='Predicted')
plt.legend()
plt.show()
#plt.plot(1000+np.array(len(J_his[1000:])),J_his[1000:])
plt.plot(J_his[1000:])
plt.show()