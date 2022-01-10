# 机器学习
1. notes1
    * Supervised Learning
    * x input features
    * y output or target variable
    * (x, y) training example
    * {(x,y)} training set
    * h : x -> y, hypothesis
    * y is continuous -> regression else -> classification
    * Part I Linear Regressioin
    * $\theta$ parameters/weights
    * make $x_0$ = 1 intercept term
    * cost function
    * 1 LMS algorithm
    * $\alpha$ learning rate
    * LMS update rule (Widrow-Hoff)
    * $\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)}-h(x^{(i)}))x^{(i)}$
    * batch gradient descent
    * stochastic gradient descent (incremental gradient descent)
    * close to the minimum faster
    * may not converge (can solve by decreasing $\alpha$)
    * 2 The Normal equations
    * $\nabla_Af(A)_{ij}=\frac{\partial f}{\partial A_{ij}}$
    * $a^Tb=b^Ta$
    * $\nabla_xb^Tx=b$
    * $\nabla_xx^TAx=2Ax$ for symmetric matrix A
    * $X^TX\theta=X^Ty$ so $\theta=(X^TX)^{-1}X^Ty$
    * 3 Probabilistic interpretation 
    * error term ($\epsilon$) are distributed IID (independently and identically distributed) according to a Gaussian distribution