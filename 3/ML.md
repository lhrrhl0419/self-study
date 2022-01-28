# 机器学习
1. **notes1**
    * Supervised Learning
    * x input features
    * y output or target variable
    * (x, y) training example
    * {(x,y)} training set
    * h : x -> y, hypothesis
    * y is continuous -> regression else -> classification
    * ***Part I Linear Regressioin***
    * $\theta$ parameters/weights
    * make $x_0$ = 1 intercept term
    * cost function
    * **1 LMS algorithm**
    * $\alpha$ learning rate
    * LMS update rule (Widrow-Hoff)
    * $\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)}-h(x^{(i)}))x^{(i)}$
    * batch gradient descent
    * stochastic gradient descent (incremental gradient descent)
    * close to the minimum faster
    * may not converge (can solve by decreasing $\alpha$)
    * **2 The Normal equations**
    * $\nabla_Af(A)_{ij}=\frac{\partial f}{\partial A_{ij}}$
    * $a^Tb=b^Ta$
    * $\nabla_xb^Tx=b$
    * $\nabla_xx^TAx=2Ax$ for symmetric matrix A
    * $X^TX\theta=X^Ty$ so $\theta=(X^TX)^{-1}X^Ty$
    * **3 Probabilistic interpretation** 
    * error term $\epsilon$ are distributed IID (independently and identically distributed) according to a Gaussian distribution
    * $p(y|x;\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y-\theta^Tx)^2}{2\sigma^2})$
    * likelihood function $L(\theta)=L(\theta;X,y)=p(y|X;\theta)$
    * choose $\theta$ to maximize $L(\theta)$
    * log likelihood $l(\theta)=log(L(\theta))=nlog\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{2\sigma^2}\cdot\sum_{i=1}^n(y-\theta^Tx)^2$
    * so maximizing $l(\theta)$ is minimizing $J(\theta)$
    * **4 Locally weighted linear regression(LWR)**
    * underfitting and overfitting
    * weight $w^{(i)}=exp(-\frac{(x^{(i)}-x)^2}{2\tau^2})$ where $\tau$ is the bandwidth parameter
    * non-parametric : the amount of stuff we need to keep in order to represent the hypothesis h grows linearly with the size of the training set
    * ***Part II Classification and Logistic Regression***
    * **5 Logistic Regression**
    * as $y \in {0,1}$, h can be set as $h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$ where $g(z)=\frac{1}{1+e^{-z}}$ is called the logistic function or the sigmoid function
    * $g'(z)=g(z)(1-g(z))$
    * $P(y=1|x;\theta)=h_\theta(x)$ $P(y=0|x;\theta)=1-h_\theta(x)$ can be written as $p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}$
    * log likelihood $l(\theta)=log L(\theta)=\sum_{i=1}^ny^{(i)}log h(x^{(i)})+(1-y^{(i)})log(1-h(x^{(i)}))$
    * **6 Digression: The Perceptron Learning Algorithm**
    * $g(z)=\left\{\begin{aligned} 1 if z\eqr 0 \\ 0 if z < 0 \end{aligned}\right.$