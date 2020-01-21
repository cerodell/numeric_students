## Chirs Rodell
### Lab2

#### Q1

- a\) 
$$
y_{i-1}=y_{i}-h y_{i}^{\prime}+\frac{h^{2}}{2 !} y_{i}^{\prime \prime}-\frac{h^{3}}{3 !} y_{i}^{\prime \prime \prime}+\frac{h^{4}}{4 !} y_{i}^{\prime \prime \prime \prime}-\ldots
$$

$$
y_{i}^{\prime}=\frac{y_{i}-y_{i-1}}{h}+\mathcal{O}(h)
$$


$$
\mathcal{O}(h) \cong \frac{h}{2} y_{i}^{\prime \prime}
$$

- b\)
The leading term (constant h) is negative for backward Euler which is different from forwarding. This leads to an overestimation that becomes less as you include higher-order calculations.

#### Q2

- a\)
In this case yes, increasing the number of times steps does increase the accuracy (as seen in the 1000 number of time steps plot).Though there is a limit to how small dt can be, beyond which round-off errors will start polluting the computation. 



- b\) 
$$\text { global error }=\left|T\left(t_{n}\right)-T_{n}\right|$$

$$
\text {See Error.png in zip file}
$$

- c\)
Δtp is much smaller for large values of p. Thus large values of p result in small errors, and in general, numerical methods with high-order converge to the exact solution more quickly than do methods with low-order as long as the exact solution is sufficiently differentiable. 
$$
c \cdot(\Delta t)^{p}
$$

 

#### Q3

- a\) As the dt becomes smaller the stability increase. You can see this in all the methods. However, leapfrog and midpoint become unstable first.

- b\) You can see that beuler is stable while leapfrog is not.


#### Q4

$$
\frac{d z}{d t}=\lambda z
$$

$$
\frac{z_{i}-z_{i-1}}{\Delta t}=\lambda z_{i}
$$

$$
\frac{z_{i}}{z_{i} \Delta t}-\frac{z_{i-1}}{z_{i} \Delta t}=\lambda
$$

$$
\frac{1}{\Delta t}-\frac{z_{i-1}}{z_{i} \Delta t}=\lambda
$$

$$
\frac{1}{\Delta t}- \lambda=\frac{z_{i-1}}{z_{i} \Delta t}
$$

$$
1-\lambda \Delta t=\frac{z_{i-1}}{z_{i}}
$$

$$
z_{i}=\frac{z_{i-1}}{1-\lambda \Delta t}
$$

$$
z_{i+1}=\frac{z_{i}}{1-\lambda \Delta t}
$$

$$
z_{i}=\left(\frac{1}{1-\lambda \Delta t}\right)^{i} \hat{z}
$$

$$
\left|\frac{1}{1-\lambda \Delta t}\right|<1
$$

$$
\text { becuase }
$$

$$
\Delta t>0
$$

$$
\text { time can't go in reverse }
$$

$$
\text { &}
$$

$$
\lambda<0
$$

$$
\text { so}
$$
$$
\Delta t\lambda<0
$$

$$
\text { Thus, backward Euler method is unconditionally
stable.}
$$

#### Q5

- a\) 


$$
y\left(t_{i+1}\right)=y\left(t_{i}\right)+y^{\prime}\left(t_{i}\right) \Delta t+\frac{y^{\prime \prime}\left(t_{i}\right)}{2 !}(\Delta t)^{2}+\frac{y^{\prime \prime \prime}\left(t_{i}\right)}{3 !}(\Delta t)^{3}+\frac{y^{\prime \prime \prime}\left(t_{i}\right)}{4 !}(\Delta t)^{4} \ldots(1)
$$


$$
\text { where } 
$$

$$
t_{i+1}=t_{i}+\Delta t
$$

$$
y\left(t_{i-1}\right)=y\left(t_{i}\right)-y^{\prime}\left(t_{i}\right) \Delta t+\frac{y^{\prime \prime}\left(t_{i}\right)}{2 !}(\Delta t)^{2}-\frac{y^{\prime \prime \prime}\left(t_{i}\right)}{3 !}(\Delta t)^{3}+\frac{y^{\prime \prime \prime \prime}\left(t_{i}\right)}{4 !}(\Delta t)^{4} \ldots \text { (2) }
$$

$$
\text { where } 
$$

$$
t_{i-1}=t_{i}-\Delta t
$$

$$
\text { Adding equations}(1)\text { and }(2), \text { gives }
$$

$$
y\left(t_{i+1}\right)+y\left(t_{i-1}\right)=2 y\left(t_{i}\right)+y^{\prime \prime}\left(t_{i}\right)(\Delta t)^{2}+y^{\prime \prime \prime}\left(t_{i}\right) \frac{(\Delta t)^{4}}{12}
$$

$$
y^{\prime \prime}\left(t_{i}\right)=\frac{y\left(t_{i+1}\right)-2 y\left(t_{i}\right)+y\left(t_{i-1}\right)}{(\Delta t)^{2}}-\frac{y^{\prime \prime \prime}\left(t_{i}\right)(\Delta t)^{2}}{12}
$$

$$
y^{\prime \prime}\left(t_{i}\right)=\frac{y\left(t_{i+1}\right)-2 y\left(t_{i}\right)+y\left(t_{i-1}\right)}{(\Delta t)^{2}}+\mathcal{O}(\Delta t)^{2}
$$




- b\) 
Same as above but for (1)

$$
t_{i+2}=t_{i}+2 \Delta t
$$

and for (2)

$$
t_{i-2}=t_{i}-2 \Delta t
$$

$$
y^{\prime \prime \prime}\left(t_{i}\right)=\frac{y\left(t_{i+2}\right)-2y\left(t_{i+1}\right) + y\left(t_{i-1}\right) - y\left(t_{i-2}\right)}{2(\Delta t)^{3}}+\mathcal{O}(\Delta t)^{3}
$$



