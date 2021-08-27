# stochastic-control

Linear control with additive noise.

As an example, we consider the control problem for the case of controlled drift-diffuation:

```math
dx = f (x, u, t)dt + G(x, t)dw
```

```math
f(x,t) = Ax+Bu,    G(x,t)=G
```

```math
g(x,u,y) = x^T Q x+u^T R u 
```
We can show that stochastic control law has the form of:

```math
u(t) = −R^{-1} BT P(t) x(t)
```

And the Ricatti equation for P(t) is:

```math
−\dot{P} = A^T P+PA+Q−PBR^{−1}BTP,      P(t_f)=M
```
