#!/usr/bin/env python
# coding: utf-8

# # More 3D Coordinate Transformations
# 
# We have already shown how Python is capable of performing and visualising the standard set of operations for 3D graphics transformations.
# 
# There are several additional useful operations than can be performed by matrices. This notebook will examine two of these: *reflection in a given plane* and *rotation about an axis*.

# In[1]:


## Libraries
import numpy as np
import math 
import matplotlib.pyplot as plt
import sympy as sym
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# ## Rotation about an arbitrary axis
# 
# Another useful tool in our 3D transformation arsenal is the ability to rotate a point (or set of points) by a specified angle about a specified axis.
# 
# In the lectures we learned the procedure to carry out this task:
# 
# 
# 1.   Translate the object to the origin.
# 2.   Align the rotation axis with the $z-$coordinate ($k$).
# 3.   Rotate the object by angle $\theta$ about $k$-axis.
# 4.   Un-align rotation axis and $k$.
# 5.   Translate object back to original position.
# 
# Mathematically, we can express these steps as follows:
# 
# $$
#  R_{\theta,L} = T_{-\underline{\bf{p}}} A_{{\underline{\bf{v}}},{\underline{\bf{k}}}} R_{\theta, {\underline{\bf{k}}}} \left(A_{{\underline{\bf{v}}},{\underline{\bf{k}}}}\right)^T T_{\underline{\bf{p}}}
# $$
# 
# where 
# 
# * $R_{\theta,L}$ is the rotation of an object through an angle $\theta$ about an axis ${\bf{v}}$,
# * $T_{\underline{\bf{-p}}}$ translates one point on the object, ${\bf{p}}$, to the origin,
# * $A_{{\underline{\bf{v}}},{\underline{\bf{k}}}}$ aligns the rotation axis with the $z$-direction,
# * $R_{\theta, {\underline{\bf{k}}}}$ performs the rotation about $z$ by angle $\theta$,
# * $\left(A_{{\underline{\bf{v}}},{\underline{\bf{k}}}}\right)^T$ re-orients the rotation vector back to its original direction ${\bf{v}}$,
# * $T_{\underline{\bf{p}}}$ moves the object back to original position ${\bf{p}}$.
# 
# Each of these steps is carried out using matrix operations, hence this whole operation is a concatenation of all these matrices.

# The fundamental idea here is that we already know how to rotate things about the z-axis:
# 
# $$
# R_{\theta_{z,\underline{\bf{k}}}}=
# 	\begin{pmatrix}
# 	\cos(\theta) & \sin(\theta) & 0  & 0 \\
# 	-\sin(\theta) & \cos(\theta) & 0 & 0 \\
# 	0 & 0 & 1 & 0 \\
#   0 & 0 & 0 & 1
# 	\end{pmatrix}.
# $$
# 
# Such a rotation would need to occur at the origin, so the familiar translation stages are required:
# $$
# T_{-\underline{p}} = 
# 	{\begin{pmatrix}
# 	1 & 0 & 0 & 0  \\
# 	0 & 1 & 0 & 0  \\
# 	0 & 0 & 1 & 0  \\
# 	-p_x & -p_y & -p_z & 1 
# 	\end{pmatrix}},~~~
# T_{\underline{p}} =
# 	\begin{pmatrix}
# 	1 & 0 & 0 & 0  \\
# 	0 & 1 & 0& 0  \\
# 	0 & 0 & 1 & 0 \\
# 	p_x & p_y & p_z & 1 
# 	\end{pmatrix}.	
# $$
# 
# The question then becomes, how to align our rotation axis with the z-axis?
# 
# The following matrix (found in the formula book) can be readily used to align a vector ${\underline{\bf{v}}}=a{\underline{\bf{\hat{i}}}}+b{\underline{\bf{\hat{j}}}}+c{\underline{\bf{\hat{k}}}}$ with ${\underline{\bf{k}}}\left(=0{\underline{\bf{\hat{i}}}}+0{\underline{\bf{\hat{j}}}}+1{\underline{\bf{\hat{k}}}}\right)$:
# 
# $$
# A_{{\underline{\bf{v}}},{\underline{\bf{k}}}}=
# 	\begin{pmatrix}
# 	\frac{ac}{\lambda|{\underline{\bf{v}}}|} & -\frac{b}{\lambda} & \frac{a}{|{\underline{\bf{v}}}|} & 0 \\
# 	\frac{bc}{\lambda|{\underline{\bf{v}}}|} & \frac{a}{\lambda} & \frac{b}{|{\underline{\bf{v}}}|} & 0 \\
# 	-\frac{\lambda}{|{\underline{\bf{v}}}|}  & 0                 & \frac{c}{|{\underline{\bf{v}}}|} & 0 \\
#     0 & 0 & 0 & 1
# 	\end{pmatrix},~~~~ \lambda = \sqrt{a^2+b^2},~~~~~ {\underline{\bf{v}}}=\sqrt{a^2+b^2+c^2}.
# $$
# 
# To undo this operation, we apply the *transpose* of this matrix:
# 
# $$
# \left(A_{{\underline{\bf{v}}},{\underline{\bf{k}}}}\right)^T=
# 	\begin{pmatrix}
# 	\frac{ac}{\lambda|{\underline{\bf{v}}}|} & \frac{bc}{\lambda|{\underline{\bf{v}}}|} & -\frac{\lambda}{|{\underline{\bf{v}}}|} & 0 \\
# 	-\frac{b}{\lambda} & \frac{a}{\lambda} & 0 & 0 \\
# 	\frac{a}{|{\underline{\bf{v}}}|}  & \frac{b}{|{\underline{\bf{v}}}|}                 & \frac{c}{|{\underline{\bf{v}}}|} & 0 \\
#     0 & 0 & 0 & 1
# 	\end{pmatrix},
# $$
# 
# which in practise swaps each row of a matrix for its equivalent column.
# 
# 

# In[2]:


from sympy import sin, cos, Matrix
from sympy.abc import rho, theta, a, b, c
p_x, p_y, p_z = sym.Symbol('p_x'), sym.Symbol('p_y'), sym.Symbol('p_z')

V = sym.Matrix([[a,b,c]])
V[0]


# In[3]:


from sympy import sin, cos, Matrix
from sympy.abc import rho, theta, a, b, c, kappa, upsilon
p_x, p_y, p_z = sym.Symbol('p_x'), sym.Symbol('p_y'), sym.Symbol('p_z')

#V = sym.Matrix([[a,b,c]])
#mV = sym.sqrt(V[0]*V[0]+V[1]*V[1]+V[2]*V[2])

#v = V /mV
v = sym.Matrix([[a,b,c]])
mv = sym.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
lam = sym.sqrt(v[0]*v[0]+v[1]*v[1])


R = Matrix([[cos(theta), sin(theta), 0, 0],
            [-sin(theta), cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
Tmp = Matrix([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-p_x, -p_y, -p_z, 1]])
Tp = Matrix([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [p_x, p_y, p_z, 1]])
Avk = Matrix([[v[0]*v[2]/lam/mv, -v[1]/lam, v[0]/mv, 0],
            [v[1]*v[2]/lam/mv, v[0]/lam, v[1]/mv, 0],
            [-lam/mv, 0, v[2]/mv, 0],
            [0, 0, 0, 1]])
AvkT = Avk.T
Avk


# In[4]:


R1 = sym.simplify(Avk*R*AvkT)


# In[5]:


R2 = R1.replace(a**2+b**2+c**2,1)


# In[6]:


sym.simplify(R2)


# In[7]:


R2[0,0]


# In[8]:


test = a*a*(1-cos(theta))+cos(theta)
test
elementdiff = sym.simplify(R2[0,0]-test)
#sym.simplify(elementdiff.replace(a**2+b**2+c**2,1)) == 0
#sym.simplify(R2[0,0].replace(a**2+b**2+c**2,1)-test) == 0
sym.simplify(elementdiff.replace(a**2+b**2+c**2,1))


# In[9]:


Rx = Matrix([[1, 0, 0, 0],
            [0,  cos(theta), sin(theta), 0],
            [0, -sin(theta), cos(theta), 0],
            [0, 0, 0, 1]]) 
Ry = Matrix([[cos(upsilon), 0, -sin(upsilon), 0],
            [0,  1, 0, 0],
            [sin(upsilon), 0, cos(upsilon), 0],
            [0, 0, 0, 1]])
Rx*Ry


# In[ ]:




