import math
import numpy as np
def get_probability(x,y,pi,mu,sigma):
	l=np.full([2,1],1)
	p=0
	for i in range(len(mu)):
		x_u=math.erf((x+l[0]-mu[i][0])/(sqrt(2)*sigma[i][0]))
		x_l=math.erf((x-l[0]-mu[i][0])/(sqrt(2)*sigma[i][0]))
		y_u=math.erf((y+l[1]-mu[i][1])/(sqrt(2)*sigma[i][1]))
		y_l=math.erf((y-l[1]-mu[i][1])/(sqrt(2)*sigma[i][1]))
		p=p+pi[i]*0.25*(x_u-x_l)*(y_u-y_l)
	return p
