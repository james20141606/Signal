Mixture Density Network for joint position coordinates prediction

We use pytorch and tensorflow to develop our mixture density network. 
- A MLP is used to generate pi, mu and sigma for 2D isotropic gaussian distribution.
- Several 2D gaussian distribution is mixed to form a mixed gaussian distribution.
- We use Maximum Likelihood Estimation, choose negetive log likelihood as our loss function for optimization



Several tricks:
- deal with loss nan: we use log transform of signa, mu and pi, also a cutoff of sigma and pseudo counts of pi is used to prevend loss explosion of vanishing
- z score normalization to optimize the model easier.
- we use mean shift to find modes in gaussian mixture
