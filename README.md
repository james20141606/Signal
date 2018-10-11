# Mixture Density Network for joint position coordinates prediction

Also see my [website page](https://www.cmwonderland.com/blog/2018/10/10/94-signal-project/)

### model
We use pytorch and tensorflow to develop our mixture density network. 
- A MLP is used to generate pi, mu and sigma for 2D isotropic gaussian distribution.
- Several 2D gaussian distribution is mixed to form a mixed gaussian distribution.
![](https://github.com/james20141606/Signal/blob/master/plot/gaussian_mixture.png)
- We use Maximum Likelihood Estimation, choose negetive log likelihood as our loss function for optimization
![](https://github.com/james20141606/Signal/blob/master/plot/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202018-10-05%20%E4%B8%8B%E5%8D%8810.04.29.png)


Several tricks:
- deal with loss nan: we use log transform of signa, mu and pi, also a cutoff of sigma and pseudo counts of pi is used to prevend loss explosion of vanishing
- z score normalization to optimize the model easier.
- we use mean shift to find modes in gaussian mixture

![](https://github.com/james20141606/Signal/blob/master/plot/mode_finding.png)
### result
![](https://github.com/james20141606/Signal/blob/master/plot/prediction_gt.png)
### data
#### mountain data
- data position
![](https://github.com/james20141606/Signal/blob/master/plot/split_data.png)
- t-SNE to cluster data
![](https://github.com/james20141606/Signal/blob/master/plot/t-SNE.png)
- feature distribution
![](https://github.com/james20141606/Signal/blob/master/plot/3D_surface_of_feature_00.png)
####  city data
- data position
![](https://github.com/james20141606/Signal/blob/master/plot/citydata.png)
- receriver position
![](https://github.com/james20141606/Signal/blob/master/plot/city_rx.png)
- transmitter position
![](https://github.com/james20141606/Signal/blob/master/plot/city_tx.png)

#### explore useful features
We use PCC to quantify the PCC between samples' features and distance. We use dynamic weight to pick features having higher relation with distance. It seems TOA has the significant higher weight
![](https://github.com/james20141606/Signal/blob/master/plot/weight_change_10.gif)

We plan to use some imputation method, including some methods from single cell analysis. We also aim to use RNN for changeable size feature and attention model to pick more related features.
