# The Leverage Experimentalist

This experimentalist uses the statistical concept of leverage by refitting the provided models iteratively with the leave-one-out method. 

---
WARNING: 
This experimentalist needs to fit each model you provide it n times, where n corresponds to the number of datapoints you have. 
As such, the computational time and power needed to run this experimentalist increases exponentially with increasing number of models and datapoints.

---

In each iteration, it computes the degree to which the currently removed datapoint has influence on the model. 
If the model remains stable, the datapoint is deemed to have little influence on the model, and as such will have a low likelyhood of being selected for further investigation.
In contrast, if the model changes, the datapoint is influential on the model, and has a higher likelihood of being selected for further investigation.

Specifically, you provide the experimentalist with a model that has been trained on all of the data. On each iteration, the experimentalist fits a new model with all data aside from one datapoint. 
Both models ($m$) then predict Y scores ($Y'$) from the original X variable and compute a mean squared error (MSE) for each X score ($i$):

$$
MSE_{m,i} = \sum(Y'_{m,i} - Y_{i})^{2} 
$$    

The experimentalist then computes a ratio of the MSE scores between the experimentalist model and the original model that you provided:

$$
{MSE_{Ratio}}_{m,i} = {MSE_{experimentalist}}_{m,i}/{MSE_{original}}_{m}
$$
As such, values above one indicates that the original model fit the data better than the experimentalist model when removing that datapoint ($i$).
In contrast, values below one indicates that the experimentalist model fit the data better than the original model when removing that datapoint ($i$).
And a value of one indicates that both models fit the data equally. If you provide multiple models, it will then average across these models to result in an aggregate MSE score for each X score. In the future, it might be a good idea to incorporate multiple models in a more sophisticated way.

Finally, the experimentalist then uses these aggregated ratios to select the next set of datapoints to explore in one of three ways, declared with the 'fit' parameter.
    -'increase' will choose samples focused on X scores where the fits got better (i.e., the smallest MSE ratios)
    -'decrease' will choose samples focused on X scores where the fits got worse (i.e., the largest MSE ratios)
    -'both' will do both of the above, or in other words focus on X scores with the most extreme scores.

