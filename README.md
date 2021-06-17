# GDL-Project2021

## Part I Approach 2
We perform Graph reduction using the Projection Operator from AMG and we basically used all the code already implemented from the original repo of DiffPool in order to be sure to have fair comparisons.


## Part I Approach 3
We implemented a new 'model', with the same structure of the softPool from the original repo, we only edited the pooling layer: we used in fact the Restriction operator from AMG.
- cross_val.py: here we precompute the Prolongation operator, both for the training set and the validation set.
- easyAMG: naive implementation of Algebraic Multigrid restriction operator.
- econders.py: contains the amg-assign model.
- load_data, graph_sampler, partition: util files necessary to upload the dataset.


## CREDIT
For all the parts the code was heavly based on https://github.com/RexYing/diffpool
