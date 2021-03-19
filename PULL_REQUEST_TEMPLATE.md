Divisive Hierarchical Clustering Algorithm added

#### References to other Issues or PRs or Relevant literature
<!-- If this pull request fixes an issue, write "Fixes #NNNN" in that exact
format, e.g. "Fixes #1234". See
https://github.com/blog/1506-closing-issues-via-pull-requests
Please also write a comment on that issue linking back to this pull request once it is
open. -->
Fixes #94

#### Brief description of what is fixed or changed
- Added divisive hierarchical clustering algorithm that uses sse as selection criteria and euclidean distance as the split measure. The distances between two clusters are found using 'median' (distance between the average of two centroids from another centroid). It works on data having redundant examples as well. #features in the dataset was taken as 2. T.C.: O(#clusters)
- A vectorized version of kmeans was also implemented that, I think, runs in O(#clusters). 

#### Other comments
Some improvements can be done like making the code in utils.allocate_clusters more better, introducing different selection, split measures and distance criteria, etc. I have added notable references in the utils file.  