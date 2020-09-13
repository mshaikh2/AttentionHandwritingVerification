# Attention based Handwriting Verification
Conference paper presented in ICFHR2020, Dortmund, Germany
<p>https://arxiv.org/abs/2009.04532</p>

# Prediction Example
<img alt="corrrespondence maps" src="/figures/PredictionExample.png" width="25%"/>

# More Results
## Correspondence region maps from Cross Attention
<img alt="corrrespondence maps" src="/figures/keypoints.png" width="50%"/>
<ol type="a">
<li>Images from same writer where the model is able to locate some points corresponding to query</li>
<li>Images from different writers where the model is unable to locate any point similar to queried pixel region</li>
</ol>

## Salient region maps from Soft Attention
<img alt="saliency maps" src="/figures/saliency.png" width="50%"/>
<ol type="a">
<li>Salient regions when the model assumes that the writers are same. Model highlights the similar looking regions.</li>
<li>Salient regions when the model assumes that the writers are different. Model highlights the dissimilar looking regions.</li>
</ol>

# Citation
- Coming soon

# Datasets
- Handwritten "AND": https://github.com/mshaikh2/HDL_Forensics
- CEDAR Signature: http://www.cedar.buffalo.edu/NIJ/data
