# Loss Curvature Matching for Dataset Selection and Condensation (LCMat) (AISTATS 2023 Poster)
**[Seungjae Shin*](https://sites.google.com/view/seungjae-shin),
[HeeSun Bae*](https://sites.google.com/view/baeheesun),
[Donghyeok Shin](https://aailab.kaist.ac.kr/xe2/members_phdstudent/20976),
[Weonyoung Joo](https://scholar.google.co.kr/citations?user=r2eJgW4AAAAJ&hl=ko&oi=ao),
and [Il-Chul Moon](https://aailab.kaist.ac.kr/xe2/members_professor/6749).**

## Overview

This paper introduces **Loss Curvature Matching (LCMat)**, that the infers an optimal dataset by matching the worst loss-curvature gap between the original dataset and the reduced dataset.

#### Paper description and Main idea:

Existing methods utilize the neural network during the dataset reduction procedure, so the model parameter becomes important factor in preserving the performance after reduction. By depending upon the importance of parameters,this paper introduces a new reduction objective, coined LCMat, which Matches the Loss Curvatures of the original dataset and reduced dataset over the model parameter space, more than the parameter point. This new objective induces a better adaptation of the reduced dataset on the perturbed parameter region than the exact point matching. Particularly, we identify the worst case of the loss curvature gap from the local parameter region, and we derive the implementable upper bound of such worst-case with theoretical analyses. 

<p align="center">
  <img 
    width="700"
    src="https://user-images.githubusercontent.com/105624747/219567990-beb0cbb7-0ebb-44bd-957f-7182a79af8ab.png"
  >
</p>

Here, $\theta$ is a parameter, $\rho$ denotes the maximum size of the perturbation. LCMat matches the Loss Curvature of the training dataset, $T$, and the resulting dataset, $S$, by reducing the gap between the curvature of $\mathcal{l}(T)$ and that of $\mathcal{l}(S)$.

By considering the sharpness on loss difference, LCMat(right) can successfully identify the reduced dataset $S$ matching the loss landscape of original $T$, although the subset selected by Craig(left) does not match the loss curvature of $T$.

<p align="center">
  <img 
    width="700"
    src="https://user-images.githubusercontent.com/105624747/219572052-fded6505-861d-4db6-bf4e-f9cde1c5a71b.png"
  >
</p>

#### Contribution:
LCMat provides the following advantages over the existing models.
* LCMat derives an implementable upper bound of the sharpness, which results in an objective of LCMat.
* LCMat adaptively transform the objective into the function of either selection or condensation objective, so LCMat becomes the fundamentally applicable mechanism for dataset reduction.
* We conduct experiments over the evaluation scenarios with different benchmark datasets, and we confirm that LCMat shows clear merit when the reduction ratio becomes significant and when the evaluation scenario becomes complex, e.g. continual learning.

## Setup
Please install required libraries as follows.

We kindly suggest other researchers to run this code on `python = 3.8` version.
```
pip install -r requirements.txt
```

## Training and Evaluation
For reproduce the results of LCMAT-S, we provide a bash file for running `main.py`, which located at: 
```
/bash/LCMat_XXX.sh
```
Here, XXX is dataset. You can get results in `result/` directory.

You can also reproduce cross-architecture generalization result by running `cross_network_generalization.py`.

## References
If you find the code useful for your research, please consider citing
```bib 
@InProceedings{pmlr-v206-shin23a,
  title = 	 {Loss-Curvature Matching for Dataset Selection and Condensation},
  author =       {Shin, Seungjae and Bae, Heesun and Shin, Donghyeok and Joo, Weonyoung and Moon, Il-Chul},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {8606--8628},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/shin23a/shin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/shin23a.html},
  abstract = 	 {Training neural networks on a large dataset requires substantial computational costs. Dataset reduction selects or synthesizes data instances based on the large dataset, while minimizing the degradation in generalization performance from the full dataset. Existing methods utilize the neural network during the dataset reduction procedure, so the model parameter becomes important factor in preserving the performance after reduction. By depending upon the importance of parameters, this paper introduces a new reduction objective, coined LCMat, which Matches the Loss Curvatures of the original dataset and reduced dataset over the model parameter space, more than the parameter point. This new objective induces a better adaptation of the reduced dataset on the perturbed parameter region than the exact point matching. Particularly, we identify the worst case of the loss curvature gap from the local parameter region, and we derive the implementable upper bound of such worst-case with theoretical analyses. Our experiments on both coreset selection and condensation benchmarks illustrate that LCMat shows better generalization performances than existing baselines.}
}
 ```

## Acknowledgements

This research was supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data(IITP) funded by the Ministry of Science and ICT(2022-0-00077).
