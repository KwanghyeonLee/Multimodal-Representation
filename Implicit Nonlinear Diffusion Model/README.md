## Maximum Likelihood Training of Implicit Nonlinear Diffusion Model (INDM) (NeurIPS 2022 Poster) <br><sub>Official PyTorch implementation of the Maximum Likelihood Training of Implicit Nonlinear Diffusion Model</sub>
**[Dongjun Kim](https://github.com/Kim-Dongjun) \*, [Byeonghu Na](https://github.com/byeonghu-na) \*, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, Il-Chul Moon**   
<sup> * Equal contribution </sup>
 
## Overview

<img src="./figures/overview.png" width="1000" title="overview" alt="INDM attains a ladder structure between the data space and the latent space. The latent vector is visualized by normalizing the latent value">

#### Paper description and Main Idea:
The nonlinearity effect has been hardly understood, but intuitively, there would be promising diffusion patterns to efficiently train the generative distribution towards the data distribution. This paper introduces a data-adaptive nonlinear diffusion process for score-based diffusion models. The proposed Implicit Nonlinear Diffusion Model (INDM) learns by combining a normalizing flow and a diffusion process. Specifically, INDM implicitly constructs a nonlinear diffusion on the data space by leveraging a linear diffusion on the latent space through a flow network with connecting by invertible function. This flow network is key to forming a nonlinear diffusion, as the nonlinearity depends on the flow network. This flexible nonlinearity improves the learning curve of INDM to nearly Maximum Likelihood Estimation (MLE) against the non-MLE curve of DDPM++, which turns out to be an inflexible version of INDM with the flow fixed as an identity mapping. This paper shows that the diffusion model learned with this non-linear diffusion process proceeds close to MLE training by overcoming the variational gap caused by robust sampling and ELBO that produces good images even if the sampling step is reduced.

#### Contribution:
INDM provides the following advantages over the existing models.
* INDM achieves fast and tractable optimization with implict modeling.
* INDM learns not only drift but volatility coefficients of the forward SDE.
* INDM trains its network with Maximum Likelihood Estimation (MLE).
* INDM is robust on the sampling discretization.

## Training and Evaluation

### 1) Requirements

This code was tested with CUDA 11.1 and Python 3.8.

```
pip install -r requirements.txt
```

### 2) Pretrained Checkpoints

We release our checkpoints [here](https://www.dropbox.com/sh/yapgdylhkm4j0hu/AACh4jvT7wtBgVNmiUprRSdCa?dl=0).


### 3) Stats files for FID evaluation

Download stats files and save it to `./assets/stats/`.

* For CIFAR-10, we use the stats file comes from [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch/blob/main/README.md#stats-files-for-quantitative-evaluation).
* For CelebA, we provide the stats file [`celeba_stats.npz`](https://www.dropbox.com/s/0y5c3q6qbehjxjx/celeba_stats.npz?dl=0).


### 4) Training and Evaluation each dataset

* When you run a deep model, change the value of `model.num_res_blocks` from 4 to 8 in the config file. (or add `--config.model.num_res_blocks 8` in your script.) 

### - CIFAR-10

#### INDM (VE, FID)

* Training
```
python main.py --mode train --config configs/ve/CIFAR10/indm.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/ve/CIFAR10/indm.py --workdir <work_dir> --config.sampling.pc_denoise=True --config.sampling.pc_denoise_time -1 --config.sampling.begin_snr 0.14 --config.sampling.end_snr 0.14 --config.eval.data_mean=True
```

#### INDM (VP, FID)

* Training
```
python main.py --mode train --config configs/vp/CIFAR10/indm_fid.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CIFAR10/indm_fid.py --workdir <work_dir> --config.sampling.temperature 1.05
```

#### INDM (VP, NLL)

* Training
```
python main.py --mode train --config configs/vp/CIFAR10/indm_nll.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CIFAR10/indm_nll.py --workdir <work_dir>
```

### - CelebA

#### INDM (VE, FID)

* Training
```
python main.py --mode train --config configs/ve/CELEBA/indm.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/ve/CELEBA/indm.py --workdir <work_dir>
```

#### INDM (VP, FID)

* Training
```
python main.py --mode train --config configs/vp/CELEBA/indm_fid.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CELEBA/indm_fid.py --workdir <work_dir>
```

#### INDM (VP, NLL)

* Training
```
python main.py --mode train --config configs/vp/CELEBA/indm_nll.py --workdir <work_dir>
```

* Evaluation (NLL/NELBO evaluation and sampling)
```
python main.py --mode eval --config configs/vp/CELEBA/indm_nll.py --workdir <work_dir>
```

## Reference
If you find the code useful for your research, please consider citing
```bib 
@inproceedings{kim2022maximum,
  title={Maximum Likelihood Training of Implicit Nonlinear Diffusion Model},
  author={Dongjun Kim and Byeonghu Na and Se Jung Kwon and Dongsoo Lee and Wanmo Kang and Il-Chul Moon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
 }
 ```
This work is heavily built upon the code from
* Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." *International Conference on Learning Representations (ICLR)*. 2021.
* Song, Yang, et al. "Maximum likelihood training of score-based diffusion models." *Advances in Neural Information Processing Systems (NeurIPS)*. 2021.
* Xuezhe Ma, et al. "Decoupling Global and Local Representations via Invertible Generative Flows." *International Conference on Learning Representations (ICLR)*. 2021.

## Acknowledgements

This research was supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data(IITP) funded by the Ministry of Science and ICT(2022-0-00077). Also, this work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (NRF-2019R1A5A1028324).
