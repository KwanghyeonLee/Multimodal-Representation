# UDIM (Unknown Domain Inconsistency Minimization for Domain Generalization, ICLR 2024)

Official PyTorch implementation of
**["Unknown Domain Inconsistency Minimization for Domain Generalization"](https://openreview.net/forum?id=eNoiRal5xi)** (ICLR 2024) 
by 
Seungjae Shin, 
HeeSun Bae, 
Byeonghu Na, 
Yoon-Yeong Kim, 
and Il-chul Moon.

## Overview

UDIM reduces the loss landscape inconsistency between source domain and unknown domains. 
As unknown domains are inaccessible, these domains are empirically crafted by perturbing instances from the source domain dataset. 
In particular, by aligning the loss landscape acquired in the source domain to the loss landscape of perturbed domains, 
we expect to achieve generalization grounded on these flat minima for the unknown domains.

Our implementation code is largely dependent on the code of **[SAGM (CVPR2023)](https://github.com/Wang-pengfei/SAGM)**. We thank the authors for providing these codes.

<p align="center">
  <img src="https://github.com/aailabkaist/UDIM/assets/20755743/b445eeba-02a9-40d4-9f9d-fc7241b3058f" alt="Image Description">
</p>

## Reproduce
To reproduce the results of UDIM, we provide a bash file, which located at: 
```
/bash/udim_pacs_xxx.sh
```
Here, XXX is a specific experimental setting. 

## License

This source code is released under the MIT license, included [here](./LICENSE). This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.

For additional questions or discussions, feel free to email me. (tmdwo0910@kaist.ac.kr)
## Acknowledgements
This research was supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data(IITP) funded by the Ministry of Science and ICT(2022-0-00077).
