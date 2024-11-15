# SAAL: Sharpness-Aware Active Learning (ICML 2023) <br><sub>Official PyTorch implementation of SAAL </sub>
 Yoon-Yeong Kim\*, Youngjae Cho\*, Joonho Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, Il-Chul Moon    
<sup> * Equal contribution </sup> <br>




## Paper description and Main Idea

To overcome overfitting, this paper introduces the first active learning method to incorporate the sharpness of loss space into the acquisition function. Specifically, our proposed method, Sharpness-Aware Active Learning (SAAL), constructs its acquisition function by selecting unlabeled instances whose perturbed loss becomes maximum. Unlike the Sharpness-Aware learning with fully-labeled datasets, we design a pseudo-labeling mechanism to anticipate the perturbed loss w.r.t. the ground-truth label, which we provide the theoretical bound for the optimization. 


## Contribution

We propose a new active learning method named Sharpness-Aware Active Learning, or SAAL. The proposed method considers theloss sharpness of data instances, which is strongly related to the generalization performance of deep learning.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training

To train the model(s) in the paper, run this command:

```train
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name Adam --isInit True
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name SAM --isInit False
```

## Evaluation

- Data will be downloaded to folder 'data'.
- Result will be recorded to folder 'Results'.

## Results

Our model achieves the following performance on active learning settings:

| Optimizer  | FashionMNIST  |      SVHN     |    CIFAR-10   |   CIFAR-100   |
| ----------- |-------------- | ------------- | ------------- | ------------- |
|    Adam     |     85.8%    |     76.8%    |     54.4%    |     47.6%    |
|    SAM     |     86.3%    |     78.8%    |     57.0%    |     48.4%    |


## Reference
```
@article{kim2023saal,
  title={SAAL: Sharpness-Aware Active Learning},
  author={Kim, Yoon-Yeong and Cho, Youngjae and Jang, JoonHo and Na, Byeonghu and Kim, Yeongmin and Song, Kyungwoo and Kang, Wanmo and Moon, Il-chul},
  year={2023}
}
```

## Acknowledgements
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. 2021R1A2C200981612). Also, this work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (NO. 2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).
