# Multimodal-Representation

#### This repository organizes researches related to AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data, especially Multimodal-Representation task. 
#### This repository summarizes following researches.

## Research list

* LAGMA: LAtent Goal-guided Multi-agent Reinforcement Learning (ICML 2024) - Hyungho Na and Il-Chul Moon.

  * The proposed LAtent Goal-guided Multi-Agent reinforcement learning (LAGMA) method generates goal-reaching trajectories in latent space and motivates agents with a latent goal-guided incentive system. LAGMA employs a modified VQ-VAE to create a quantized latent space and an extended VQ codebook for trajectory generation, significantly improving task performance in complex environments.

* Diffusion Rejection Sampling (ICML 2024)- Byeonghu Na, Yeongmin Kim, Minsang Park, Donghyeok Shin, Wanmo Kang, and Il-Chul Moon.

  * The proposed Diffusion Rejection Sampling (DiffRS) employs a rejection sampling scheme that aligns transition kernels with true kernels at each timestep, improving the accuracy of diffusion models. Theoretical analysis confirms that DiffRS achieves a tighter bound on sampling error compared to existing pre-trained models.

* Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning (ICLR 2024 Oral) - Hyungho Na, Yunkyeong Seo, and Il-Chul Moon.

  * The proposed Efficient episodic Memory Utilization (EMU) method for multi-agent reinforcement learning accelerates learning by leveraging semantically coherent memory from an episodic buffer and introduces a novel reward structure called episodic incentive to promote desirable transitions and prevent local optima. This approach utilizes a trainable encoder/decoder to create memory embeddings that facilitate exploratory recall and improve the Q-learning TD target.

* Training Unbiased Diffusion Models From Biased Dataset (ICLR 2024) - Yeongmin Kim, Byeonghu Na, JoonHo Jang, Minsang Park, Dongjun Kim, Wanmo Kang, Il-Chul Moon.

  * The proposed method uses time-dependent importance reweighting to address dataset bias in diffusion models, enhancing the precision of the density ratio for improved error minimization in generative learning. This technique makes the objective function tractable by combining reweighting and score correction, leading to the regeneration of unbiased data density.

* Unknown Domain Inconsistency Minimization for Domain Generalization (ICLR 2024) - Seungjae Shin, Heesun Bae, Byeonghu Na, Yoon-Yeong Kim, and Il-Chul Moon.

  * The proposed Unknown Domain Inconsistency Minimization (UDIM) strategy enhances domain generalization by reducing loss landscape inconsistencies between known and crafted unknown domains, using parameter and data perturbations. This approach aligns the loss landscapes from the source to perturbed domains, aiming to improve model transferability to unobserved domains.

* Make Prompts Adaptable : Bayesian Modeling for Vision-Language Prompt Learning with Data-Dependent Prior (AAAI 2024) - Youngjae Cho, HeeSun Bae, Seungjae Shin, Yeo Dong Youn, Weonyoung Joo, Il-Chul Moon.

  * The proposed Bayesian framework utilizes Wasserstein Gradient Flow for estimating the target posterior distribution, enabling flexible prompt adaptation to capture the complex modes of image features in VLP models, enhancing both adaptability and generalization in few-shot learning scenarios.

* SAAL: Sharpness-Aware Active Learning (ICML 2023) - Yoon-Yeong Kim, Youngjae Cho, Joonho Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

  * The proposed Sharpness-Aware Active Learning (SAAL), is the first active learning method to incorporate the sharpness of loss space into the acquisition function. SAAL constructs its acquisition function by selecting unlabeled instances whose perturbed loss becomes maximum.

* Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (ICML 2023 Oral) - Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon.

  * The proposed Discriminator Guidance (DG), aims to improve sample generation of pre-trained diffusion models. The approach introduces a discriminator that gives explicit supervision to a denoising sample path whether it is realistic or not.

* Loss Curvature Matching for Dataset Selection and Condensation (AISTATS 2023) - Seungjae Shin, HeeSun Bae, Donghyeok Shin, Weonyoung Joo, and Il-Chul Moon.

  * The proposed Loss Curvature Matching (LCMat), is a new reduction objective matching the loss curvatures of the original dataset and reduced dataset over the model parameter space, more than the parameter point. This new objective induces a better adaptation of the reduced dataset on the perturbed parameter region than the exact point matching.

* Maximum Likelihood Training of Implicit Nonlinear Diffusion Model (NeurIPS 2022) - Dongjun Kim, Byeonghu Na, Se Jung Kwon, Dongsoo Lee, Wanmo Kang, and Il-Chul Moon.

  * The proposed Implicit Nonlinear Diffusion Model (INDM) learns by combining a normalizing flow and a diffusion process. INDM implicitly constructs a nonlinear diffusion on the data space by leveraging a linear diffusion on the latent space through a flow network with connecting by invertible function.

* From Noisy Prediction to True Label: Noisy Prediction Calibration via Generative Model (ICML 2022) - HeeSun Bae, Seungjae Shin, Byeonghu Na, JoonHo Jang, Kyungwoo Song, and Il-Chul Moon.

  * The proposed Noise Prediction Calibration (NPC), calibrates the prediction from a imperfectly pre-trained classifier to a true label via utilizing a deep generative model. NPC operates as a post-processing module to a black-box classifier, without further access into classifier parameters.

* Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation (ICML 2022) - Dongjun Kim,Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

  * The proposed Soft Truncation (ST), is a universally applicable training technique for diffusion models, softens the fixed and static truncation hyperparameter into a random variable. ST softens the truncation level at each mini-batch update, and this simple modification is connected to the general weighted diffusion loss and the concept of Maximum Perturbed Likelihood Estimation.
 
  * 
