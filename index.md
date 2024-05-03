---
title: Yuhao Zhou | Home
nav-title: Home
---
I am a Ph.D. candidate (from fall, 2020) in the Department of Computer Science and Technology at [Tsinghua University](https://www.tsinghua.edu.cn/en/), advised by [Prof. Jun Zhu](http://ml.cs.tsinghua.edu.cn/~jun). 

I received my bachelor degree at Tsinghua University, majored in _Computer Science and Technology_. 
Meanwhile, I also received my second bachelor degree, majored in _Mathematics_.

<!-- You can find me at [Google Scholar](https://scholar.google.com/citations?user=GKLRbxoAAAAJ&hl=en) and [GitHub](http://github.com/miskcoo/)! -->

E-mails: 
yuhaoz.cs [at] gmail.com _(preferred)_; 
zhouyh20 [at] mails.tsinghua.edu.cn.

[[Google Scholar]](https://scholar.google.com/citations?user=GKLRbxoAAAAJ&hl=en)
[[GitHub]](https://github.com/miskcoo/)
[[Blog (in Chinese)]](https://blog.miskcoo.com)

# Publications

_(*) denotes equal contribution._

<!--## Preprints-->

## Journal Papers

* **A Semismooth Newton based Augmented Lagrangian Method for Nonsmooth Optimization on Matrix Manifolds**{:.paper-title}  
  **Yuhao Zhou**{:.author-me}, **Chenglong Bao**{:.author}, **Chao Ding**{:.author}, **Jun Zhu**{:.author}.  
  _**Mathematical Programming**_, 201, 1-61 (2023).  
  [[pdf]](https://link.springer.com/article/10.1007/s10107-022-01898-1)
  [[code]](https://github.com/miskcoo/almssn)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: This paper is devoted to studying an augmented Lagrangian method for solving a class of manifold optimization problems, which have nonsmooth objective functions and nonlinear constraints. Under the constant positive linear dependence condition on manifolds, we show that the proposed method converges to a stationary point of the nonsmooth manifold optimization problem. Moreover, we propose a globalized semismooth Newton method to solve the augmented Lagrangian subproblem on manifolds efficiently. The local superlinear convergence of the manifold semismooth Newton method is also established under some suitable conditions. We also prove that the semismoothness on submanifolds can be inherited from that in the ambient manifold. Finally, numerical experiments on compressed modes and (constrained) sparse principal component analysis illustrate the advantages of the proposed method.

## Conference Papers

* **Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations**{:.paper-title}  
  **Kaiwen Xue**{:.author}\*, **Yuhao Zhou**{:.author-me}\*, **Shen Nie**{:.author}, **Xu Min**{:.author}, **Xiaolu Zhang**{:.author}, **Jun Zhou**{:.author}, **Chongxuan Li**{:.author}.  
  _International Conference on Machine Learning (**ICML**)_, 2024.  
  [[pdf]](https://arxiv.org/abs/2404.15766)
  [[code]](https://github.com/ML-GSAI/BFN-Solver)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: Bayesian flow networks (BFNs) iteratively refine the parameters, instead of the samples in diffusion models (DMs), of distributions at various noise levels through Bayesian inference. Owing to its differentiable nature, BFNs are promising in modeling both continuous and discrete data, while simultaneously maintaining fast sampling capabilities. This paper aims to understand and enhance BFNs by connecting them with DMs through stochastic differential equations (SDEs). We identify the linear SDEs corresponding to the noise-addition processes in BFNs, demonstrate that BFN's regression losses are aligned with denoise score matching, and validate the sampler in BFN as a first-order solver for the respective reverse-time SDE. Based on these findings and existing recipes of fast sampling in DMs, we propose specialized solvers for BFNs that markedly surpass the original BFN sampler in terms of sample quality with a limited number of function evaluations (e.g., 10) on both image and text datasets. Notably, our best sampler achieves an increase in speed of 5~20 times for free.

* **The Blessing of Randomness: SDE Beats ODE in General Diffusion-based Image Editing**{:.paper-title}  
  **Shen Nie**{:.author}, **Hanzhong Allan Guo**{:.author}, **Cheng Lu**{:.author}, **Yuhao Zhou**{:.author-me}, **Chenyu Zheng**{:.author}, **Chongxuan Li**{:.author}.  
  _International Conference on Learning Representations (**ICLR**)_, 2024.  
  [[pdf]](https://arxiv.org/abs/2311.01410)
  [[code]](https://github.com/ML-GSAI/SDE-Drag)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: We present a unified probabilistic formulation for diffusion-based image editing, where a latent variable is edited in a task-specific manner and generally deviates from the corresponding marginal distribution induced by the original stochastic or ordinary differential equation (SDE or ODE). Instead, it defines a corresponding SDE or ODE for editing. In the formulation, we prove that the Kullback-Leibler divergence between the marginal distributions of the two SDEs gradually decreases while that for the ODEs remains as the time approaches zero, which shows the promise of SDE in image editing. Inspired by it, we provide the SDE counterparts for widely used ODE baselines in various tasks including inpainting and image-to-image translation, where SDE shows a consistent and substantial improvement. Moreover, we propose SDE-Drag -- a simple yet effective method built upon the SDE formulation for point-based content dragging. We build a challenging benchmark (termed DragBench) with open-set natural, art, and AI-generated images for evaluation. A user study on DragBench indicates that SDE-Drag significantly outperforms our ODE baseline, existing diffusion-based methods, and the renowned DragGAN. Our results demonstrate the superiority and versatility of SDE in image editing and push the boundary of diffusion-based editing methods.

* **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps**{:.paper-title}  
  **Cheng Lu**{:.author}, **Yuhao Zhou**{:.author-me}, **Fan Bao**{:.author}, **Jianfei Chen**{:.author}, **Chongxuan Li**{:.author}, **Jun Zhu**{:.author}.  
  _Neural Information Processing Systems (**NeurIPS**)_, 2022.  
  **Oral Presentation**{:.p-highlight}.  
  [[pdf]](https://arxiv.org/abs/2206.00927)
  [[improved version]](https://arxiv.org/abs/2211.01095)
  [[code]](https://github.com/LuChengTHU/dpm-solver)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: Diffusion probabilistic models (DPMs) are emerging powerful generative models. Despite their high-quality generation performance, DPMs still suffer from their slow sampling as they generally need hundreds or thousands of sequential function evaluations (steps) of large neural networks to draw a sample. Sampling from DPMs can be viewed alternatively as solving the corresponding diffusion ordinary differential equations (ODEs). In this work, we propose an exact formulation of the solution of diffusion ODEs. The formulation analytically computes the linear part of the solution, rather than leaving all terms to black-box ODE solvers as adopted in previous works. By applying change-of-variable, the solution can be equivalently simplified to an exponentially weighted integral of the neural network. Based on our formulation, we propose DPM-Solver, a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee. DPM-Solver is suitable for both discrete-time and continuous-time DPMs without any further training. Experimental results show that DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets. We achieve 4.70 FID in 10 function evaluations and 2.87 FID in 20 function evaluations on the CIFAR10 dataset, and a 4∼16× speedup compared with previous state-of-the-art training-free samplers on various datasets.

* **Fast Instrument Learning with Faster Rates**{:.paper-title}  
  **Ziyu Wang**{:.author}, **Yuhao Zhou**{:.author-me}, **Jun Zhu**{:.author}.  
  _Neural Information Processing Systems (**NeurIPS**)_, 2022.  
  [[pdf]](https://arxiv.org/abs/2205.10772)
  [[code]](https://github.com/meta-inf/fil)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: We investigate nonlinear instrumental variable (IV) regression given high-dimensional instruments. We propose a simple algorithm which combines kernelized IV methods and an arbitrary, adaptive regression algorithm, accessed as a black box. Our algorithm enjoys faster-rate convergence and adapts to the dimensionality of informative latent features, while avoiding an expensive minimax optimization procedure, which has been necessary to establish similar guarantees. It further brings the benefit of flexible machine learning models to quasi-Bayesian uncertainty quantification, likelihood-based model selection, and model averaging. Simulation studies demonstrate the competitive performance of our method.

* **Gradient Estimation with Discrete Stein Operators**{:.paper-title}  
  **Jiaxin Shi**{:.author}, **Yuhao Zhou**{:.author-me}, **Jessica Hwang**{:.author}, **Michalis K. Titsias**{:.author}, **Lester Mackey**{:.author}.  
  _Neural Information Processing Systems (**NeurIPS**)_, 2022.  
  **[Outstanding Paper Award](https://blog.neurips.cc/2022/11/21/announcing-the-neurips-2022-awards/)**{:.p-highlight}.  
  [[pdf]](https://arxiv.org/abs/2202.09497) 
  [[code]](https://github.com/thjashin/rodeo)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: Gradient estimation -- approximating the gradient of an expectation with respect to the parameters of a distribution—is central to the solution of many machine learning problems. However, when the distribution is discrete, most common gradient estimators suffer from excessive variance. To improve the quality of gradient estimation, we introduce a variance reduction technique based on Stein operators for discrete distributions. We then use this technique to build flexible control variates for the REINFORCE leave-one-out estimator. Our control variates can be adapted online to minimize the variance and do not require extra evaluations of the target function. In benchmark generative modeling tasks such as training binary variational autoencoders, our gradient estimator achieves substantially lower variance than state-of-the-art estimators with the same number of function evaluations.

* **Scalable Quasi-Bayesian Inference for Instrumental Variable Regression**{:.paper-title}  
  **Ziyu Wang**{:.author}\*, **Yuhao Zhou**{:.author-me}\*, **Tongzheng Ren**{:.author}, **Jun Zhu**{:.author}.  
  Short version in _Neural Information Processing Systems (**NeurIPS**)_, 2021.  
  [[short version]](https://proceedings.neurips.cc/paper/2021/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html)
  [[full version]](https://arxiv.org/abs/2106.08750) 
  [[code]](https://github.com/meta-inf/qbdiv)
  [[slides]](https://ml.cs.tsinghua.edu.cn/~ziyu/static/p/qbdiv/slides-1.pdf)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: Recent years have witnessed an upsurge of interest in employing flexible machine learning models for instrumental variable (IV) regression, but the development of uncertainty quantification methodology is still lacking. In this work we present a scalable quasi-Bayesian procedure for IV regression, building upon the recently developed kernelized IV models. Contrary to Bayesian modeling for IV, our approach does not require additional assumptions on the data generating process, and leads to a scalable approximate inference algorithm with time cost comparable to the corresponding point estimation methods. Our algorithm can be further extended to work with neural network models. We analyze the theoretical properties of the proposed quasi-posterior, and demonstrate through empirical evaluation the competitive performance of our method. 

* **Nonparametric Score Estimators**{:.paper-title}  
  **Yuhao Zhou**{:.author-me}, **Jiaxin Shi**{:.author}, **Jun Zhu**{:.author}.  
  _International Conference on Machine Learning (**ICML**)_, 2020.  
  [[pdf]](http://proceedings.mlr.press/v119/zhou20c.html) 
  [[code]](https://github.com/miskcoo/kscore) 
  [[slides]](https://ml.cs.tsinghua.edu.cn/~yuhao/slides/nonparametric score estimators, icml2020.pdf)
  [[abstract]](javascript:void(0);)

  {:.paper-abstract .paper-toggle}
  **Abstract**: Estimating the score, i.e., the gradient of log density function, from a set of samples generated by an unknown distribution is a fundamental task in inference and learning of probabilistic models that involve flexible yet intractable densities. Kernel estimators based on Stein's methods or score matching have shown promise, however their theoretical properties and relationships have not been fully-understood. We provide a unifying view of these estimators under the framework of regularized nonparametric regression. It allows us to analyse existing estimators and construct new ones with desirable properties by choosing different hypothesis spaces and regularizers. A unified convergence analysis is provided for such estimators. Finally, we propose score estimators based on iterative regularization that enjoy computational benefits from curl-free kernels and fast convergence.
