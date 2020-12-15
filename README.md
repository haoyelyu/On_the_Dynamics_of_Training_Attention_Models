# On the Dynamics of Training Attention Models

Code for [On the Dynamics of Training Attention Models](https://arxiv.org/abs/2011.10036?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29), submitted to ICLR 2021 [[OpenReview](https://openreview.net/forum?id=1OCTOShAmqB&noteId=ITjHwKuKK_U)]. This repository contains the implementations for the experiments in Section 5.

## Instructions for running the code
Dependency: PyTorch 1.6.0 with CUDA 10.1, matplotlib 3.3.2, sklearn 0.23.2 and numpy 1.19.2

Run the following cmd for reproducing the results by training Attn\_FC, Attn\_TC and Attn_TL on the artificial dataset introduced in Section 3:

~~~ shell
python perform_synthetic_analysis.py
~~~

Run the following cmd for reproducing the results for ablation study:

~~~ shell
python ablation_analysis.py
~~~ 
	
Run the following cmd for reproducing the results by training Attn\_FC and Attn\_TC on SST2:

~~~ shell
python perform_sst2_analysis.py FC
python perform_sst2_analysis.py TC
~~~ 

Run the following cmd for reproducing the results by training Attn\_FC and Attn\_TC on SST5:

~~~ shell
python perform_sst5_analysis.py FC
python perform_sst5_analysis.py TC
~~~ 

All plots will be stored in

~~~
results/
~~~