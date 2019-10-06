# Complex-Valued Deep Neural Network with Weighted Fréchet Mean

## Abstract

Complex-valued deep learning has attracted increasing attention in recent years, due to its versatility and ability to capture more information. However, the lack of well-defined complex-valued operations remains a bottleneck for further advancement. In this work, we propose a geometric way to define deep neural networks on the space of complex numbers by utilizing weighted Fr\'{e}chet mean. We mathematically prove the viability of our algorithm. We also define basic building blocks such as convolution, non-linearity, and residual connections tailored for the space of complex numbers. To demonstrate the effectiveness of our proposed model, we compare our complex-valued network comprehensively with its real state-of-the-art counterpart on the MSTAR classification task and achieve better performance, while utilizing less than 1% of the parameters. 


<img src='./assets/summary.png' width=800>

Further information please contact [Rudrasis Chakraborty](https://github.com/rudra1988) and [Yifei Xing](mailto:xingyifei2016@berkeley.edu).

<img src='./assets/results_merge.pdf' width=800>

## Requirements
* [PyTorch](https://pytorch.org/)

## Data Preparation

- First, run `cat data_split* > data_polar.zip` inside the `data` folder.

- Next, extract `data_polar.zip` and set the correct path to the data_polar folder inside the argparse configuration in `train_demo.py`


## Getting Started (Training & Testing)


- To train the model: 
```
python train_demo.py
```

## CAUTION
The current code was prepared using single GPU. The use of multi-GPU may cause problems. 

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.
