This repo provides the code used in the paper
# [End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization (CVPR 2020)](https://arxiv.org/pdf/1909.06043.pdf)

![](demo_data/cover.png)

## Install

`bash requirements.sh`


## Back-propagatable PnP (BPnP)

Using BPnP is easy. Just add the following line in your code
````bash
import BPnP
bpnp = BPnP.BPnP.apply
````
Then you can use it as any autograd function in Pytorch.

## Demo experiments

To see the demos presented in the paper, run
````bash
python demoPoseEst.py
````
or
````bash
python demoSfM.py
````
or
````bash
python demoCamCali.py
````
