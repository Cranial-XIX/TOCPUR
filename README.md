# Team Orienteering Coverage Planning with Uncertain Reward
This is the implementation for Team Orienteering Coverage Planning with Uncertain Reward (TOCPUR) that includes the 600 randomly generated graphs over 5 different horizons of planning (H=[2,4,6,8,10]). The paper has been accepted to ICRA 2021.
The codebase includes the implementation of a greedy method, the exact MIP TOP method, and the proposed MIP TOCP method.

# Required Dependency
```
pip install mip
```

# Reproduce the Results
```
chmod +x run.sh && bash run.sh
```
# Citations
Please consider citing [this paper](https://arxiv.org/pdf/2105.03721.pdf)
 if you use the code or find the work interesting
```
@article{liu2021team,
  title={Team Orienteering Coverage Planning with Uncertain Reward},
  author={Liu, Bo and Xiao, Xuesu and Stone, Peter},
  journal={arXiv preprint arXiv:2105.03721},
  year={2021}
}
```
