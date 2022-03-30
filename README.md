# OneFi: One-Shot Recognition for Unseen Gesture with COTS WiFi


This repo contains the code of the following paper:

> OneFi: One-shot Recognition for Unseen Gesture with COTS WiFi

## How to use the code

1. Please follow the instructions under virtual-gesture-generation folder to perform data augmentation. (corresponding to Sec.4 in paper).
2. Please follow the instructions under few-shot-learning folder to perform one-shot learning (corresponding to Sec.5 in paper). Please replace `train_data.npy` and `test_data.npy` with your own data. 

## Appendix

Training Set:

![image-20210918160208690](./figures/train.png)

Testing Set:

![image-20210918160153004](./figures/test.png)

H: horizontally; V: Vertically

CW: clockwise; CCW: counterclockwise