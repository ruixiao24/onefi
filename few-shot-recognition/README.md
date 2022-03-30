## How to run the code

1. Run train.py
2. Run test.py

## Data Format

Format of train set:  `[n_action, n_trial, n_orientation, n_receiver, n_feature, n_timestamp]`

Format of test set:  `[n_class, n_trial, n_receiver, n_feature, n_timestamp]`

| Name          | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| n_action      | no. of gestures                                              |
| n_trial       | no. of trials for each gesture                               |
| n_orientation | no. of orientations for each gesture. Each orientation means an data augmentation by rotating the original gesture. (Please refer to Sec.4 in paper.) |
| n_receiver    | no. of WiFi receivers                                        |
| n_feature     | no. of features for each timestamp                           |
| n_timestamp   | no. of timestamps                                            |

