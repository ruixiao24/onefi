1. modify params in config.m and obtain_meta_data.m
2. run compute_velocity_distribution.m to obtain velocity distribution
3. run create_virtual_gesture.m to obtain virtual gesture

Note that our sample code set `meta_data('segment_length')=100` and thus the generated virtual gesture is of shape `[4, 121, 20]`. `meta_data('segment_length')=40` will generate virtual gestures of shape `[4, 121, 50]`, which is identical of that used in few-shot-recognition directory.
