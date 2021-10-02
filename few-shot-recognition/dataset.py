import os, torch
import numpy as np
from torch.utils.data import Dataset
import config

class AugmentSet(Dataset):
    # DATA SHAPE: [n_action, n_trial, n_orientation, n_receiver, n_feature, n_timestamp]
    path = os.path.join(config.data_dir, "train_data.npy")

    def __init__(self):
        # load data from npy file
        self.data = torch.from_numpy(np.load(self.path)).float()
        
        # obtain dataset params
        self.n_action = 20
        self.n_orientation = 12
        self.n_trial = 30

        self.n_class = self.n_action * self.n_orientation
        self.len = self.n_class * self.n_trial

    def __getitem__(self, index):
        class_id = index // self.n_trial
        image_id = index %  self.n_trial

        action_id = class_id // self.n_orientation
        orient_id = class_id %  self.n_orientation
        item = self.data[action_id, image_id, orient_id]
        label = class_id

        # [n_receiver, n_feature, n_timestamp] --> [n_receiver*n_feature, n_timestamp]
        item = np.reshape(item, (4 * 121, 51))

        return item, label

    def __len__(self):
        return self.len


class TestSet():
    # DATA SHAPE: [n_class, n_trial, n_receiver, n_feature, n_timestamp]
    path = os.path.join(config.data_dir, "test_data.npy")
    
    def __init__(self, n_way, k_shot, k_query):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        
        data = np.load(self.path)
        self.data = data[:]
        self.n_img = self.data.shape[1]
        print(self.data.shape)
        self.resize = (4, 121, 51)
    def load_test_set(self):
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        qrysz = self.k_query * self.n_way

        # selected_cls = np.random.choice(self.data.shape[0], self.n_way, False)
        selected_cls = np.arange(self.n_way)

        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i, cur_class in enumerate(selected_cls):

            # select k_shot+k_query images from all images in the given class randomly
            selected_img = np.random.choice(self.n_img, self.k_shot + self.k_query, False)

            x_spt.append(self.data[cur_class][selected_img[:self.k_shot]])
            x_qry.append(self.data[cur_class][selected_img[self.k_shot:]])
            y_spt.append([i for _ in range(self.k_shot)])
            y_qry.append([i for _ in range(self.k_query)])
        
        # shuffle inside a batch
        perm = np.random.permutation(self.n_way * self.k_shot)
        
        x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize[0], self.resize[1], self.resize[2])[perm]
        y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
        perm = np.random.permutation(self.n_way * self.k_query)
        x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize[0], self.resize[1], self.resize[2])[perm]
        y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

        # Transformer needs a different input format: [n_timestamps, setsz, n_features] (i.e. [w, setsz, c*h])
        # [setsz, c, h, w] ==> [setsz, c*h, w], where w is n_timestamps
        x_spt = x_spt.astype(np.float32).reshape(setsz, self.resize[0] * self.resize[1], self.resize[2])
        x_qry = x_qry.astype(np.float32).reshape(qrysz, self.resize[0] * self.resize[1], self.resize[2])
        
        # [setsz, c*h, n_time] ==> [n_time, setsz, c*h]
        x_spt = np.transpose(x_spt, (2, 0, 1))
        x_qry = np.transpose(x_qry, (2, 0, 1))

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), \
                                     torch.from_numpy(x_qry), torch.from_numpy(y_qry)

        return x_spt, y_spt, x_qry, y_qry
