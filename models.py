import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear()
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        pass



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        ''' 
            Why use 1x1 1d Convs?
            -> equivalent to linear layers
            -> input dims are (batch, n points, hidden_dim) for linear
            -> input dims are (batch, hidden_dim, n points) for conv1d
            -> input dims are (batch, hidden_dim, n points) for batchnorm1d
                -> doing conv1ds saves computation

            output of all fcs outside of the last mlp are concatenated to the input of last mlp
            final output is of shape      
        '''
        ##reusables
        self.relu = nn.ReLU()

        self.fc1 = nn.Conv1d(3, 64, kernel_size=1)
        self.fc2 = nn.Conv1d(64, 128, kernel_size=1)
        self.fc3 = nn.Conv1d(128, 128, kernel_size=1)
        self.fc4 = nn.Conv1d(128, 512, kernel_size=1)
        self.fc5 = nn.Conv1d(512, 2048, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)

        #output of final_mlp is (batch, classes, n points)
        #permute the result -> whoever this prof is prob permutes it
        #back in training loop
        self.final_mlp = nn.Sequential(
            nn.Conv1d(64+128+128+512+2048, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, kernel_size=1)
        )
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        #dim of all these is (batch, hidden_dim, n points)
        points = points.permute(0,2,1)
        fc1_out = self.relu(self.bn1(self.fc1(points)))
        fc2_out = self.relu(self.bn2(self.fc2(fc1_out)))
        fc3_out = self.relu(self.bn3(self.fc3(fc2_out)))
        fc4_out = self.relu(self.bn4(self.fc4(fc3_out)))
        fc5_out = self.relu(self.bn5(self.fc5(fc4_out)))

        #global_feats is now of dimension (B, hidden_dim)
        global_feats, max_points = torch.max(fc5_out, dim=-1)

        #repeat the global to concat on all the locals -> shape (b, hidden_dim, n points)
        global_repeat = global_feats.repeat(fc5_out.shape[2], 1, 1).permute(1,2,0)

        local_global = torch.cat((fc1_out, fc2_out, fc3_out, fc4_out, global_repeat), dim=1)

        #output is (B, num_classes, n points)
        seg_feat_maps = self.final_mlp(local_global)

        return seg_feat_maps.permute(0,2,1)





