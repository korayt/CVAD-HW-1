from torch.utils.data import Dataset
import torchvision
import json
from torchvision import transforms
from PIL import Image
import numpy as np



class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, length):
        self.length = length
        self.data_root = data_root
        self.images_list = []
        self.masurements_list = []
        self.transformer = transforms.Compose([transforms.Resize((224, 224))])
        f = None
        for i in range(length):
            self.images_list.append(torchvision.io.read_image(data_root + '/rgb/' + str(i).zfill(8) + '.png')[[2, 1, 0], :, :])
            f = open(data_root + '/measurements/' + str(i).zfill(8) + '.json')
            data = json.load(f)
            self.masurements_list.append((data['command'], data['speed'], data['throttle'], data['brake'], data['steer'], data['lane_dist'], data['route_angle'], data['tl_dist'], data['tl_state']))
        f.close()

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        return self.transformer(self.images_list[index]), self.masurements_list[index]

    def __len__(self):
        return self.length
"""
def main():
    train_root = '/home/koray/expert_data/train'
    im_rgb = torchvision.io.read_image(train_root + '/rgb/' + str(0).zfill(8) + '.png')
    print(im_rgb.dtype)
    print(im_rgb.shape)
    im_rgb = im_rgb[[2, 1, 0], :, :]
    im_rgb = np.array(im_rgb.permute(1, 2, 0), dtype=np.uint8)

    Image.fromarray(im_rgb).show()

if __name__ == "__main__":
    main()
"""