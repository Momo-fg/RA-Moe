# dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class RegionDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        self.image_paths = []
        self.gt_paths = []
        self.transform = transform

        split_dir = os.path.join(root_path, split)
        
        if not os.path.exists(split_dir):
            return

        for emotion_class in os.listdir(split_dir):
            emotion_path = os.path.join(split_dir, emotion_class)
            if not os.path.isdir(emotion_path):
                continue

            files = os.listdir(emotion_path)
            for file in files:
                if not file.endswith('.jpg'):
                    continue
                
                if '_apex.jpg' in file or '_onset.jpg' in file or '_offset.jpg' in file:
                    img_path = os.path.join(emotion_path, file)
                    # print("img_path :",img_path)
                    base_name = file[:-4]
                    
                    npy_file = f"{base_name}_region.npy"
                    npy_path = os.path.join(emotion_path, npy_file)
                    
                    if os.path.exists(npy_path):
                        self.image_paths.append(img_path)
                        self.gt_paths.append(npy_path)

        print(f"Loaded {len(self.image_paths)} samples from {split}")
        
        # ✅ 统计 active samples
        if len(self.gt_paths) > 0:
            active_count = 0
            # print("gt_paths::",self.gt_paths)
            for gt_path in self.gt_paths:
                if np.load(gt_path).sum() > 0:
                    active_count += 1
            print(f"  Active samples: {active_count}/{len(self.gt_paths)} ({100*active_count/len(self.gt_paths):.1f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        gt = np.load(self.gt_paths[idx]).astype(np.float32)
        return img, gt