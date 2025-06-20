# from utils.datasets.CUB200 import Cub2011
from utils.datasets.imagenet_1k import ImageNet1K
from utils.datasets.inat_dataset import iNatDataset
from torchvision.datasets import Flowers102
from torch.utils.data import Dataset
from PIL import Image
import PIL.Image
import pathlib
import os
import torch
from torchvision.datasets import folder as dataset_parser
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

NUM_CLASSES_DICT = {
    'imagenet_1k': 1000,
    'stanford_cars': 196,
    'fgvc-aircraft': 100,
    'eurosat': 10,
    'dtd': 47,
    'semi-aves': 200,
    'flowers102': 102,
    'semi-inat-2021': 810,
}


def load_dataset(dataset_root, split, preprocess, tokenized_text_prompts, pl_list=None):
    
    dataset = MyDataset(dataset_root=dataset_root,
                        split=split, transform=preprocess, 
                        tokenized_text_prompts=tokenized_text_prompts)

    return dataset


class SemiAvesDataset(Dataset):
    def __init__(self, dataset_root, split, transform, tokenized_text_prompts,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        with open(os.path.join(self.dataset_root, split), 'r') as f:
            lines = f.readlines()

        self.data = []
        self.labels = []
        for line in lines:
            path, id, is_fewshot = line.strip('\n').split(' ')
            file_path = os.path.join(self.dataset_root, path)
            self.data.append((file_path, int(id), int(is_fewshot)))
            self.labels.append(int(id))
        
        self.targets = self.labels  # Sampler needs to use targets

        self.transform = transform
        print(f'# of images in {split}: {len(self.data)}')
        self.tokenized_text_prompts = tokenized_text_prompts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        source = self.data[i][2] # 0 for retrived data, 1 for fewshot data
        img = self.transform(img)
        tokenized_text = self.tokenized_text_prompts[str(label)]['all'] # use all prompts

        # random sample one prompt from the list of prompts
        random_idx = torch.randint(0, tokenized_text.size(0), (1,))

        tokenized_text = tokenized_text[random_idx]

        return img, label, tokenized_text, source


class MyDataset(Dataset):
    def __init__(self, dataset_root, split, transform, tokenized_text_prompts,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        file_list = split[0]
        path_list = split[1]

        lines = []
        for file, path in zip(file_list, path_list):
            with open(os.path.join(self.dataset_root, file), 'r') as f:
                line = f.readlines()
                # prepend the path to the each line !!!
                line = [os.path.join(path, l) for l in line]
            lines.extend(line)

        self.data = []
        self.labels = []
        for line in lines:
            path, id, is_fewshot = line.strip('\n').split(' ')
            file_path = path
            self.data.append((file_path, int(id), int(is_fewshot)))
            self.labels.append(int(id))
        
        self.targets = self.labels  # Sampler needs to use targets

        self.transform = transform

        self.tokenized_text_prompts = tokenized_text_prompts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        source = self.data[i][2] # 0 for retrived data, 1 for fewshot data
        img = self.transform(img)

        tokenized_text = self.tokenized_text_prompts[str(label)]['all'] # use all prompts

        # random sample one prompt from the list of prompts
        random_idx = torch.randint(0, tokenized_text.size(0), (1,))

        tokenized_text = tokenized_text[random_idx]

        return img, label, tokenized_text, source
    


class MyUnlabeledDataset(Dataset):
    def __init__(self, dataset_root, split, transform,
                 loader=dataset_parser.default_loader):
        
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader

        file_list = split[0]
        path_list = split[1]

        lines = []
        for file, path in zip(file_list, path_list):
            with open(os.path.join(self.dataset_root, file), 'r') as f:
                line = f.readlines()
                # prepend the path to the each line !!!
                line = [os.path.join(path, l) for l in line]
            lines.extend(line)

        self.data = []
        self.labels = []
        for line in lines:
            path, id, is_fewshot = line.strip('\n').split(' ')
            file_path = path
            self.data.append((file_path, int(id), int(is_fewshot)))
            self.labels.append(int(id))
        
        self.targets = self.labels  # Sampler needs to use targets

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        
        img = self.loader(self.data[i][0])
        label = self.data[i][1]
        source = self.data[i][2] # 0 for retrived data, 1 for fewshot data
        img = self.transform(img) # this will return weak aug and strong aug
        tokenized_text = torch.zeros(1, 1).long() # dummy tokenized text

        return img, label, tokenized_text, source
    


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, pre_extracted_path=None, device='cuda:0'):
        
        if pre_extracted_path is None:
            raise NotImplementedError
        else:
            pre_extracted_path = pre_extracted_path

        self.dataset = torch.load(pre_extracted_path, map_location=device)
        self.input_tensor = self.dataset['image_features']
        self.label_tensor = self.dataset['labels']
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.label_tensor[index], -1 # note here I use label as text
    
    def __len__(self):
        return self.input_tensor.size(0)


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_tensors, device):

        labels_list = []
        prompt_tensors_list = []
        for class_id, info in prompt_tensors.items():
            prompt_num = int(info['all'].shape[0])
            labels = torch.Tensor([int(class_id) for i in range(prompt_num)]).long()
            labels_list.append(labels)
            
            prompt_embedding = prompt_tensors[class_id]['all']
            prompt_tensors_list.append(prompt_embedding)

        self.labels = torch.cat(labels_list).flatten().to(device)
        self.prompt_tensors = torch.cat(prompt_tensors_list, dim=0).to(device)
        # print('Loaded Text Tensor Embeddings - ', self.prompt_tensors.shape, self.labels.shape)
    
    def __getitem__(self, index):
        return self.prompt_tensors[index], self.labels[index]
    
    def __len__(self):
        return self.prompt_tensors.size(0)


class MinedDataset(Dataset):
    def __init__(self, transform=None, dataset_root=".", caption_map=None):
        self.dataset_root = pathlib.Path(dataset_root)
        self.fnames = list(self.dataset_root.glob("**/*.jpg"))
        
        if len(self.fnames) == 0:
            raise ValueError('Check image suffix. No images found in the specified dataset_root: {}'.format(dataset_root))
        print('len(self.fnames): ', len(self.fnames))

        # get the num_classes from the folder names
        self.num_classes = len(set([fname.parent.name for fname in self.fnames]))
        print('retrieved folder counts: ', self.num_classes)

        self.caption_map = caption_map
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img = Image.open(self.fnames[i])
        folder_name = self.fnames[i].parent.name
       
        img_id = self.fnames[i].name.split('.')[0]
        file_path = str(self.fnames[i])

        label = int(folder_name)

        if self.transform is not None:
            img = self.transform(img)

        if self.caption_map is not None:
            caption = self.caption_map[str(label)][img_id]
            return img, label, file_path, caption
        else:
            return img, label, file_path, None