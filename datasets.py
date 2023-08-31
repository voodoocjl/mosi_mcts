import pickle
import torch
from torch.utils.data import Dataset
from Arguments import Arguments


class CustomDataset(Dataset):
    def __init__(self, audio, visual, text, target):
        self.audio = audio
        self.visual = visual
        self.text = text
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        audio_val = self.audio[index]
        visual_val = self.visual[index]
        text_val = self.text[index]
        target = self.target[index]
        return audio_val, visual_val, text_val, target


def MOSIDataLoaders(args):
    with open('data/mosi', 'rb') as file:
        tensors = pickle.load(file)
    AUDIO = 'COVAREP'
    VISUAL = 'FACET_4.2'
    TEXT = 'glove_vectors'
    TARGET = 'Opinion Segment Labels'

    train_data = tensors[0]
    train_audio = torch.from_numpy(train_data[AUDIO]).float()
    train_visual = torch.from_numpy(train_data[VISUAL]).float()
    train_text = torch.from_numpy(train_data[TEXT]).float()
    train_target = torch.from_numpy(train_data[TARGET]).squeeze()
    
    val_data = tensors[1]
    val_audio = torch.from_numpy(val_data[AUDIO]).float()
    val_visual = torch.from_numpy(val_data[VISUAL]).float()
    val_text = torch.from_numpy(val_data[TEXT]).float()
    val_target = torch.from_numpy(val_data[TARGET]).squeeze()
    
    test_data = tensors[2]
    test_audio = torch.from_numpy(test_data[AUDIO]).float()
    test_visual = torch.from_numpy(test_data[VISUAL]).float()
    test_text = torch.from_numpy(test_data[TEXT]).float()
    test_target = torch.from_numpy(test_data[TARGET]).squeeze()
    
    train = CustomDataset(train_audio, train_visual, train_text, train_target)
    val = CustomDataset(val_audio, val_visual, val_text, val_target)
    test = CustomDataset(test_audio, test_visual, test_text, test_target)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=val_audio.shape[0], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=test_audio.shape[0], pin_memory=True)
    return train_loader, val_loader, test_loader

def MOSEIDataLoaders(args):
    with open('data/mosei', 'rb') as file:
        tensors = pickle.load(file)
    
    AUDIO = "b'COAVAREP'"
    VISUAL = "b'FACET 4.2'"
    TEXT = "b'glove_vectors'"
    TARGET = "b'All Labels'"
    batch_size = 128

    train_data = tensors[0]
    train_audio = torch.from_numpy(train_data[AUDIO][:4000]).float().mean(dim=1)
    train_visual = torch.from_numpy(train_data[VISUAL][:4000]).float().mean(dim=1)
    train_text = torch.from_numpy(train_data[TEXT][:4000]).float().mean(dim=1)
    train_target = torch.from_numpy(train_data[TARGET][:4000])[:, 0, 0]
    
    val_data = tensors[1]
    val_audio = torch.from_numpy(val_data[AUDIO][:1000]).float().mean(dim=1)
    val_visual = torch.from_numpy(val_data[VISUAL][:1000]).float().mean(dim=1)
    val_text = torch.from_numpy(val_data[TEXT][:1000]).float().mean(dim=1)
    val_target = torch.from_numpy(val_data[TARGET][:1000])[:, 0, 0]
    
    test_data = tensors[2]
    test_audio = torch.from_numpy(test_data[AUDIO][:1500]).float().mean(dim=1)
    test_visual = torch.from_numpy(test_data[VISUAL][:1500]).float().mean(dim=1)
    test_text = torch.from_numpy(test_data[TEXT][:1500]).float().mean(dim=1)
    test_target = torch.from_numpy(test_data[TARGET][:1500])[:, 0, 0]
    
    train = CustomDataset(train_audio, train_visual, train_text, train_target)
    val = CustomDataset(val_audio, val_visual, val_text, val_target)
    test = CustomDataset(test_audio, test_visual, test_text, test_target)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=val_audio.shape[0], shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=test_audio.shape[0], shuffle=True, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    args = Arguments()
    train_loader, val_loader, test_loader = MOSIDataLoaders(args)
    for data_a, data_v, data_t, target in train_loader:
        print(data_a.shape, data_v.shape, data_t.shape, target.shape)
        print(data_a.dtype, data_v.dtype, data_t.dtype, target.dtype)
        break
    