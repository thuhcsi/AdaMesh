import os
import glob
import torch
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from torch.utils import data


def read_txt(txt_path):
    """Read .txt

    Args:
        txt-path (str): Save path of .txt.
    Return:
        content (list[str]): Each element in list corresponds to a single line in .txt.
    """
    with open(txt_path) as f:
        content = [line.strip() for line in f]
    return content


def write_txt(content, txt_path):
    """Write to .txt

    Args:
        content (list[str]): Each element in list corresponds to a single line in .txt.
        txt-path (str): Save path of .txt.
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(content))


def get_filelist(data_dir, list_subject):
    filelist_train, filelist_test = [], []
    for subject in list_subject:
        sub_dir = os.path.join(data_dir, subject, "deca_feature")
        # if dataset already split into train and test subfolders
        if "test" in os.listdir(sub_dir):
            filelist_train += sorted(glob.glob(os.path.join(sub_dir, "train", "**/*_3dparams.pt"), recursive=True))
            filelist_test += sorted(glob.glob(os.path.join(sub_dir, "test", "**/*_3dparams.pt"), recursive=True))

        # if dataset not split into train and test subfolders, but has corresponding .txt files
        elif "file_list_test.txt" in os.listdir(sub_dir):
            list_train_item = read_txt(os.path.join(sub_dir, "file_list_train_filter.txt"))
            list_test_item = read_txt(os.path.join(sub_dir, "file_list_test_filter.txt"))
            filelist_train += sorted([os.path.join(sub_dir, item, f"{item}_3dparams.pt") for item in list_train_item])
            filelist_test += sorted([os.path.join(sub_dir, item, f"{item}_3dparams.pt") for item in list_test_item])
        
        else:
            raise ValueError("No data found!")

    return filelist_train, filelist_test


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, statiscs_path=None):
        self.data = data
        self.len = len(self.data)
        self.statiscs_path = statiscs_path
        if self.statiscs_path is not None:
            stcs_dict = np.load(statiscs_path, allow_pickle=True).item()
            self.mean, self.std = stcs_dict['mean'], stcs_dict['std']
            # print(self.mean)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        exp_file_path = self.data[index]

        param = torch.load(exp_file_path)
        expression = param["exp"].numpy()
        pose = param["pose"].numpy()
        cam = param["cam"].numpy()
        cam[:, 0] = 1 / cam[:, 0]
        motion_all = np.concatenate((expression, pose, cam), axis=1)
        if self.statiscs_path is not None:
            motion_all = (motion_all - self.mean) / self.std
        
        motion = np.concatenate((motion_all[:, :50], motion_all[:, 53:56]), axis=1)
        
        speaker_dir = exp_file_path.split("deca_feature")[0]
        basename = os.path.basename(exp_file_path).split("_3dparams.pt")[0]
        speech_feat_path = os.path.join(speaker_dir, "hubert_feature", basename + "_hubert.npy")
        speech_feat = np.load(speech_feat_path)

        assert abs(speech_feat.shape[0] // 2 - motion.shape[0]) < 5, f"{exp_file_path} Speech and exp feature length wrong!"
        exp_len = min(speech_feat.shape[0] // 2, motion.shape[0])

        speech_feat= speech_feat[:exp_len * 2]
        motion = motion[:exp_len]

        return {"exp_path": exp_file_path, "exp_len": exp_len,
                "speech_feat": torch.FloatTensor(speech_feat), 
                "motion": torch.FloatTensor(motion)}

    def __len__(self):
        return self.len


class BatchCollate(object):
  """ Collates batch with padding and decreasing sort by input length. """

  def __init__(self, motion_dim=59):
      self.motion_dim = motion_dim

  def __call__(self, batch):
    batch_size = len(batch)

    # Sorting batch by length of inputs
    _, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([x["exp_len"] for x in batch]), dim=0, descending=True)

    max_ouput_len = torch.max(torch.LongTensor([x["exp_len"] for x in batch]))

    speech_feat_padded = torch.zeros(batch_size, max_ouput_len * 2, 1024)
    motion_padded = torch.zeros(batch_size, max_ouput_len, self.motion_dim)
    output_mask = torch.zeros(batch_size, max_ouput_len, dtype=torch.bool)

    input_len = []
    for index, i in enumerate(ids_sorted_decreasing):
      x = batch[i]
      exp_len = x["exp_len"]
      input_len.append(exp_len * 2)
      
      speech_feat_padded[index, 0:exp_len*2, :] = x["speech_feat"]
      motion_padded[index, 0:exp_len, :] = x["motion"]
      
      output_mask[index, 0:exp_len] = True
      
    return speech_feat_padded, motion_padded, output_mask, torch.LongTensor(input_len)


def read_data(args):
    print("Loading data...")
    train_data = []
    test_data = []

    list_subject = [i for i in args.subjects.split(" ")]
    train_data, test_data = get_filelist(args.data_root, list_subject)

    print('Loaded data: Train-{}, Test-{}'.format(len(train_data), len(test_data)))
    return train_data, test_data

def get_dataloaders(args):
    dataset = {}
    train_data, test_data = read_data(args)
    train_data = Dataset(train_data, args.statiscs_path)
    valid_data = Dataset(test_data, args.statiscs_path)
    if args.batch_size == 1:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    else:
        dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, 
                                           num_workers=args.workers, collate_fn=BatchCollate(args.out_dim), 
                                           drop_last=False)
        dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.workers, collate_fn=BatchCollate(args.out_dim), 
                                           drop_last=False)
    # if args.model_path is not None:
    #     dataset["test"] = data.DataLoader(dataset=valid_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    return dataset


if __name__ == "__main__":
    get_dataloaders()