import numpy as np
import torch.utils.data as data
from PIL import Image
from os.path import join


class Dataset(data.Dataset):
    def __init__(self, data_dir, filenames, input_transform,
                 target_transform, target_transform_binary):
        super(Dataset, self).__init__()
        image_dir = join(data_dir, 'images')
        filenames_lookup = set(filenames)
        fname_to_attr = {}
        # This is a pain to do in pandas because there is no column for the
        # filename.
        # Need to get the attributes for the specified filenames
        with open(join(data_dir, 'list_attr_celeba.txt'), 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    pass  # First line contains just number of lines
                elif i == 1:
                    attr_names = np.array(line.strip().split())
                else:
                    fname_attrs = line.strip().split()
                    fname, attrs = fname_attrs[0], fname_attrs[1:]
                    # Avoid loading unnecessary attributes into memory
                    if fname in filenames_lookup:
                        fname_to_attr[fname] = np.array(attrs, dtype=np.int32)

        self.image_filenames = [join(image_dir, x) for x in filenames]
        attr_vals = np.vstack(fname_to_attr[fname] for fname in filenames)
        self.attribute_names  = attr_names
        self.attribute_values = attr_vals
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.target_transform_binary = target_transform_binary

    def __getitem__(self, index):
        fp = self.image_filenames[index]
        x = self.input_transform(Image.open(fp))
        yb = self.target_transform_binary(self.attribute_values[index])
        yt = self.target_transform(self.attribute_values[index])

        return x, yb, yt, fp

    def __len__(self):
        return len(self.image_filenames)
