import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path, test_frac, val_frac):
        self.path = path
        # load dataframe
        self.df = pd.read_csv(path)
        self.n_examples = self.df.shape[0]

        # shuffle it
        self.df = self.df.sample(frac=1)

        # split to train, test , validation
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.train_frac = 1 - self.test_frac + self.val_frac
        self.train_df = self.df

        self.test_df = self.train_df.sample(frac=self.test_frac)
        self.train_df = self.train_df.drop(self.test_df.index)

        val_number = self.n_examples * self.val_frac
        new_val_frac = val_number / self.train_df.shape[0]
        self.val_frac = new_val_frac
        self.val_df = self.train_df.sample(frac=self.val_frac)
        self.train_df = self.train_df.drop(self.val_df.index)

        # split labels and data
        self.train_labels = self.train_df.iloc[:, 1]
        self.train_data = self.train_df.iloc[:, 1:]

        self.test_labels = self.test_df.iloc[:, 1]
        self.test_data = self.test_df.iloc[:, 1:]

        self.val_labels = self.val_df.iloc[:, 1]
        self.val_data = self.val_df.iloc[:, 1:]

        # number of examples
        self.train_n_examples = self.train_df.shape[0]
        self.test_n_examples = self.test_df.shape[0]
        self.val_n_examples = self.val_df.shape[0]

        # delete unused df
        del (self.df)
        del (self.train_df)
        del (self.test_df)
        del (self.val_df)

        # convert to numpy
        self.train_data = self.train_data.to_numpy()
        self.train_labels = self.train_labels.to_numpy()

        self.test_data = self.test_data.to_numpy()
        self.test_labels = self.test_labels.to_numpy()

        self.val_data = self.val_data.to_numpy()
        self.val_labels = self.val_labels.to_numpy()


class Image(Data):
    def __init__(self, path, image_size, test_frac=0.2, val_frac=0.15, colour='rgb'):
        super().__init__(path, test_frac, val_frac)
        self.path = path
        self.test_frac = test_frac
        self.val_frac = val_frac
        self.image_size = image_size
        self.colour = colour

        if (self.colour == 'gray'):
            self.cmap = 'gray'
            self.channels = 1
        elif (self.colour == 'rgb'):
            self.cmap = 'viridis'
            self.channels = 3

        # reshape to images
        self.train_data = self.train_data.reshape(
            (self.train_n_examples, self.channels, image_size[0], image_size[1])) / 255.0
        self.train_labels = self.train_labels.reshape((self.train_n_examples, 1))

        self.test_data = self.test_data.reshape(
            (self.test_n_examples, self.channels, image_size[0], image_size[1])) / 255.0
        self.test_labels = self.test_labels.reshape((self.test_n_examples, 1))

        self.val_data = self.val_data.reshape(
            (self.val_n_examples, self.channels, image_size[0], image_size[1])) / 255.0
        self.val_labels = self.val_labels.reshape((self.val_n_examples, 1))

    def train2img(self, image):
        c, h, w = image.shape
        image = image.reshape(h, w, c)

    def get_img(self, image_number):
        img = self.train_data[image_number, :, :, :]
        c, h, w = img.shape
        img = img.reshape(h, w, c)
        return img

    def display_sample(self, samples=1):
        random_numbers = np.random.randint(low=0, high=self.train_n_examples - 1, size=samples)
        for number in random_numbers:
            img = self.get_img(number)
            plt.figure()
            plt.imshow(img, cmap=self.cmap)