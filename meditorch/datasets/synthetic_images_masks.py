import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms

class SyntheticImagesMasks(Dataset):

    def __init__(self, size=200, transform=None, result_resolution=(224,224)):

        self.transform = transform

        self.data, self.targets = self.generate_random_data(result_resolution[0], result_resolution[1], size)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is mask.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)
        else:
            transform_to_tensor = transforms.Compose([
                transforms.ToTensor(),
            ])
            img = transform_to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def generate_random_data(self, height, width, count):
        x, y = zip(*[self.generate_img_and_mask(height, width) for i in range(0, count)])

        X = np.asarray(x) * 255
        X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
        Y = np.asarray(y)

        return X, Y

    def generate_img_and_mask(self, height, width):
        shape = (height, width)

        triangle_location = self.get_random_location(*shape)
        circle_location1 = self.get_random_location(*shape, zoom=0.7)
        circle_location2 = self.get_random_location(*shape, zoom=0.5)
        mesh_location = self.get_random_location(*shape)
        square_location = self.get_random_location(*shape, zoom=0.8)
        plus_location = self.get_random_location(*shape, zoom=1.2)

        # Create input image
        arr = np.zeros(shape, dtype=bool)
        arr = self.add_triangle(arr, *triangle_location)
        arr = self.add_circle(arr, *circle_location1)
        arr = self.add_circle(arr, *circle_location2, fill=True)
        arr = self.add_mesh_square(arr, *mesh_location)
        arr = self.add_filled_square(arr, *square_location)
        arr = self.add_plus(arr, *plus_location)
        arr = np.reshape(arr, (1, height, width)).astype(np.float32)

        # Create target masks
        masks = np.asarray([
            self.add_filled_square(np.zeros(shape, dtype=bool), *square_location),
            self.add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
            self.add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
            self.add_circle(np.zeros(shape, dtype=bool), *circle_location1),
             self.add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
            # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
            self.add_plus(np.zeros(shape, dtype=bool), *plus_location)
        ]).astype(np.float32)

        return arr, masks

    def add_square(self, arr, x, y, size):
        s = int(size / 2)
        arr[x-s,y-s:y+s] = True
        arr[x+s,y-s:y+s] = True
        arr[x-s:x+s,y-s] = True
        arr[x-s:x+s,y+s] = True

        return arr

    def add_filled_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self.logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))

    def logical_and(self, arrays):
        new_array = np.ones(arrays[0].shape, dtype=bool)
        for a in arrays:
            new_array = np.logical_and(new_array, a)

        return new_array

    def add_mesh_square(self, arr, x, y, size):
        s = int(size / 2)

        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

        return np.logical_or(arr, self.logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))

    def add_triangle(self, arr, x, y, size):
        s = int(size / 2)

        triangle = np.tril(np.ones((size, size), dtype=bool))

        arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

        return arr

    def add_circle(self, arr, x, y, size, fill=False):
        xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
        circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

        return new_arr

    def add_plus(self, arr, x, y, size):
        s = int(size / 2)
        arr[x-1:x+1,y-s:y+s] = True
        arr[x-s:x+s,y-1:y+1] = True

        return arr

    def get_random_location(self, width, height, zoom=1.0):
        x = int(width * random.uniform(0.1, 0.9))
        y = int(height * random.uniform(0.1, 0.9))

        size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

        return (x, y, size)
