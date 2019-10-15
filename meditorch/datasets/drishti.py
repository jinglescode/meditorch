import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from meditorch.utils.files import makedir_exist_ok, download_and_extract_archive
from meditorch.utils.images import load_set, load_image, resize_image_to_square

class Drishti(Dataset):

    """`Drishti-GS <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php>`_ Dataset.

    Full images with optic disc and optic cup segmentation.
    Average segmentation and "softmap" segmentation image are given.
    50 images of various resolution close to 2040 x 1740.
    Data set is split into training and test sets.

    Required schema:
    db_folder/
        Drishti-GS1_files/
            Training/
                Images/
                    drishtiGS_{:03}.png    # some numbers are omitted, like 001, 003, 004, ...
                GT/
                    drishtiGS_{:03}/
                        drishtiGS_{:03}_cdrValues.txt
                        AvgBoundary/
                            drishtiGS_{:03}_ODAvgBoundary.txt
                            drishtiGS_{:03}_CupAvgBoundary.txt
                            drishtiGS_{:03}_diskCenter.txt
                        SoftMap/
                            drishtiGS_{:03}_ODsegSoftmap.png
                            drishtiGS_{:03}_cupsegSoftmap.png

    Args:
        root (string): Root directory of dataset.
        train (bool, optional): If True, read images from ``Training`` folder, otherwise from ``Test``. (default: True)
        return_disc (bool, optional): ``targets`` will contain segmentation of optic disc. (default: True)
        return_cup (bool, optional): ``targets`` will contain segmentation of optic cup. (default: True)
        result_resolution (tuple, optional): ``data`` and ``targets`` will be resized to this resolution. (default: (224,224))
    """

    def __init__(self, root, train=True, transform=None, return_disc=True, return_cup=True, result_resolution=(224,224)):

        self.root = root
        self.transform = transform

        self.data = None
        self.targets = None

        self._download()

        self._extract_images_and_segments(train, return_disc, return_cup, result_resolution)

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

    def _extract_images_and_segments(self, is_train, return_disc, return_cup, result_resolution):
        disc_all, cup_all = [], []
        # file_codes_all = []

        if is_train:
            set_path = os.path.join(self.extracted_folder, 'Training')
        else:
            set_path = os.path.join(self.extracted_folder, 'Test')

        print('Extracting data from', set_path)

        images_path = os.path.join(set_path, 'Images')
        X_all, file_names = load_set(folder=images_path)

        if len(file_names) == 0:
            raise Exception('No files')

        rel_file_names = [os.path.split(fn)[-1] for fn in file_names]
        rel_file_names_wo_ext = [fn[:fn.rfind('.')] for fn in rel_file_names]
        # file_codes = ['Training' + fn[fn.find('_'):] for fn in rel_file_names_wo_ext]
        # file_codes_all.extend(file_codes)

        for fn in rel_file_names_wo_ext:
            gt_folder = 'GT'
            if not is_train:
                gt_folder = 'Test_GT'

            if return_disc:
                disc_segmn = load_image(os.path.join(set_path, gt_folder, fn, 'SoftMap', fn + '_ODsegSoftmap.png'))
                disc_all.append(disc_segmn)

            if return_cup:
                cup_segmn = load_image(os.path.join(set_path, gt_folder, fn, 'SoftMap', fn + '_cupsegSoftmap.png'))
                cup_all.append(cup_segmn)

        for i in range(len(X_all)):
            side = result_resolution[0]

            X_all[i] = resize_image_to_square(X_all[i], side, pad_cval=0)

            if return_disc:
                disc_all[i] = resize_image_to_square(disc_all[i], side, pad_cval=0)
                disc_all[i] = disc_all[i].reshape(disc_all[i].shape + (1,))
            if return_cup:
                cup_all[i] = resize_image_to_square(cup_all[i], side, pad_cval=0)
                cup_all[i] = cup_all[i].reshape(cup_all[i].shape + (1,))

        input_images = np.asarray(X_all)
        input_images = input_images.astype(np.uint8)

        if return_disc:
            disc_converted = np.asarray(disc_all)
            disc_converted = np.where(disc_converted > 0, 1, 0)
            disc_converted = disc_converted.astype(np.float32)

        if return_cup:
            cup_converted = np.asarray(cup_all)
            cup_converted = np.where(cup_converted > 0, 1, 0)
            cup_converted = cup_converted.astype(np.float32)

        if return_disc and return_cup:
            target_masks = np.concatenate((disc_converted, cup_converted), axis=3)
        elif return_disc:
            target_masks = disc_converted
        elif return_cup:
            target_masks = cup_converted

        target_masks = np.rollaxis(target_masks, 3, 1)

        self.data = input_images
        self.targets = target_masks

        print('Completed extracting `data` and `targets`.')

    @property
    def data_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def extracted_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'Drishti-GS1_files', 'Drishti-GS1_files')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.data_folder)))

    def _download(self):

        if self._check_exists():
            return

        makedir_exist_ok(self.data_folder)

        urls = ['https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Drishti-GS1_files.rar']

        for url in urls:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.data_folder, filename=filename)

        print('Finished download and extract')
