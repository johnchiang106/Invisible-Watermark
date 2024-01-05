import numpy as np
import cv2

from watermark_core import WaterMarkCore

class WaterMark:
    def __init__(self, password_wm=1, password_img=1):
        print("Welcome!!")

        self.iwm_core = WaterMarkCore(password_img=password_img)

        self.password_wm = password_wm

        self.wm_bit = None
        self.wm_size = 0
        print("Finish INIT")

    def read_img(self, filename=None, img=None):
        print("Reading img...")
        if img is None:
            # Read the image from a file
            img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            assert img is not None, f"Image file '{filename}' not read"

        self.iwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm_content):
        print("Reading wm...")
        wm = cv2.imread(filename=wm_content, flags=cv2.IMREAD_GRAYSCALE)
        assert wm is not None, f'File "{wm_content}" not read'
        # Read the image-format watermark and convert it to one-dimensional bit format, discarding grayscale levels
        self.wm_bit = wm.flatten() > 128
        self.wm_size = self.wm_bit.size

        # Watermark encryption:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.iwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None, compression_ratio=None):
        '''
        :param filename: string
            Save the image file as filename
        :param compression_ratio: int or None
            If compression_ratio = None, do not compress,
            If compression_ratio is an integer between 0 and 100, the smaller, the output file is smaller.
        :return:
        '''
        print("Embedding...")
        embed_img = self.iwm_core.embed()
        if filename is not None:
            if compression_ratio is None:
                cv2.imwrite(filename=filename, img=embed_img)
            elif filename.endswith('.jpg'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
            elif filename.endswith('.png'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_PNG_COMPRESSION, compression_ratio])
            else:
                cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def extract_decrypt(self, wm_avg):
        print("Extracting & Decrypting...")
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None):
        print("Extracting...")
        assert wm_shape is not None, 'wm_shape needed'

        if filename is not None:
            embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
            assert embed_img is not None, f"{filename} not read"

        self.wm_size = np.array(wm_shape).prod()
        wm_avg = self.iwm_core.extract(img=embed_img, wm_shape=wm_shape)

        # Decrypt:
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # Convert to the specified format:
        wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
        cv2.imwrite(out_wm_name, wm)

        return wm