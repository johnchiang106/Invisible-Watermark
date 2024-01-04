import numpy as np
from numpy.linalg import svd
import copy
import cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2
from tqdm import tqdm


class WaterMarkCore:
    def __init__(self, password_img=1):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20 # increasing d1/d2 will increase robustness, but also causing distortion

        # init data
        self.img, self.img_YUV = None, None # original image, YUV image
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # dct results for each channels
        self.ca_block = [np.array([])] * 3 # 4-dimention-array for each channel, representing the results of 4-dimention-block
        self.ca_part = [np.array([])] * 3  # storing the truncated part of self.ca after dividing into 4-dimention-block, since it might not be able to divided exactly

        self.wm_size, self.block_num = 0, 0  # watermark size, number of maximum kb that the image can embed

        self.alpha = None  # For RGBA format

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            'At most {} kb can be embedded, you have {} kb, overflow!'.format(self.block_num / 1000, self.wm_size / 1000))
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # read image->convert to YUV->add padding to make pixels even number->4 dimention block
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # add padding if not divisible by 2
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # transform to 4 dimentions
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size
    

    def block_add_wm(self, arg, mode = "dct"):
        block, shuffler, i = arg
        # dct->(flatten->shuffle->unflatten)->svd->embed watermark->inverse svd->(flatten->unshffle->unflatten)->inverse dct
        wm_1 = self.wm_bit[i % self.wm_size]

        if mode == "fft":
            block = np.fft.fftshift((block))
        else:
            block = dct(block)
            # print(block)
        # shuffle
        block_shuffled = block.flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_shuffled)
        
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2
        
        if mode == "fft":
            block_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
            block_flatten[shuffler] = block_flatten.copy()
            block_flatten = np.fft.ifftshift(block_flatten)
            block_flatten = np.fft.ifftshift(block_flatten).reshape(self.block_shape)
        else:
            block_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
            block_flatten[shuffler] = block_flatten.copy()
            block_flatten = idct(block_flatten.reshape(self.block_shape))
            # print(block_flatten)
        return block_flatten
    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        self.idx_shuffle = random_strategy(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in tqdm(range(3)):
            tmp = list(map(self.block_add_wm, [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)]))

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # 4d to 2d
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # truncate the right and the bottom part that is not divisible when dividing into 4 dimention block
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # inverse DWT
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # stack all 3 channels
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # remove the added padding if not divisible by 2
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm_slow(self, args):
        block, shuffler = args
        # dct->flatten->shuffle->unflatten->svd->extract watermark
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract_raw(self, img):
        # Extract bit by bit from each block
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3 channels，self.block_num blocks

        self.idx_shuffle = random_strategy(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in tqdm(range(3), desc="Extracting raw..."):
            wm_block_bit[channel, :] = list(map(self.block_get_wm_slow, [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)]))
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # average 3 channels
        wm_avg = np.zeros(shape=self.wm_size)
        for i in tqdm(range(self.wm_size), desc="Extracting average..."):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        # extract the embedded bits from each block
        wm_block_bit = self.extract_raw(img=img)
        # average the extracted values of 3 channels
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg


def random_strategy(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)

