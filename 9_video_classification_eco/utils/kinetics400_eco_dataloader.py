# 9장 동영상 분류(ECO: Efficient 3DCNN)
# 9.4 Kinetics 동영상 데이터 세트로 DataLoader 작성

# 필요한 패키지 import
import os
from PIL import Image
import csv
import numpy as np

import torch
import torch.utils.data
from torch import nn

import torchvision


def make_datapath_list(root_path):
    """
   동영상을 화상 데이터로 만든 폴더의 파일 경로 리스트를 작성한다.
    root_path : str, 데이터 폴더로의 root 경로
    Returns: ret : video_list, 동영상을 화상 데이터로 만든 폴더의 파일 경로 리스트
    """

    # 동영상을 화상 데이터로 만든 폴더의 파일 경로 리스트
    video_list = list()

    # root_path의 클래스 종류와 경로를 취득
    class_list = os.listdir(path=root_path)

    # 각 클래스의 동영상 파일을 화상으로 만든 폴더의 경로를 취득
    for class_list_i in (class_list):  # 클래스별로 루프

        # 클래스의 폴더 경로를 취득
        class_path = os.path.join(root_path, class_list_i)

        # 각 클래스의 폴더 내 화상 폴더를 취득하는 루프
        for file_name in os.listdir(class_path):

            # 파일명과 확장자로 분할
            name, ext = os.path.splitext(file_name)

            # mp4 파일이 아니거나, 폴더 등은 무시
            if ext == '.mp4':
                continue

            # 동영상 파일을 화상으로 분할해 저장한 폴더의 경로를 취득
            video_img_directory_path = os.path.join(class_path, name)

            # vieo_list에 추가
            video_list.append(video_img_directory_path)

    return video_list


class VideoTransform():
    """
    동영상을 화상으로 만드는 전처리 클래스. 학습시와 추론시 다르게 작동합니다.
    동영상을 화상으로 분할하고 있으므로, 분할된 화상을 한꺼번에 전처리하는 점에 주의하십시오.
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                # DataAugumentation()  # 이번에는 생략
                GroupResize(int(resize)),  # 화상을 한꺼번에 리사이즈
                GroupCenterCrop(crop_size),  # 화상을 한꺼번에 center crop
                GroupToTensor(),  # 데이터를 PyTorch 텐서로
                GroupImgNormalize(mean, std),  # 데이터를 표준화
                Stack()  # 여러 화상을 frames차원으로 결합시킨다
            ]),
            'val': torchvision.transforms.Compose([
                GroupResize(int(resize)),  # 화상을 한꺼번에 리사이즈
                GroupCenterCrop(crop_size),  # 화상을 한꺼번에 center crop
                GroupToTensor(),  # 데이터를 PyTorch 텐서로
                GroupImgNormalize(mean, std),  # 데이터를 표준화
                Stack()  # 여러 화상을 frames차원으로 결합시킨다
            ])
        }

    def __call__(self, img_group, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리 모드 지정
        """
        return self.data_transform[phase](img_group)


# 전처리로 사용할 클래스들을 정의
class GroupResize():
    '''화상 크기를 한꺼번에 재조정(rescale)하는 클래스.
    화상의 짧은 변의 길이가 resize로 변환된다.
    화면 비율은 유지된다.
    '''

    def __init__(self, resize, interpolation=Image.BILINEAR):
        '''rescale 처리 준비'''
        self.rescaler = torchvision.transforms.Resize(resize, interpolation)

    def __call__(self, img_group):
        '''img_group(리스트)의 각 img에 rescale 실시'''
        return [self.rescaler(img) for img in img_group]


class GroupCenterCrop():
    ''' 화상을 한꺼번에 center crop 하는 클래스.
        (crop_size, crop_size)의 화상을 잘라낸다.
    '''

    def __init__(self, crop_size):
        '''center crop 처리를 준비'''
        self.ccrop = torchvision.transforms.CenterCrop(crop_size)

    def __call__(self, img_group):
        '''img_group(리스트)의 각 img에 center crop 실시'''
        return [self.ccrop(img) for img in img_group]


class GroupToTensor():
    '''화상을 한꺼번에 텐서로 만드는 클래스.
    '''

    def __init__(self):
        '''텐서화하는 처리를 준비'''
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        '''img_group(리스트)의 각 img에 텐서화 실시
        0부터 1까지가 아니라, 0부터 255까지를 다루므로, 255를 곱해서 계산한다.
        0부터 255로 다루는 것은, 학습된 데이터 형식에 맞추기 위함
        '''

        return [self.to_tensor(img)*255 for img in img_group]


class GroupImgNormalize():
    '''화상을 한꺼번에 표준화하는 클래스.
    '''

    def __init__(self, mean, std):
        '''표준화 처리를 준비'''
        self.normlize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        '''img_group(리스트)의 각 img에 표준화 실시'''
        return [self.normlize(img) for img in img_group]


class Stack():
    ''' 화상을 하나의 텐서로 정리하는 클래스.
    '''

    def __call__(self, img_group):
        '''img_group은 torch.Size([3, 224, 224])를 요소로 하는 리스트
        '''
        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
                         for x in img_group], dim=0)  # frames 차원으로 결합
        # x.flip(dims=[0])은 색상 채널을 RGB에서 BGR으로 순서를 바꾸고 있습니다(원래의 학습 데이터가 BGR이었기 때문입니다)
        # unsqueeze(dim=0)은 새롭게 frames용의 차원을 작성하고 있습니다

        return ret


def get_label_id_dictionary(label_dicitionary_path='./video_download/kinetics_400_label_dicitionary.csv'):
    label_id_dict = {}
    id_label_dict = {}

    # encoding은 Ubuntu에서도 이것으로 좋은지 확인 필요
    with open(label_dicitionary_path, encoding="utf-8_sig") as f:

        # 읽어들임
        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        # 한 줄씩 읽어 사전형 변수에 추가합니다
        for row in reader:
            label_id_dict.setdefault(
                row["class_label"], int(row["label_id"])-1)
            id_label_dict.setdefault(
                int(row["label_id"])-1, row["class_label"])

    return label_id_dict,  id_label_dict


class VideoDataset(torch.utils.data.Dataset):
    """
    동영상 Dataset
    """

    def __init__(self, video_list, label_id_dict, num_segments, phase, transform, img_tmpl='image_{:05d}.jpg'):
        self.video_list = video_list  # 동영상 폴더의 경로 리스트
        self.label_id_dict = label_id_dict  # 라벨명을 id로 변환하는 사전형 변수
        self.num_segments = num_segments  # 동영상을 어떻게 분할해 사용할지를 결정
        self.phase = phase  # train or val
        self.transform = transform  # 전처리
        self.img_tmpl = img_tmpl  # 읽어들일 화상 파일명의 템플릿

    def __len__(self):
        '''동영상 수를 반환'''
        return len(self.video_list)

    def __getitem__(self, index):
        '''
        전처리한 화상들의 데이터와 라벨, 라벨 ID를 취득
        '''
        imgs_transformed, label, label_id, dir_path = self.pull_item(index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item(self, index):
        '''전처리한 화상들의 데이터와 라벨, 라벨 ID를 취득'''

        # 1. 화상들을 리스트에서 읽기
        dir_path = self.video_list[index]  # 화상이 저장된 폴더
        indices = self._get_indices(dir_path)  # 읽어 들일 화상 idx를 구하기
        img_group = self._load_imgs(
            dir_path, self.img_tmpl, indices)  # 리스트로 읽기

        # 2. 라벨을 취득해 id로 변환
        label = (dir_path.split('/')[3].split('/')[0])  # 주의: windows OS의 경우
        label_id = self.label_id_dict[label]  # id를 취득

        # 3. 전처리 실시
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, dir_path

    def _load_imgs(self, dir_path, img_tmpl, indices):
        '''화상을 한꺼번에 읽어들여, 리스트화하는 함수'''
        img_group = []  # 화상을 저장할 리스트

        for idx in indices:
            # 화상 경로 취득
            file_path = os.path.join(dir_path, img_tmpl.format(idx))

            # 화상 읽기
            img = Image.open(file_path).convert('RGB')

            # 리스트에 추가
            img_group.append(img)
        return img_group

    def _get_indices(self, dir_path):
        """
        동영상 전체를 self.num_segment로 분할했을 때의 동영상 idx의 리스트를 취득
        """
        # 동영상 프레임 수 구하기
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)

        # 동영상의 간격을 구하기
        tick = (num_frames) / float(self.num_segments)
        # 250 / 16 = 15.625
        # 동영상 간격으로 꺼낼 때 idx를 리스트로 구하기
        indices = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)])+1
        # 250frame에서 16frame 추출의 경우
        # indices = [  8  24  40  55  71  86 102 118 133 149 165 180 196 211 227 243]

        return indices
