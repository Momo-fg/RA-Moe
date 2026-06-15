import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path('./RetinaFace')

from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.timer import Timer
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.box_utils import decode, decode_landm
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms

args = {
    'network': 'resnet50',  
    'confidence_threshold': 0.02,
    'nms_threshold': 0.4,
    'vis_thres': 0.5,
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, device):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Facedetecor():
    def __init__(self, model_path, device):        
        torch.set_grad_enabled(False)
        cfg = cfg_re50
        net = RetinaFace(cfg=cfg, phase='test')
        net = load_model(net, model_path, device)
        net.eval()
        print('retinaface: Finished loading model!')
        cudnn.benchmark = True
        net = net.to(device)
        self.net = net
        self.device = device
        self.cfg = cfg

    def get_face_box(self, img):
        if img is None or img.size == 0:
            return None  # 防止空图像

        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance']) * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # 筛选得分高的框
        inds = np.where(scores > args['confidence_threshold'])[0]
        if len(inds) == 0:
            return None

        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args['nms_threshold'])

        if len(keep) == 0:
            return None

        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        # 再次检查是否还有检测结果
        if dets.shape[0] == 0:
            return None

        b = dets[0]
        if b[4] > args['vis_thres']:
            return int(b[0]), int(b[1]), int(b[2]), int(b[3])
        else:
            return None
