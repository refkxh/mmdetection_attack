from mmdet.apis import init_detector, inference_detector
import mmcv
import os

config_file = 'configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
input_dir = '../eval_code/select1000_new/'
output_dir = 'outputs/'
attack_iters = 300
attack_epsilon = 0.005

model = init_detector(config_file, checkpoint_file, device='cuda:0')

names = os.listdir(input_dir)
for cnt, name in enumerate(names):
    img = mmcv.imread(input_dir + name)
    result = inference_detector(model, img)
    model.show_result(img, result, out_file='outputs/{}'.format(name))
