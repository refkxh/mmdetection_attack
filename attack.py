from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import time

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
    start_t = time.time()
    result = inference_detector(model, img)
    end_t = time.time()
    cost_t = 1000 * (end_t - start_t)
    print("===>successfully processed img %d, cost %.2f ms." % (cnt + 1, cost_t))
    # model.show_result(img, result, out_file=(output_dir + name))
