import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print(f'model {args.model_path} does not exist')
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    metrics = ['psnr', 'ssim', 'psnr_y', 'ssim_y', 'psnrb', 'psnrb_y']
    test_results = {metric: [] for metric in metrics}

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # 读取图片
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # 模型推理
        with torch.no_grad():
            output, h_old, w_old = perform_inference(img_lq, model, window_size, args.scale, args)

        # save image
        output = save_image(output, save_dir, imgname)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y = evaluate_image(output, img_gt, test_results, args, border, h_old, w_old)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNRB: {:.2f} dB;'
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; PSNRB_Y: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        summarize_results(test_results, img_gt, args, save_dir)

def perform_inference(img_lq, model, window_size, scale, args):
    """执行推理，确保图像尺寸与窗口大小对齐"""
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    output = test(img_lq, model, args, window_size)
    return output[..., :h_old * scale, :w_old * scale], h_old, w_old

def summarize_results(test_results, img_gt, args, save_dir):
    """
    计算并打印 PSNR、SSIM 及其他指标的平均值。
    """
    # 计算并打印 PSNR 和 SSIM 平均值
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print(f'\n{save_dir} \n-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')

    # 如果是 RGB 图像，计算并打印 PSNR_Y 和 SSIM_Y 平均值
    if img_gt.ndim == 3:
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print(f'-- Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')

    # 如果任务是 JPEG 压缩伪影去除，计算并打印 PSNRB 平均值
    if args.task in ['jpeg_car', 'color_jpeg_car']:
        ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
        print(f'-- Average PSNRB: {ave_psnrb:.2f} dB')

        # 如果任务是 color_jpeg_car，计算并打印 PSNRB_Y 平均值
        if args.task == 'color_jpeg_car':
            ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
            print(f'-- Average PSNRB_Y: {ave_psnrb_y:.2f} dB')


def save_image(output, save_dir, imgname):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)
    return output

def evaluate_image(output, img_gt, test_results, args, border, h_old, w_old):
    img_gt = (img_gt * 255.0).round().astype(np.uint8)
    img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # Crop GT to match output size
    img_gt = np.squeeze(img_gt)
    psnr = util.calculate_psnr(output, img_gt, crop_border=border)
    ssim = util.calculate_ssim(output, img_gt, crop_border=border)
    test_results['psnr'].append(psnr)
    test_results['ssim'].append(ssim)
    psnrb, psnr_y, ssim_y, psnrb_y = 0, 0, 0, 0
    if img_gt.ndim == 3:  # RGB image
        psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
        ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
        test_results['psnr_y'].append(psnr_y)
        test_results['ssim_y'].append(ssim_y)
    if args.task in ['jpeg_car', 'color_jpeg_car']:
        psnrb = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=False)
        test_results['psnrb'].append(psnrb)
        if args.task == 'color_jpeg_car':  # 计算 Y 通道的 PSNRB
            psnrb_y = util.calculate_psnrb(output, img_gt, crop_border=border, test_y_channel=True)
            test_results['psnrb_y'].append(psnrb_y)
    return psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y

def define_model(args):
    task_config = {
        'classical_sr': {
            'upscale': args.scale, 'in_chans': 3, 'img_size': args.training_patch_size, 
            'window_size': 8, 'img_range': 1., 'depths': [6]*6, 'embed_dim': 180, 
            'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': 'pixelshuffle', 'resi_connection': '1conv',
            'param_key_g': 'params'
        },
        'lightweight_sr': {
            'upscale': args.scale, 'in_chans': 3, 'img_size': 64, 
            'window_size': 8, 'img_range': 1., 'depths': [6]*4, 'embed_dim': 60, 
            'num_heads': [6]*4, 'mlp_ratio': 2, 'upsampler': 'pixelshuffledirect', 'resi_connection': '1conv',
            'param_key_g': 'params'
        },
        'real_sr': {
            'upscale': args.scale, 'in_chans': 3, 'img_size': 64, 
            'window_size': 8, 'img_range': 1., 'depths': [6]*9 if args.large_model else [6]*6, 
            'embed_dim': 240 if args.large_model else 180, 
            'num_heads': [8]*9 if args.large_model else [6]*6, 'mlp_ratio': 2, 
            'upsampler': 'nearest+conv', 'resi_connection': '3conv' if args.large_model else '1conv',
            'param_key_g': 'params_ema' if args.large_model else 'params'
        },
        'gray_dn': {
            'upscale': 1, 'in_chans': 1, 'img_size': 128, 
            'window_size': 8, 'img_range': 1., 'depths': [6]*6, 'embed_dim': 180, 
            'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': '', 'resi_connection': '1conv',
            'param_key_g': 'params'
        },
        'color_dn': {
            'upscale': 1, 'in_chans': 3, 'img_size': 128, 
            'window_size': 8, 'img_range': 1., 'depths': [6]*6, 'embed_dim': 180, 
            'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': '', 'resi_connection': '1conv',
            'param_key_g': 'params'
        },
        'jpeg_car': {
            'upscale': 1, 'in_chans': 1, 'img_size': 126, 
            'window_size': 7, 'img_range': 255., 'depths': [6]*6, 'embed_dim': 180, 
            'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': '', 'resi_connection': '1conv',
            'param_key_g': 'params'
        },
        'color_jpeg_car': {
            'upscale': 1, 'in_chans': 3, 'img_size': 126, 
            'window_size': 7, 'img_range': 255., 'depths': [6]*6, 'embed_dim': 180, 
            'num_heads': [6]*6, 'mlp_ratio': 2, 'upsampler': '', 'resi_connection': '1conv',
            'param_key_g': 'params'
        }
    }
    
    if args.task not in task_config:
        raise ValueError(f"Unsupported task: {args.task}")
    config = task_config[args.task]
    param_key_g = config.pop('param_key_g')
    model = net(**config)
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model.get(param_key_g, pretrained_model), strict=True)

    return model


def setup(args):
    # 默认配置表
    task_configs = {
        'classical_sr': {
            'save_dir': f'results/swinir_classical_sr_x{args.scale}',
            'folder': args.folder_gt,
            'border': args.scale,
            'window_size': 8
        },
        'lightweight_sr': {
            'save_dir': f'results/swinir_lightweight_sr_x{args.scale}',
            'folder': args.folder_gt,
            'border': args.scale,
            'window_size': 8
        },
        'real_sr': {
            'save_dir': f'results/swinir_real_sr_x{args.scale}' + ('_large' if args.large_model else ''),
            'folder': args.folder_lq,
            'border': 0,
            'window_size': 8
        },
        'gray_dn': {
            'save_dir': f'results/swinir_gray_dn_noise{args.noise}',
            'folder': args.folder_gt,
            'border': 0,
            'window_size': 8
        },
        'color_dn': {
            'save_dir': f'results/swinir_color_dn_noise{args.noise}',
            'folder': args.folder_gt,
            'border': 0,
            'window_size': 8
        },
        'jpeg_car': {
            'save_dir': f'results/swinir_jpeg_car_jpeg{args.jpeg}',
            'folder': args.folder_gt,
            'border': 0,
            'window_size': 7
        },
        'color_jpeg_car': {
            'save_dir': f'results/swinir_color_jpeg_car_jpeg{args.jpeg}',
            'folder': args.folder_gt,
            'border': 0,
            'window_size': 7
        }
    }

    # 检查任务是否有效
    if args.task not in task_configs:
        raise ValueError(f"Invalid task: {args.task}. Supported tasks: {list(task_configs.keys())}")

    # 从配置表中提取参数
    config = task_configs[args.task]
    folder = config['folder']
    save_dir = config['save_dir']
    border = config['border']
    window_size = config['window_size']

    return folder, save_dir, border, window_size



def get_image_pair(args, path):
    def read_image(filepath, mode=cv2.IMREAD_COLOR):
        img = cv2.imread(filepath, mode)
        if img is None:
            raise FileNotFoundError(f"Image file not found: {filepath}")
        return img.astype(np.float32) / 255.

    def add_noise(image, noise_level):
        np.random.seed(0)
        noise = np.random.normal(0, noise_level / 255., image.shape)
        return image + noise

    imgname, imgext = os.path.splitext(os.path.basename(path))
    img_gt, img_lq = None, None

    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = read_image(path)
        lq_path = f'{args.folder_lq}/{imgname}x{args.scale}{imgext}'
        img_lq = read_image(lq_path)

    elif args.task == 'real_sr':
        img_lq = read_image(path)

    elif args.task == 'gray_dn':
        img_gt = read_image(path, mode=cv2.IMREAD_GRAYSCALE)
        img_lq = add_noise(img_gt, args.noise)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    elif args.task == 'color_dn':
        img_gt = read_image(path)
        img_lq = add_noise(img_gt, args.noise)

    elif args.task == 'jpeg_car':
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt.ndim != 2:
            img_gt = util.bgr2ycbcr(img_gt, y_only=True)
        _, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        img_gt = np.expand_dims(img_gt.astype(np.float32) / 255., axis=2)
        img_lq = np.expand_dims(img_lq.astype(np.float32) / 255., axis=2)

    elif args.task == 'color_jpeg_car':
        img_gt = read_image(path)
        _, encimg = cv2.imencode('.jpg', img_gt * 255., [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    def check_tile_size(tile, window_size):
        if tile % window_size != 0:
            raise ValueError("Tile size should be a multiple of window_size")
    if args.tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        check_tile_size(tile, window_size)
        tile_overlap = args.tile_overlap
        scale = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        E = torch.zeros(b, c, h * scale, w * scale, device=img_lq.device)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]

                out_patch = model(in_patch)

                h_start, h_end = h_idx * scale, (h_idx + tile) * scale
                w_start, w_end = w_idx * scale, (w_idx + tile) * scale

                E[..., h_start:h_end, w_start:w_end] += out_patch
                W[..., h_start:h_end, w_start:w_end] += 1

        output = E / W

    return output

if __name__ == '__main__':
    main()
