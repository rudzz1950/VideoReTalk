import numpy as np
import cv2, argparse, torch
import torchvision.transforms.functional as TF

from models import load_network, load_DNet
from tqdm import tqdm
from PIL import Image
from scipy.spatial import ConvexHull
from third_part import face_detection
from third_part.face3d.models import networks

import warnings
warnings.filterwarnings("ignore")

def options():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--DNet_path', type=str, default='checkpoints/DNet.pt')
    parser.add_argument('--LNet_path', type=str, default='checkpoints/LNet.pth')
    parser.add_argument('--ENet_path', type=str, default='checkpoints/ENet.pth') 
    parser.add_argument('--face3d_net_path', type=str, default='checkpoints/face3d_pretrain_epoch_20.pth')                      
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--exp_img', type=str, help='Expression template. neutral, smile or image path', default='neutral')
    parser.add_argument('--outfile', type=str, help='Video path to save result')

    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 30, 0, 0], help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=8)
    parser.add_argument('--LNet_batch_size', type=int, help='Batch size for LNet', default=32)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--mel_step_size', type=int, default=12,
                        help='Number of mel frames per chunk. Lower = tighter sync, higher = smoother')
    parser.add_argument('--mel_multiplier_scale', type=float, default=1.0,
                        help='Scale factor for mel-to-video frame mapping. <1 if lips ahead; >1 if lagging')
    parser.add_argument('--gpen_size', type=int, default=512,
                        help='GPEN enhancer internal size (e.g., 512 or 1024)')
    parser.add_argument('--gpen_model', type=str, default='GPEN-BFR-512',
                        help='GPEN model name to load from checkpoints (e.g., GPEN-BFR-512)')
    # Audio preprocessing controls
    parser.add_argument('--audio_preprocess', action='store_true', default=True,
                        help='Preprocess audio with ffmpeg: force 16kHz mono and apply optional filters')
    parser.add_argument('--audio_normalize', action='store_true', default=True,
                        help='Normalize audio levels (ffmpeg dynaudnorm) during preprocessing')
    parser.add_argument('--audio_denoise', action='store_true', default=False,
                        help='Apply simple spectral denoise (ffmpeg afftdn) during preprocessing')
    parser.add_argument('--audio_target_sr', type=int, default=16000,
                        help='Target audio sampling rate for preprocessing')
    parser.add_argument('--audio_eps_on_nan', action='store_true', default=True,
                        help='If mel contains NaN, add epsilon noise to audio and retry once')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--static', default=False, action='store_true')

    
    parser.add_argument('--up_face', default='original')
    parser.add_argument('--one_shot', action='store_true')
    parser.add_argument('--without_rl1', default=False, action='store_true', help='Do not use the relative l1')
    parser.add_argument('--tmp_dir', type=str, default='temp', help='Folder to save tmp results')
    parser.add_argument('--re_preprocess', action='store_true')
    parser.add_argument('--temporal_smooth_window', type=int, default=5,
                        help='Window size for smoothing face detections over time when temporal smoothing is enabled')
    parser.add_argument('--lip_temporal_smooth', action='store_true',
                        help='Enable temporal smoothing for synthesized lip frames before blending back to the video')
    parser.add_argument('--temporal_blend_weight', type=float, default=0.3,
                        help='Blend weight used when lip temporal smoothing is enabled (0 disables, closer to 1 increases smoothing)')
    parser.add_argument('--temporal_blend_auto', action='store_true', default=False,
                        help='Adapt smoothing by mel energy (less smoothing on loud phonemes)')
    parser.add_argument('--temporal_blend_k', type=float, default=5.0,
                        help='Sharpness of energy-to-weight mapping when temporal_blend_auto is enabled')
    parser.add_argument('--mouth_region_smoothing', type=float, default=None,
                        help='Override lip temporal smoothing weight (0..1). Higher reduces jitter, lower keeps more detail')
    parser.add_argument('--lip_temporal_mode', type=str, default='ema', choices=['ema', 'median', 'both'],
                        help='Temporal smoothing mode for lips: ema (prev-frame), median (window), or both')
    parser.add_argument('--lip_temporal_window', type=int, default=5,
                        help='Window size for median temporal smoothing when enabled')
    parser.add_argument('--mel_offset_frames', type=int, default=0,
                        help='Global offset in mel frames applied to alignment (positive -> later)')
    parser.add_argument('--auto_sync_calibrate', action='store_true', default=True,
                        help='Estimate best mel-frame offset by correlating mouth openness and mel energy')
    parser.add_argument('--calibrate_window', type=int, default=6,
                        help='Max absolute frame offset to search during calibration')
    parser.add_argument('--auto_sync_scale', action='store_true', default=False,
                        help='Also search for a mel/frame scale factor during sync calibration')
    parser.add_argument('--sync_scale_range', type=float, default=0.08,
                        help='Percent range (+/-) for scale search when auto_sync_scale is enabled (e.g. 0.08 => ±8%)')
    parser.add_argument('--sync_scale_steps', type=int, default=5,
                        help='Number of steps on each side of scale search (total samples = 2*steps+1)')
    # Sync gating controls
    parser.add_argument('--enable_syncnet_gate', action='store_true', default=False,
                        help='Gate mouth blending based on lip–audio sync confidence')
    parser.add_argument('--lip_sync_conf_threshold', type=float, default=0.85,
                        help='Confidence threshold (0..1) to accept mouth update when gating is enabled')
    parser.add_argument('--syncnet_window', type=int, default=5,
                        help='Temporal window (frames) for sync confidence correlation')
    parser.add_argument('--syncnet_model_path', type=str, default='checkpoints/syncnet.pt',
                        help='Optional TorchScript SyncNet model path; proxy confidence used if missing')
    parser.add_argument('--lock_box_from_first', action='store_true',
                        help='Detect box on first frame and reuse for all frames to reduce jitter')
    parser.add_argument('--mask_feather', type=int, default=20,
                        help='Feather width (in pixels) for mouth mask postprocess to reduce seams')
    parser.add_argument('--bbox_ema_beta', type=float, default=0.6,
                        help='EMA smoothing factor for face detection boxes (0 disables)')
    parser.add_argument('--quad_ema_beta', type=float, default=0.5,
                        help='EMA smoothing factor for alignment quads (used in warping)')
    # Chunking and warmup controls
    parser.add_argument('--enable_chunking', action='store_true', default=True,
                        help='Automatically split long videos into chunks for stable sync')
    parser.add_argument('--chunk_duration', type=float, default=5.0,
                        help='Chunk duration in seconds (set 0 to disable)')
    parser.add_argument('--chunk_overlap', type=float, default=0.1,
                        help='Overlap in seconds between consecutive chunks')
    parser.add_argument('--chunk_crossfade', action='store_true', default=False,
                        help='Apply crossfade between chunk boundaries (re-encodes)')
    parser.add_argument('--chunk_temp_dir', type=str, default='parts',
                        help='Subfolder under temp/<tmp_dir> to store chunk files')
    parser.add_argument('--warmup_frames', type=int, default=2,
                        help='Run warmup frames per chunk before writing outputs')
    # Optical flow temporal smoothing controls
    parser.add_argument('--flow_temporal_smooth', action='store_true',
                        help='Enable optical-flow-guided temporal smoothing of the lip patch')
    parser.add_argument('--flow_blend_weight', type=float, default=0.4,
                        help='Blend weight for warped previous lip patch (0..1)')
    parser.add_argument('--flow_pyr_scale', type=float, default=0.5,
                        help='Farneback pyramid scale (0..1)')
    parser.add_argument('--flow_levels', type=int, default=3,
                        help='Farneback number of pyramid levels')
    parser.add_argument('--flow_winsize', type=int, default=21,
                        help='Farneback averaging window size (odd)')
    parser.add_argument('--flow_iters', type=int, default=3,
                        help='Farneback iterations per pyramid level')
    parser.add_argument('--flow_poly_n', type=int, default=5,
                        help='Farneback size of pixel neighborhood (odd)')
    parser.add_argument('--flow_poly_sigma', type=float, default=1.2,
                        help='Farneback standard deviation of Gaussian used')
    parser.add_argument('--latent_noise_scale', type=float, default=0.0,
                        help='Scale of latent noise injected into mel features (set 0 to disable)')
    # Optional full-frame stabilization using ffmpeg vidstab
    parser.add_argument('--pre_stabilize_video', action='store_true', default=False,
                        help='Apply ffmpeg vidstab stabilization pass on the input video before processing')
    parser.add_argument('--stabilize_shakiness', type=int, default=5,
                        help='vidstabdetect shakiness parameter (1-10). Higher handles larger shakes but is slower')
    parser.add_argument('--stabilize_smoothing', type=int, default=30,
                        help='vidstabtransform smoothing parameter controlling strength of stabilization')
    parser.add_argument('--stabilize_crop', type=str, default='keep',
                        help='vidstabtransform cropping behaviour: keep, content, or black')
    # (SyncNet gating args defined above to avoid duplicates)
    
    args = parser.parse_args()
    return args

exp_aus_dict = {        # AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r.
    'sad': torch.Tensor([[ 0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'angry':torch.Tensor([[0,     0,      0.3,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]]),
    'surprise': torch.Tensor([[0, 0,      0,      0.2,    0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]])
}

def mask_postprocess(mask, thres=20):
    # Clamp feather threshold to mask size
    h, w = mask.shape[:2]
    t = int(max(1, min(thres, min(h, w) // 4)))
    mask[:t, :] = 0; mask[-t:, :] = 0
    mask[:, :t] = 0; mask[:, -t:] = 0
    # Choose a valid odd Gaussian kernel <= min(h, w)
    base = max(3, min(101, (min(h, w) // 2) * 2 + 1))
    if base % 2 == 0:
        base = max(3, base - 1)
    sigma = max(1.0, base / 9.0)
    mask = cv2.GaussianBlur(mask, (base, base), sigma)
    mask = cv2.GaussianBlur(mask, (base, base), sigma)
    return mask.astype(np.float32)

def trans_image(image):
    image = TF.resize(
        image, size=256, interpolation=Image.BICUBIC)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return image

def obtain_seq_index(index, num_frames):
    seq = list(range(index-13, index+13))
    seq = [ min(max(item, 0), num_frames-1) for item in seq ]
    return seq

def transform_semantic(semantic, frame_index, crop_norm_ratio=None):
    index = obtain_seq_index(frame_index, semantic.shape[0])
    
    coeff_3dmm = semantic[index,...]
    ex_coeff = coeff_3dmm[:,80:144] #expression # 64
    angles = coeff_3dmm[:,224:227] #euler angles for pose
    translation = coeff_3dmm[:,254:257] #translation
    crop = coeff_3dmm[:,259:262] #crop param

    if crop_norm_ratio:
        crop[:, -3] = crop[:, -3] * crop_norm_ratio

    coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
    return torch.Tensor(coeff_3dmm).permute(1,0)   

def find_crop_norm_ratio(source_coeff, target_coeffs):
    alpha = 0.3
    exp_diff = np.mean(np.abs(target_coeffs[:,80:144] - source_coeff[:,80:144]), 1) # mean different exp
    angle_diff = np.mean(np.abs(target_coeffs[:,224:227] - source_coeff[:,224:227]), 1) # mean different angle
    index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)  # find the smallerest index
    crop_norm_ratio = source_coeff[:,-3] / target_coeffs[index:index+1, -3]
    return crop_norm_ratio

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, args, jaw_correction=False, detector=None):
    if detector == None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                flip_input=False, device=device)

    # If user provided a fixed bounding box, respect it for all frames
    if hasattr(args, 'box') and args.box is not None and len(args.box) == 4 and all([v != -1 for v in args.box]):
        top, bottom, left, right = args.box
        pady1, pady2, padx1, padx2 = args.pads if jaw_correction else (0,30,0,0)
        results = []
        for image in images:
            h, w = image.shape[:2]
            y1 = max(0, top - pady1)
            y2 = min(h, bottom + pady2)
            x1 = max(0, left - padx1)
            x2 = min(w, right + padx2)
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        del detector
        torch.cuda.empty_cache()
        return results

    # Optionally lock the detection box from the first frame for all subsequent frames
    if getattr(args, 'lock_box_from_first', False):
        pady1, pady2, padx1, padx2 = args.pads if jaw_correction else (0,30,0,0)
        first = images[0]
        rect0 = detector.get_detections_for_batch(np.array([first]))[0]
        if rect0 is None:
            h, w = first.shape[:2]
            rect0 = [0, 0, w, h]
        y1 = max(0, rect0[1] - pady1)
        y2 = min(first.shape[0], rect0[3] + pady2)
        x1 = max(0, rect0[0] - padx1)
        x2 = min(first.shape[1], rect0[2] + padx2)
        boxes = np.array([[x1, y1, x2, y2] for _ in images])
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image in images]
        del detector
        torch.cuda.empty_cache()
        return results

    batch_size = args.face_det_batch_size    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size),desc='FaceDet:'):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads if jaw_correction else (0,30,0,0)
    for rect, image in zip(predictions, images):
        if rect is None:
            # Fallback to full frame if detection fails, and continue
            h, w = image.shape[:2]
            rect = [0, 0, w, h]

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        smooth_window = max(1, args.temporal_smooth_window)
        boxes = get_smoothened_boxes(boxes, T=smooth_window)
        # Apply EMA smoothing across frames for additional stability
        beta = float(getattr(args, 'bbox_ema_beta', 0.0))
        if beta > 0.0 and len(boxes) > 1:
            ema_boxes = [boxes[0].astype(np.float32)]
            prev = ema_boxes[0]
            for b in boxes[1:]:
                b = b.astype(np.float32)
                prev = beta * prev + (1.0 - beta) * b
                ema_boxes.append(prev)
            boxes = np.stack(ema_boxes, axis=0)
    # Convert to safe integer boxes per-frame and clamp to image bounds
    int_boxes = []
    for image, (x1, y1, x2, y2) in zip(images, boxes):
        h, w = image.shape[:2]
        x1i, y1i, x2i, y2i = [int(round(v)) for v in (x1, y1, x2, y2)]
        x1i = max(0, min(x1i, w - 1))
        x2i = max(0, min(x2i, w))
        y1i = max(0, min(y1i, h - 1))
        y2i = max(0, min(y2i, h))
        # Ensure valid ordering and non-empty area
        if x2i <= x1i:
            if x1i + 1 < w:
                x2i = x1i + 1
            else:
                x1i = max(0, w - 2)
                x2i = w - 1
        if y2i <= y1i:
            if y1i + 1 < h:
                y2i = y1i + 1
            else:
                y1i = max(0, h - 2)
                y2i = h - 1
        int_boxes.append((x1i, y1i, x2i, y2i))
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, int_boxes)]

    del detector
    torch.cuda.empty_cache()
    return results 

def _load(checkpoint_path, device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtract upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        gm = gm[:,:,np.newaxis]
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        # --- Fix: handle shape mismatches between pyramid levels ---
        if ls_.shape[:2] != LS[i].shape[:2]:
            LS[i] = cv2.resize(LS[i], (ls_.shape[1], ls_.shape[0]))
            print(f"[Fix] Resized LS[{i}] from pyramid to match {ls_.shape}")
        ls_ = cv2.add(ls_, LS[i])
    return ls_

def load_model(args, device):
    D_Net = load_DNet(args).to(device)
    model = load_network(args).to(device)
    return D_Net, model

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new

def load_face3d_net(ckpt_path, device):
    net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)    
    net_recon.load_state_dict(checkpoint['net_recon'])
    net_recon.eval()
    return net_recon
