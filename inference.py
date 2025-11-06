import numpy as np
import cv2, os, sys, subprocess, platform, torch
import math
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, 'third_part')
sys.path.insert(0, 'third_part/GPEN')
sys.path.insert(0, 'third_part/GFPGAN')

# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict, mask_postprocess
import warnings
warnings.filterwarnings("ignore")

args = options()

def _ffprobe_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nk=1:nw=1', path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return float(out.decode().strip())
    except Exception as e:
        print(f"[Warn] ffprobe failed for {path}: {e}")
        return None

def _split_into_chunks(face_path, audio_path, out_dir, fps, chunk_duration, overlap, enforce_sr=16000, crossfade=False):
    os.makedirs(out_dir, exist_ok=True)
    video_dur = _ffprobe_duration(face_path) or 0
    audio_dur = _ffprobe_duration(audio_path) or 0
    total_dur = min(video_dur, audio_dur) if (video_dur > 0 and audio_dur > 0) else max(video_dur, audio_dur)
    if total_dur == 0:
        raise RuntimeError('Could not determine media duration')
    eff_overlap = overlap if crossfade else 0.0
    step = max(0.0, chunk_duration - eff_overlap) if chunk_duration > 0 else total_dur
    starts = []
    t = 0.0
    while t < total_dur:
        starts.append(t)
        t += step if step > 0 else total_dur
    chunk_paths = []
    for idx, start in enumerate(starts):
        dur = min(chunk_duration if chunk_duration > 0 else total_dur, total_dur - start)
        v_out = os.path.join(out_dir, f'video_{idx+1:03d}.mp4')
        a_out = os.path.join(out_dir, f'audio_{idx+1:03d}.wav')
        # Re-encode video chunk to enforce exact boundaries & fps
        vcmd = ['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{start:.3f}', '-i', face_path, '-t', f'{dur:.3f}',
                '-r', f'{fps:.3f}', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', v_out]
        acmd = ['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{start:.3f}', '-i', audio_path, '-t', f'{dur:.3f}',
                '-ac', '1', '-ar', str(enforce_sr), a_out]
        try:
            subprocess.check_call(vcmd, shell=False)
            subprocess.check_call(acmd, shell=False)
        except subprocess.CalledProcessError as e:
            print(f"[Warn] Failed to split chunk {idx+1}: {e}")
            continue
        # Validate audio sample rate
        try:
            sr = _probe_audio_sr(a_out)
            if sr != enforce_sr:
                print(f"[Warn] Fixing sample rate for {a_out} (got {sr}); forcing {enforce_sr}")
                fixcmd = ['ffmpeg', '-loglevel', 'error', '-y', '-i', a_out, '-ac', '1', '-ar', str(enforce_sr), a_out]
                subprocess.check_call(fixcmd, shell=False)
        except Exception as e:
            print(f"[Warn] Could not validate audio sr for {a_out}: {e}")
        chunk_paths.append((v_out, a_out))
    return chunk_paths

def _probe_audio_sr(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'default=nk=1:nw=1', path]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return int(out.decode().strip())

def _concat_media(list_file, output_path, stream='video', reencode=False, drop_audio=False):
    # stream: 'video' or 'audio'
    if stream == 'video':
        if reencode:
            cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c:v', 'libx264', '-pix_fmt', 'yuv420p']
        else:
            cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy']
        if drop_audio:
            cmd += ['-an']
        cmd += [output_path]
    else:
        cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path]
    subprocess.check_call(cmd, shell=False)

def _xfade_concatenate(videos, out_path, overlap):
    # Simple pairwise xfade across N segments
    if len(videos) == 1:
        return videos[0]
    cur = videos[0]
    for i in range(1, len(videos)):
        nxt = videos[i]
        tmp = out_path.replace('.mp4', f'_xf{i:03d}.mp4')
        cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-i', cur, '-i', nxt,
               '-filter_complex', f"[0:v][1:v]xfade=transition=fade:duration={overlap:.3f}:offset=PTS-STARTPTS[v];[0:a][1:a]acrossfade=d={overlap:.3f}[a]",
               '-map', '[v]', '-map', '[a]', tmp]
        subprocess.check_call(cmd, shell=False)
        cur = tmp
    os.replace(cur, out_path)
    return out_path

def main():    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[Info] Using {} for inference.'.format(device))
    os.makedirs(os.path.join('temp', args.tmp_dir), exist_ok=True)

    enhancer = FaceEnhancement(base_dir='checkpoints', size=args.gpen_size, model=args.gpen_model, use_sr=False, \
                               sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
    restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
                        channel_multiplier=2, bg_upsampler=None)

    base_name = args.face.split('/')[-1]
    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True
        # Enforce one-shot for static images
        if not args.one_shot:
            print('[Info] Static image detected; enabling one-shot mode.')
        args.one_shot = True
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        # Disable one-shot for videos
        if args.one_shot:
            print('[Info] Video detected; disabling one-shot mode (only for static images).')
        args.one_shot = False
        # Chunked processing for long videos
        if args.enable_chunking and args.chunk_duration and args.chunk_duration > 0:
            try:
                video_dur = _ffprobe_duration(args.face) or 0
                if video_dur > args.chunk_duration + 0.5:
                    # Probe fps
                    cap_tmp = cv2.VideoCapture(args.face)
                    fps_probe = cap_tmp.get(cv2.CAP_PROP_FPS)
                    cap_tmp.release()
                    if not fps_probe or math.isclose(fps_probe, 0.0) or np.isnan(fps_probe):
                        fps_probe = args.fps
                    parts_dir = os.path.join('temp', args.tmp_dir, args.chunk_temp_dir)
                    out_parts_dir = os.path.join('temp', args.tmp_dir, 'out_parts')
                    os.makedirs(parts_dir, exist_ok=True)
                    os.makedirs(out_parts_dir, exist_ok=True)
                    print(f"[Chunk] Splitting into ~{args.chunk_duration}s chunks with overlap={args.chunk_overlap}s")
                    chunks = _split_into_chunks(args.face, args.audio, parts_dir, fps_probe, args.chunk_duration, args.chunk_overlap, enforce_sr=16000, crossfade=args.chunk_crossfade)
                    # Minimal required seconds for a mel window
                    from utils import audio as _audio
                    min_sec = (args.mel_step_size * _audio.hp.hop_size) / _audio.hp.sample_rate
                    successes = []
                    for ci, (vpath, apath) in enumerate(chunks, start=1):
                        try:
                            # Validate length
                            c_dur = _ffprobe_duration(vpath) or 0
                            if c_dur < min_sec:
                                print(f"[Warn] Skipping chunk {ci:03d} too short: {c_dur:.3f}s < {min_sec:.3f}s")
                                continue
                            out_path = os.path.join(out_parts_dir, f'output_{ci:03d}.mp4')
                            cmd = [sys.executable, os.path.abspath(__file__), '--face', vpath, '--audio', apath, '--outfile', out_path, '--fps', f'{fps_probe:.3f}', '--tmp_dir', f"{args.tmp_dir}/chunk_{ci:03d}", '--chunk_duration', '0']
                            # Propagate useful flags
                            if args.audio_preprocess: cmd.append('--audio_preprocess')
                            if args.audio_normalize: cmd.append('--audio_normalize')
                            if args.audio_denoise: cmd.append('--audio_denoise')
                            if args.lip_temporal_smooth: cmd.append('--lip_temporal_smooth')
                            # Always re-preprocess per chunk to avoid stale landmark/cached files mismatch
                            cmd.append('--re_preprocess')
                            try:
                                subprocess.check_call(cmd, shell=False)
                                successes.append(out_path)
                            except subprocess.CalledProcessError as e:
                                print(f"[Warn] Chunk {ci:03d} failed: {e}")
                                continue
                        except Exception as e:
                            print(f"[Warn] Exception on chunk {ci:03d}: {e}")
                            continue
                    if len(successes) == 0:
                        raise RuntimeError('No chunk processed successfully')
                    final_out = args.outfile
                    if args.chunk_crossfade:
                        # Build crossfaded concatenation including audio
                        concat_video = os.path.join('temp', args.tmp_dir, 'concat_video.mp4')
                        _xfade_concatenate(successes, concat_video, max(0.0, args.chunk_overlap))
                        # Just move to final (already has audio)
                        if not os.path.isdir(os.path.dirname(final_out)):
                            os.makedirs(os.path.dirname(final_out), exist_ok=True)
                        if platform.system() == 'Windows':
                            subprocess.check_call(['ffmpeg', '-loglevel', 'error', '-y', '-i', concat_video, '-c', 'copy', final_out], shell=False)
                        else:
                            subprocess.check_call(f'ffmpeg -loglevel error -y -i "{concat_video}" -c copy "{final_out}"', shell=True)
                    else:
                        # Concatenate videos (drop audio)
                        concat_v_list = os.path.join('temp', args.tmp_dir, 'out_parts', 'concat_v.txt')
                        with open(concat_v_list, 'w', encoding='utf-8') as f:
                            for p in successes:
                                _safe = os.path.abspath(p).replace('\\', '/')
                                f.write(f"file '{_safe}'\n")
                        concat_video = os.path.join('temp', args.tmp_dir, 'concat_video.mp4')
                        _concat_media(concat_v_list, concat_video, stream='video', reencode=False, drop_audio=True)
                        # Concatenate audios
                        chunk_audios = [os.path.join(parts_dir, f) for f in sorted(os.listdir(parts_dir)) if f.lower().startswith('audio_') and f.lower().endswith('.wav')]
                        concat_a_list = os.path.join('temp', args.tmp_dir, 'out_parts', 'concat_a.txt')
                        with open(concat_a_list, 'w', encoding='utf-8') as f:
                            for p in chunk_audios:
                                _safe = os.path.abspath(p).replace('\\', '/')
                                f.write(f"file '{_safe}'\n")
                        concat_audio = os.path.join('temp', args.tmp_dir, 'concat_audio.wav')
                        _concat_media(concat_a_list, concat_audio, stream='audio', reencode=False)
                        # Merge A/V with -shortest
                        if not os.path.isdir(os.path.dirname(final_out)):
                            os.makedirs(os.path.dirname(final_out), exist_ok=True)
                        if platform.system() == 'Windows':
                            mcmd = ['ffmpeg', '-loglevel', 'error', '-y', '-i', concat_video, '-i', concat_audio, '-shortest', '-c:v', 'copy', final_out]
                            subprocess.call(mcmd, shell=False)
                        else:
                            mcmd = f'ffmpeg -loglevel error -y -i "{concat_video}" -i "{concat_audio}" -shortest -c:v copy "{final_out}"'
                            subprocess.call(mcmd, shell=True)
                    print('outfile:', final_out)
                    return
            except Exception as e:
                print(f"[Warn] Chunked processing fallback due to error: {e}")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print ("[Step 0] Number of frames available for inference: "+str(len(full_frames)))
    # face detection & cropping, cropping the first frame as the style of FFHQ
    croper = Croper('checkpoints/shape_predictor_68_face_landmarks.dat')
    full_frames_RGB = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]
    full_frames_RGB, crop, quad = croper.crop(full_frames_RGB, xsize=512)

    clx, cly, crx, cry = crop
    lx, ly, rx, ry = quad
    lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
    oy1, oy2, ox1, ox2 = cly+ly, min(cly+ry, full_frames[0].shape[0]), clx+lx, min(clx+rx, full_frames[0].shape[1])
    # original_size = (ox2 - ox1, oy2 - oy1)
    frames_pil = [Image.fromarray(cv2.resize(frame,(256,256))) for frame in full_frames_RGB]

    # get the landmark according to the detected face.
    if not os.path.isfile('temp/'+base_name+'_landmarks.txt') or args.re_preprocess:
        print('[Step 1] Landmarks Extraction in Video.')
        kp_extractor = KeypointExtractor()
        lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
    else:
        print('[Step 1] Using saved landmarks.')
        lm_path = 'temp/'+base_name+'_landmarks.txt'
        try:
            lm_raw = np.loadtxt(lm_path).astype(np.float32)
        except Exception as e:
            print(f"[Step 1] Failed to load saved landmarks ({e}); re-extracting.")
            kp_extractor = KeypointExtractor()
            lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
        else:
            total_vals = lm_raw.size
            denom = 2 * len(full_frames)
            if denom > 0 and total_vals % denom == 0:
                k = total_vals // denom
                try:
                    lm = lm_raw.reshape([len(full_frames), k, 2])
                except Exception:
                    print('[Step 1] Saved landmarks shape mismatch; re-extracting.')
                    kp_extractor = KeypointExtractor()
                    lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
            else:
                print('[Step 1] Saved landmarks length does not match frames; re-extracting.')
                kp_extractor = KeypointExtractor()
                lm = kp_extractor.extract_keypoint(frames_pil, './temp/'+base_name+'_landmarks.txt')
       
    if not os.path.isfile('temp/'+base_name+'_coeffs.npy') or args.exp_img is not None or args.re_preprocess:
        net_recon = load_face3d_net(args.face3d_net_path, device)
        lm3d_std = load_lm3d('checkpoints/BFM')

        video_coeffs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 2] 3DMM Extraction In Video:"):
            frame = frames_pil[idx]
            W, H = frame.size
            lm_idx = lm[idx].reshape([-1, 2])
            if np.mean(lm_idx) == -1:
                lm_idx = (lm3d_std[:, :2]+1) / 2.
                lm_idx = np.concatenate([lm_idx[:, :1] * W, lm_idx[:, 1:2] * H], 1)
            else:
                lm_idx[:, -1] = H - 1 - lm_idx[:, -1]

            trans_params, im_idx, lm_idx, _ = align_img(frame, lm_idx, lm3d_std)
            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_idx_tensor = torch.tensor(np.array(im_idx)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0) 
            with torch.no_grad():
                coeffs = split_coeff(net_recon(im_idx_tensor))

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            pred_coeff = np.concatenate([pred_coeff['id'], pred_coeff['exp'], pred_coeff['tex'], pred_coeff['angle'],\
                                         pred_coeff['gamma'], pred_coeff['trans'], trans_params[None]], 1)
            video_coeffs.append(pred_coeff)
        semantic_npy = np.array(video_coeffs)[:,0]
        np.save('temp/'+base_name+'_coeffs.npy', semantic_npy)
    else:
        print('[Step 2] Using saved coeffs.')
        semantic_npy = np.load('temp/'+base_name+'_coeffs.npy').astype(np.float32)

    # generate the 3dmm coeff from a single image
    if args.exp_img is not None and ('.png' in args.exp_img or '.jpg' in args.exp_img):
        print('extract the exp from',args.exp_img)
        exp_pil = Image.open(args.exp_img).convert('RGB')
        lm3d_std = load_lm3d('third_part/face3d/BFM')
        
        W, H = exp_pil.size
        kp_extractor = KeypointExtractor()
        lm_exp = kp_extractor.extract_keypoint([exp_pil], 'temp/'+base_name+'_temp.txt')[0]
        if np.mean(lm_exp) == -1:
            lm_exp = (lm3d_std[:, :2] + 1) / 2.
            lm_exp = np.concatenate(
                [lm_exp[:, :1] * W, lm_exp[:, 1:2] * H], 1)
        else:
            lm_exp[:, -1] = H - 1 - lm_exp[:, -1]

        trans_params, im_exp, lm_exp, _ = align_img(exp_pil, lm_exp, lm3d_std)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_exp_tensor = torch.tensor(np.array(im_exp)/255., dtype=torch.float32).permute(2, 0, 1).to(device).unsqueeze(0)
        # Ensure face3d net is loaded when deriving expression from an image
        if 'net_recon' not in locals():
            net_recon = load_face3d_net(args.face3d_net_path, device)
        with torch.no_grad():
            expression = split_coeff(net_recon(im_exp_tensor))['exp'][0]
        del net_recon
    elif isinstance(args.exp_img, str) and args.exp_img.lower() == 'neutral':
        print('using neutral expression')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]
    elif isinstance(args.exp_img, str) and args.exp_img.lower() == 'smile':
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_mouth'])[0]
    else:
        print('using neutral expression (default)')
        expression = torch.tensor(loadmat('checkpoints/expression.mat')['expression_center'])[0]

    # load DNet, model(LNet and ENet)
    D_Net, model = load_model(args, device)

    if not os.path.isfile('temp/'+base_name+'_stablized.npy') or args.re_preprocess:
        imgs = []
        for idx in tqdm(range(len(frames_pil)), desc="[Step 3] Stabilize the expression In Video:"):
            if args.one_shot:
                source_img = trans_image(frames_pil[0]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[0:1]
            else:
                source_img = trans_image(frames_pil[idx]).unsqueeze(0).to(device)
                semantic_source_numpy = semantic_npy[idx:idx+1]
            ratio = find_crop_norm_ratio(semantic_source_numpy, semantic_npy)
            coeff = transform_semantic(semantic_npy, idx, ratio).unsqueeze(0).to(device)
        
            # hacking the new expression
            coeff[:, :64, :] = expression[None, :64, None].to(device) 
            with torch.no_grad():
                output = D_Net(source_img, coeff)
            img_stablized = np.uint8((output['fake_image'].squeeze(0).permute(1,2,0).cpu().clamp_(-1, 1).numpy() + 1 )/2. * 255)
            imgs.append(cv2.cvtColor(img_stablized,cv2.COLOR_RGB2BGR)) 
        np.save('temp/'+base_name+'_stablized.npy',imgs)
        del D_Net
    else:
        print('[Step 3] Using saved stabilized video.')
        imgs = np.load('temp/'+base_name+'_stablized.npy')
    torch.cuda.empty_cache()

    # Ensure 16kHz mono WAV and clean levels prior to mel extraction
    processed_audio_path = 'temp/{}/audio_proc.wav'.format(args.tmp_dir)
    if args.audio_preprocess or (not args.audio.endswith('.wav')):
        afilters = []
        if args.audio_denoise:
            afilters.append('afftdn')
        if args.audio_normalize:
            afilters.append('dynaudnorm=f=150:g=15')
        af_clause = ' -af "{}"'.format(','.join(afilters)) if len(afilters) > 0 else ''
        command = 'ffmpeg -loglevel error -y -i "{}" -ac 1 -ar {}{} "{}"'.format(
            args.audio, args.audio_target_sr, af_clause, processed_audio_path)
        subprocess.call(command, shell=True)
        args.audio = processed_audio_path
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        if args.audio_eps_on_nan:
            # Add small epsilon noise (useful for some TTS outputs) and retry once
            wav = wav + (1e-6 * np.random.randn(len(wav))).astype(wav.dtype)
            mel = audio.melspectrogram(wav)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Consider denoising/normalizing or adding epsilon noise and try again')

    # Derive mel frames-per-second from audio hyperparams to stay in sync with hop_size
    mel_frames_per_second = audio.hp.sample_rate / audio.hp.hop_size
    mel_step_size = args.mel_step_size
    mel_idx_multiplier = (mel_frames_per_second / fps) * args.mel_multiplier_scale
    # Build one mel chunk per video frame (1:1 alignment)
    def build_mel_chunks(offset_frames):
        chunks = []
        T = mel.shape[1]
        for frame_idx in range(len(full_frames)):
            start_idx = int(round(frame_idx * mel_idx_multiplier)) + int(offset_frames)
            end_idx = start_idx + mel_step_size
            if start_idx >= T:
                m = np.zeros((mel.shape[0], mel_step_size), dtype=mel.dtype)
            elif end_idx > T:
                pad = end_idx - T
                m = np.pad(mel[:, start_idx:], ((0, 0), (0, pad)), mode='constant')
            elif start_idx < 0:
                pad = -start_idx
                take = mel[:, :max(0, end_idx)]
                m = np.pad(take, ((0,0),(pad,0)), mode='constant')
                if m.shape[1] < mel_step_size:
                    m = np.pad(m, ((0,0),(0, mel_step_size - m.shape[1])), mode='constant')
            else:
                m = mel[:, start_idx:end_idx]
            chunks.append(m)
        return chunks

    base_offset = int(args.mel_offset_frames)
    mel_chunks = build_mel_chunks(base_offset)

    # Optional: auto calibration of global offset by correlating mouth openness with mel energy
    if (not args.static) and getattr(args, 'auto_sync_calibrate', False):
        try:
            # Compute mouth openness per frame from 68-point landmarks
            # Using pairs: (62,66), (63,67), (61,65) in 0-based indices
            def mouth_open_metric(lms):
                if lms is None or len(lms) < 68:
                    return 0.0
                p = lms
                pairs = [(62,66), (63,67), (61,65)]
                vals = []
                for a,b in pairs:
                    ya, yb = float(p[a][1]), float(p[b][1])
                    vals.append(abs(yb - ya))
                return float(np.mean(vals)) if len(vals) > 0 else 0.0

            mouth_series = np.array([mouth_open_metric(land) for land in lm[:len(full_frames)]], dtype=np.float32)
            energy_series = np.array([float((mc * mc).mean()) for mc in mel_chunks], dtype=np.float32)
            # Normalize
            def nz_norm(x):
                x = x - x.mean()
                s = x.std()
                return x / (s if s > 1e-6 else 1.0)
            m_norm = nz_norm(mouth_series)
            e_norm = nz_norm(energy_series)
            max_lag = int(getattr(args, 'calibrate_window', 6))
            best_corr, best_off = -1e9, 0
            for off in range(-max_lag, max_lag+1):
                if off >= 0:
                    mn = m_norm[off:]
                    en = e_norm[:len(m_norm)-off]
                else:
                    mn = m_norm[:off]
                    en = e_norm[-off:]
                L = min(len(mn), len(en))
                if L < 10:
                    continue
                c = float(np.corrcoef(mn[:L], en[:L])[0,1])
                if c > best_corr:
                    best_corr, best_off = c, off
            if best_off != 0:
                print(f"[Calibrate] Best offset {best_off} frames (corr={best_corr:.3f}). Rebuilding mel chunks.")
                base_offset += best_off
                mel_chunks = build_mel_chunks(base_offset)
        except Exception as e:
            print(f"[Calibrate] Skipped due to error: {e}")

    print(f"✅ Built {len(mel_chunks)} mel chunks for {len(full_frames)} frames (multiplier={mel_idx_multiplier:.3f}, offset_frames={base_offset})")
    print("[Step 4] Load audio; Length of mel chunks: {}".format(len(mel_chunks)))
    
    imgs_enhanced = []
    for idx in tqdm(range(len(imgs)), desc='[Step 5] Reference Enhancement'):
        img = imgs[idx]
        pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
        imgs_enhanced.append(pred)
    gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/{}/result.mp4'.format(args.tmp_dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    
    if args.up_face != 'original':
        instance = GANimationModel()
        instance.initialize()
        instance.setup()

    kp_extractor = KeypointExtractor()
    prev_lip_patch = None
    prev_face_gray = None
    temporal_buffer = []
    temporal_window = max(1, int(getattr(args, 'lip_temporal_window', 5)))
    lip_mode = str(getattr(args, 'lip_temporal_mode', 'ema')).lower()
    # Optional warmup run to stabilize first frames
    if getattr(args, 'warmup_frames', 0) and args.warmup_frames > 0:
        try:
            warm_item = next(gen)
            img_batch, mel_batch, frames, coords, img_original, f_frames = warm_item
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255.
            with torch.no_grad():
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                _ = model(mel_batch, img_batch, reference)
        except StopIteration:
            pass
        # Recreate generator so we don't lose the first batch
        gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames, None, (oy1,oy2,ox1,ox2))
    for i, (img_batch, mel_batch, frames, coords, img_original, f_frames) in enumerate(tqdm(gen, desc='[Step 6] Lip Synthesis:', total=int(np.ceil(float(len(mel_chunks)) / args.LNet_batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        img_original = torch.FloatTensor(np.transpose(img_original, (0, 3, 1, 2))).to(device)/255. # BGR -> RGB
        
        # Validate mel batches to ensure minimum spatial dimensions
        valid_indices = []
        for bi in range(mel_batch.shape[0]):
            sample = mel_batch[bi]
            if sample.shape[-1] < 3 or sample.shape[-2] < 3:
                print(f"⚠️ Skipping invalid mel chunk shape: {tuple(sample.shape)}")
            else:
                valid_indices.append(bi)

        if len(valid_indices) == 0:
            raise ValueError("❌ All mel chunks were invalid. Check audio length or hop_size.")

        if len(valid_indices) != mel_batch.shape[0]:
            mel_batch = mel_batch[valid_indices]
            img_batch = img_batch[valid_indices]
            img_original = img_original[valid_indices]
            # Also filter associated metadata to keep alignment
            frames = [frames[j] for j in valid_indices]
            coords = [coords[j] for j in valid_indices]
            f_frames = [f_frames[j] for j in valid_indices]

        print(f"✅ Proceeding with {mel_batch.shape[0]} valid mel chunks of shape {tuple(mel_batch.shape[-2:])}")

        with torch.no_grad():
            incomplete, reference = torch.split(img_batch, 3, dim=1) 
            pred, low_res = model(mel_batch, img_batch, reference)
            pred = torch.clamp(pred, 0, 1)

            if args.up_face in ['sad', 'angry', 'surprise']:
                tar_aus = exp_aus_dict[args.up_face]
            else:
                pass
            
            if args.up_face == 'original':
                cur_gen_faces = img_original
            else:
                test_batch = {'src_img': torch.nn.functional.interpolate((img_original * 2 - 1), size=(128, 128), mode='bilinear'), 
                              'tar_aus': tar_aus.repeat(len(incomplete), 1)}
                instance.feed_batch(test_batch)
                instance.forward()
                cur_gen_faces = torch.nn.functional.interpolate(
                    instance.fake_img / 2. + 0.5,
                    size=(args.img_size, args.img_size),
                    mode='bilinear'
                )
                
            if args.without_rl1 is not False:
                incomplete, reference = torch.split(img_batch, 3, dim=1)
                mask = torch.where(incomplete==0, torch.ones_like(incomplete), torch.zeros_like(incomplete)) 
                pred = pred * mask + cur_gen_faces * (1 - mask) 
        
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        if args.lip_temporal_smooth:
            base_weight = min(max(args.temporal_blend_weight, 0.0), 1.0)
            adaptive = getattr(args, 'temporal_blend_auto', False)
            k = float(getattr(args, 'temporal_blend_k', 5.0))
            smoothed = []
            # Compute per-sample energy from current mel_batch
            try:
                mel_energy = (mel_batch ** 2).mean(dim=[1,2,3]).detach().cpu().numpy()
                e_min, e_max = float(mel_energy.min()), float(mel_energy.max())
                denom = (e_max - e_min) if (e_max - e_min) > 1e-6 else 1.0
                mel_energy_n = (mel_energy - e_min) / denom
            except Exception:
                adaptive = False
                mel_energy_n = None
            for idx, p in enumerate(pred):
                out_patch = p
                # EMA smoothing using previous frame
                if lip_mode in ['ema', 'both']:
                    if prev_lip_patch is not None:
                        if adaptive and mel_energy_n is not None:
                            w = base_weight * float(np.exp(-k * mel_energy_n[idx]))
                        else:
                            w = base_weight
                        w = float(min(max(w, 0.0), 1.0))
                        out_patch = (1.0 - w) * p + w * prev_lip_patch
                # Optical-flow guided warping of previous patch to current geometry
                if getattr(args, 'flow_temporal_smooth', False) and prev_lip_patch is not None:
                    try:
                        # Prepare current face gray image from original face patch
                        cur_face_tensor = img_original[idx]  # shape: 3xH xW (RGB, 0..1)
                        cur_face = (cur_face_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
                        cur_gray = cv2.cvtColor(cur_face, cv2.COLOR_RGB2GRAY)
                        if prev_face_gray is not None and prev_face_gray.shape == cur_gray.shape:
                            flow = cv2.calcOpticalFlowFarneback(
                                prev_face_gray, cur_gray, None,
                                float(getattr(args, 'flow_pyr_scale', 0.5)),
                                int(getattr(args, 'flow_levels', 3)),
                                int(getattr(args, 'flow_winsize', 21)),
                                int(getattr(args, 'flow_iters', 3)),
                                int(getattr(args, 'flow_poly_n', 5)),
                                float(getattr(args, 'flow_poly_sigma', 1.2)),
                                0
                            )
                            h, w = cur_gray.shape
                            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                            map_x = (grid_x + flow[..., 0]).astype(np.float32)
                            map_y = (grid_y + flow[..., 1]).astype(np.float32)
                            warped_prev = cv2.remap(prev_lip_patch.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                            wf = float(getattr(args, 'flow_blend_weight', 0.4))
                            wf = max(0.0, min(1.0, wf))
                            out_patch = (1.0 - wf) * out_patch + wf * warped_prev
                        # update prev_face_gray for next iteration
                        prev_face_gray = cur_gray
                    except Exception:
                        # if any failure in flow, just skip silently
                        pass
                # Median over a small temporal window to remove flicker
                if lip_mode in ['median', 'both']:
                    temporal_buffer.append(out_patch)
                    if len(temporal_buffer) > temporal_window:
                        temporal_buffer.pop(0)
                    stack = np.stack(temporal_buffer, axis=0)
                    out_patch = np.median(stack, axis=0)
                smoothed.append(out_patch)
                prev_lip_patch = out_patch
            pred = np.stack(smoothed, axis=0)

        torch.cuda.empty_cache()
        for p, f, xf, c in zip(pred, frames, f_frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            ff = xf.copy() 
            ff[y1:y2, x1:x2] = p
            
            # month region enhancement by GFPGAN
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                ff, has_aligned=False, only_center_face=True, paste_back=True)
                # 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
            mm = [0,   0,   0,   0,   0,   0,   0,   0,   0,  0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            mouse_mask = np.zeros_like(restored_img)
            tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
            tmp_mask = mask_postprocess(tmp_mask, thres=getattr(args, 'mask_feather', 20))
            mouse_mask[y1:y2, x1:x2]= cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

            height, width = ff.shape[:2]
            # Ensure shapes align for blending at native resolution
            restored_img = restored_img if restored_img.shape[:2] == (height, width) else cv2.resize(restored_img, (width, height))
            ff = ff if ff.shape[:2] == (height, width) else cv2.resize(ff, (width, height))
            full_mask = np.float32(mouse_mask)
            if full_mask.shape[:2] != (height, width):
                full_mask = cv2.resize(full_mask, (width, height))
            # --- Shape alignment fix for Laplacian blending ---
            if restored_img.shape != ff.shape:
                ff = cv2.resize(ff, (restored_img.shape[1], restored_img.shape[0]))
                print(f"[Fix] Resized ff to {ff.shape} to match restored_img")
            if full_mask.shape[:2] != restored_img.shape[:2]:
                full_mask = cv2.resize(full_mask, (restored_img.shape[1], restored_img.shape[0]))
                print(f"[Fix] Resized full_mask to {full_mask.shape} to match restored_img")
            print("Shapes before blending:", restored_img.shape, ff.shape, full_mask.shape)
            img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 12)
            img = np.clip(img, 0, 255)
            if img.shape[:2] != (height, width):
                pp = np.uint8(cv2.resize(img, (width, height)))
            else:
                pp = np.uint8(img)

            pp, orig_faces, enhanced_faces = enhancer.process(pp, xf, bbox=c, face_enhance=True, possion_blending=False)
            out.write(pp)
    out.release()
    
    if not os.path.isdir(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    if platform.system() == 'Windows':
        command = ['ffmpeg', '-loglevel', 'error', '-y', '-i', args.audio, '-i', 'temp/{}/result.mp4'.format(args.tmp_dir), '-shortest', '-c:v', 'copy', args.outfile]
        subprocess.call(command, shell=False)
    else:
        command = 'ffmpeg -loglevel error -y -i {} -i {} -shortest -c:v copy {}'.format(args.audio, 'temp/{}/result.mp4'.format(args.tmp_dir), args.outfile)
        subprocess.call(command, shell=True)
    print('outfile:', args.outfile)


# frames:256x256, full_frames: original size
def datagen(frames, mels, full_frames, frames_pil, cox):
    img_batch, mel_batch, frame_batch, coords_batch, ref_batch, full_frame_batch = [], [], [], [], [], []
    base_name = args.face.split('/')[-1]
    refs = []
    image_size = 256 

    # original frames
    kp_extractor = KeypointExtractor()
    fr_pil = [Image.fromarray(frame) for frame in frames]
    lms = kp_extractor.extract_keypoint(fr_pil, 'temp/'+base_name+'x12_landmarks.txt')
    frames_pil = [ (lm, frame) for frame,lm in zip(fr_pil, lms)] # frames is the croped version of modified face
    crops, orig_images, quads  = crop_faces(image_size, frames_pil, scale=1.0, use_fa=True)
    # Guard against invalid (NaN/inf) quads by reusing last valid or falling back to identity
    clean_quads = []
    identity_quad = np.array([[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]], dtype=np.float32)
    for q in quads:
        if q is None or (not np.isfinite(q).all()):
            q = clean_quads[-1] if len(clean_quads) > 0 else identity_quad
        clean_quads.append(q)
    # EMA smooth the alignment quads to reduce temporal jitter in warping
    beta_q = float(getattr(args, 'quad_ema_beta', 0.0))
    if beta_q > 0.0 and len(clean_quads) > 1:
        sm_quads = []
        prev_q = clean_quads[0].astype(np.float32)
        sm_quads.append(prev_q)
        for q in clean_quads[1:]:
            qf = q.astype(np.float32)
            prev_q = beta_q * prev_q + (1.0 - beta_q) * qf
            sm_quads.append(prev_q)
    else:
        sm_quads = clean_quads
    inverse_transforms = [calc_alignment_coefficients(q + 0.5, [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]) for q in sm_quads]
    del kp_extractor.detector

    oy1,oy2,ox1,ox2 = cox
    face_det_results = face_detect(full_frames, args, jaw_correction=True)

    for inverse_transform, crop, full_frame, face_det in zip(inverse_transforms, crops, full_frames, face_det_results):
        imc_pil = paste_image(inverse_transform, crop, Image.fromarray(
            cv2.resize(full_frame[int(oy1):int(oy2), int(ox1):int(ox2)], (256, 256))))

        ff = full_frame.copy()
        ff[int(oy1):int(oy2), int(ox1):int(ox2)] = cv2.resize(np.array(imc_pil.convert('RGB')), (ox2 - ox1, oy2 - oy1))
        oface, coords = face_det
        y1, y2, x1, x2 = coords
        refs.append(ff[y1: y2, x1:x2])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i
        frame_to_save = frames[idx].copy()
        face = refs[idx]
        oface, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        oface = cv2.resize(oface, (args.img_size, args.img_size))

        img_batch.append(oface)
        ref_batch.append(face) 
        mel_batch.append(m)
        coords_batch.append(coords)
        frame_batch.append(frame_to_save)
        full_frame_batch.append(full_frames[idx].copy())

        if len(img_batch) >= args.LNet_batch_size:
            img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
            img_masked = img_batch.copy()
            img_original = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch
            img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch, ref_batch  = [], [], [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch, ref_batch = np.asarray(img_batch), np.asarray(mel_batch), np.asarray(ref_batch)
        img_masked = img_batch.copy()
        img_original = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch = np.concatenate((img_masked, ref_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch, img_original, full_frame_batch


if __name__ == '__main__':
    main()
