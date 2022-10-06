#!/usr/bin/env python3

import argparse
import librosa
import torchaudio
import yaml
import noisereduce as nr
import PySimpleGUI as sg
import threading

# import sounddevice as sd
import soundcard as sc

from models import *
from Utils.JDC.model import JDCNet
# MAX_WAV_VALUE = 32768.0


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--input-device", type=str, help="input device name", default="")
parser.add_argument("-o", "--output-device", type=str, help="output device name", default="")
# parser.add_argument("-c", "--channels", type=int, default=1,
#                     help="number of channels") # not currently used
parser.add_argument("-t", "--dtype", help="audio data type")
parser.add_argument("-s", "--samplerate", type=float, help="sampling rate", default=24000)
parser.add_argument("-b", "--blocksize", type=int, help="block size", default=30*300)
parser.add_argument("-w", "--wave-buffer-size", type=float, help="wave buffer size in frames", default=24000)
parser.add_argument("-bl", "--blend-length", type=float, help="number of frames to crossfade between each block", default=300)
parser.add_argument("-ngh", "--noise-gate-hold", type=float, help="time in ms to hold open the noise gate", default=70.0)
parser.add_argument("-oms", "--out-mel-shift", type=int, 
                    help="shift converted mel back by this many mels, prevents artifacts at end of converted audio", default=2)
parser.add_argument("-ws", "--wave-shift", type=int, 
                    help="shift vocoder output back by this many samples, prevents artifacts at end of vocoder output", default=300)
parser.add_argument("-v", "--vocoder-model", type=str, help="path to vocoder model to use",
                    default="Vocoder/pretrained_PWG/checkpoint-400000steps.pkl")
parser.add_argument("-sg", "--stargan-model", type=str, help="path to trained starganv2VC model to use", 
                    default="Models/VCTK20/epoch_00150.pth")
parser.add_argument("-sgc", "--stargan-model-config", type=str, help="path to stargan config file",
                    default="Models/VCTK20/config.yml")
parser.add_argument("-f0p", "--f0-model", type=str, help="path to f0 model to use", 
                    default="Utils/JDC/bst.t7")
parser.add_argument("-p", "--speaker", type=float, help="speaker id", default=0)
parser.add_argument("-r", "--reference", type=str, 
                    help="path to reference wav for style encoding, set blank to use mapping network", 
                    default="")
parser.add_argument("-rs", "--reference-seed", type=int, 
                    help="seed for the speaker style rng, only used when reference is left blank", default=None)

args, unknown = parser.parse_known_args()


def load_vocoder(vocoder_path):
    try:
        from parallel_wavegan.utils import load_model
        vocoder = load_model(vocoder_path).to('cuda').eval()
        vocoder.remove_weight_norm()
        _ = vocoder.eval()
        return vocoder
    except BaseException as e:
        raise SystemExit(str(e))

def build_model(model_params={}):
    params = Munch(model_params)
    generator = Generator(params.dim_in, params.style_dim, params.max_conv_dim, w_hpf=params.w_hpf, F0_channel=params.F0_channel)
    mapping_network = MappingNetwork(params.latent_dim, params.style_dim, params.num_domains, hidden_dim=params.max_conv_dim)
    style_encoder = StyleEncoder(params.dim_in, params.style_dim, params.num_domains, params.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def load_models(stargan_path, stargan_config_path, f0_path):
    try:
        F0_model = JDCNet(num_class=1, seq_len=192)
        params = torch.load(f0_path)['net']
        F0_model.load_state_dict(params)
        _ = F0_model.eval()
        F0_model = F0_model.to('cuda')

        with open(stargan_config_path) as f:
            starganv2_config = yaml.safe_load(f)
        starganv2 = build_model(model_params=starganv2_config["model_params"])
        num_speakers = starganv2_config["model_params"]["num_domains"]
        params = torch.load(stargan_path, map_location='cpu')
        params = params['model_ema']
        _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
        _ = [starganv2[key].eval() for key in starganv2]
        starganv2.style_encoder = starganv2.style_encoder.to('cuda')
        starganv2.mapping_network = starganv2.mapping_network.to('cuda')
        starganv2.generator = starganv2.generator.to('cuda')

        return starganv2, F0_model, num_speakers

    except BaseException as e:
        raise SystemExit(str(e))

# pads on both sides 
# Sample rate is unspecified and defaults to 16000, but changing it requires retraining all models.
# TODO: have preprocessing change according to model config
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300).to('cuda')

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float().to('cuda')
    mel_tensor = to_mel(wave_tensor).to('cuda')
    mean, std = -4, 4
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# From losses.py, but repurposed here for use with detecting silence
def log_norm(x, mean=-4, std=4, dim=1):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def compute_style(ref_path, starganv2, speaker, rng):
    with torch.no_grad():
        if ref_path == "":
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim, generator=rng).to('cuda'), label)
        else:
            wave, sr = librosa.load(ref_path, sr=args.samplerate) # providing an sr resamples the audio to that sr
            wave, index = librosa.effects.trim(wave, top_db=30)
            mel_tensor = preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        
        return ref

# removes the last args.blend_length worth from the returned wave, at the cost of latency
# crossfades the first args.blend_length with wave_cut cut from the previous wave
# returns the cut part to blend with the next wave
# |  wave  |cut|
#          | next wave |cut|
#                      | next next wave |cut|
def next_waves(wave, wave_cut):
    wave_left = wave.shape[0] - args.blocksize - args.blend_length - args.wave_shift
    wave_right = wave.shape[0] - args.blend_length - args.wave_shift
    wave_return = wave[wave_left:wave_right].squeeze()
    wave_cut_next = wave[wave_right:-args.wave_shift].squeeze()

    # TODO: improve blend smoothness, or find some way to do phase alignment
    buffer_weight = np.linspace(0, 1, args.blend_length)
    wave_return[:args.blend_length] = buffer_weight * wave_return[:args.blend_length] + (1 - buffer_weight) * wave_cut
    wave_return = np.clip(wave_return, -1, 1)

    return wave_return, wave_cut_next

def convert(mel, starganv2, F0_model, ref, speaker):
    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(mel.unsqueeze(1))
        out = starganv2.generator(mel.unsqueeze(1), ref, F0=f0_feat)
        
        c = out.transpose(-1, -2).squeeze().to('cuda')
    return c


generator = load_vocoder(args.vocoder_model).inference
starganv2, F0_model, num_speakers = load_models(args.stargan_model, args.stargan_model_config, args.f0_model)
if args.reference_seed is None:
    rng = torch.Generator()
    rng.seed()
else:
    rng = torch.Generator().manual_seed(args.reference_seed)

speaker = args.speaker
ref_path = args.reference
ref = compute_style(ref_path, starganv2, speaker, rng=rng)

# Set the speaker and mic to be used for VC, use the system defaults if no device specified
input_device = sc.default_microphone()
output_device = sc.default_speaker()
if args.input_device != "":
    input_device = sc.get_microphone(args.input_device)
if args.output_device != "":
    output_device = sc.get_speaker(args.output_device)

wave_buffer = np.zeros(args.wave_buffer_size) # Should this be on cuda?
wave_cut = None
noise = np.zeros(args.blocksize)
noise_threshold = 0.0
enable_conversion = True
enable_noisereduction = False
enable_stationary_nr = False
stop_audio = False

# Find the minimum mel bin amplitude by processing a wave of zeroes and taking the first element
minimum_mel = preprocess(np.zeros(args.blocksize))[0, 0, 0]

# TODO: implement a proper noise gate
# Create a filter to smoothen the silence detection with. Should be an odd number of entries.
silence_filter = torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]).to('cuda')
# Size of the kernel for max pooling, acting as the 'pre-open' and 'hold' time on a noise gate
# noise_gate_hold is in ms, so /1000 to convert to seconds, then * sample rate to get # samples, then / 300 to get # of mel frames.
# The * 2 + 1 is to center the kernel with the noise gate length on either side
silence_hold_size = int(np.ceil(args.noise_gate_hold / 1000. * args.samplerate / 300.) * 2 + 1)


def callback(indata, block_num, time, status):
    global wave_buffer
    global wave_cut

    with torch.no_grad():
        audio = indata[:, 0].squeeze() # only gets first channel
        
        # TODO: Add settings in GUI for noise reduction
        if enable_noisereduction:
            if enable_stationary_nr:
                audio = nr.reduce_noise(audio, sr=args.samplerate, stationary=True, y_noise=noise)
            else:
                audio = nr.reduce_noise(audio, sr=args.samplerate)

        wave_buffer = np.concatenate((wave_buffer, audio), axis=0) # appends audio to end of wave_buffer
        buffer_cut = int(wave_buffer.shape[0] - args.wave_buffer_size)
        wave_buffer = wave_buffer[max(0, buffer_cut):, ...] # shift back by blocksize, so size is args.wave_buffer_size

        mel = preprocess(wave_buffer).to('cuda') 
        # mel = mel[..., 1:] # omits first in 3rd dim

        # TODO: add these all to a pytorch sequence for neatness
        # Overwrite garbage noisy conversion output produced when input is just noise by 
        # detecting threshold before conversion, and applying it after
        # Make mask of mel where frames with a log norm below noise_threshold are set to the minimum mel value after conversion
        log_norm_mel = log_norm(mel, mean=-4, std=4, dim=1)
        # Some single frames were exceeding the noise_threshold, so weight them with their neighboring frames to smoothen their peak
        # Smoothen the silence detection using a filter across multiple frames
        filtered_mel = torch.nn.functional.conv1d(log_norm_mel.unsqueeze(0), silence_filter, padding='same').squeeze(0)
        # Use a max pool layer as the 'pre-open' and 'hold' on a noise gate, with duration == the size of the kernel in frames / 2
        pooled_mel = torch.nn.functional.max_pool1d(filtered_mel, kernel_size=silence_hold_size, stride=1, padding=silence_hold_size//2)
        silence_mask = pooled_mel < noise_threshold

        out = convert(mel, starganv2, F0_model, ref, args.speaker)

        silence_mask = silence_mask.T[1:, ...]
        mel_silenced = out.masked_fill_(silence_mask, minimum_mel)

        # TODO: implement 'attack' and 'release' times as in a noise gate
        c = mel_silenced.squeeze().to('cuda')

        # Use vocoder to convert back to wave
        wave = generator(c)
        wave = wave.cpu().numpy()
        wave.dtype = np.float32

        if wave_cut is None:
            wave_left = wave.shape[0] - args.blocksize - args.blend_length - args.wave_shift
            wave_right = wave.shape[0] - args.blend_length - args.wave_shift
            wave_cut = wave[wave_right:-args.wave_shift, ...].squeeze()
            wave = wave[wave_left:wave_right, ...]
        else:
            wave, wave_cut = next_waves(wave, wave_cut)

    out = np.expand_dims(wave.squeeze(), axis=1)
    return out

# sounddevice replacement
# numframes must be two or four times smaller than the blocksize of the recorder and player
def convert_audio():
    global noise_threshold
    global enable_conversion
    global stop_audio
    with input_device.recorder(samplerate=args.samplerate, blocksize=args.blocksize*2) as mic_recorder, \
        output_device.player(samplerate=args.samplerate, blocksize=args.blocksize*2) as spk_player:
        indata = np.zeros((args.blocksize, 1))
        outdata = np.zeros((args.blocksize, 1))
        block_num = 0

        noise = mic_recorder.record(numframes=args.blocksize)[:, 0].squeeze() # [:, 0] only gets one channel
        # Use the loudest recorded noise as the threshold for silencing audio
        noise_threshold = torch.max(log_norm(preprocess(noise)))

        print("Beginning Voice Conversion with the following settings:",
            "\nReference RNG Seed:", rng.initial_seed())

        while True:
            block_num += 1
            indata[:] = mic_recorder.record(numframes=args.blocksize)[:, 0:1] # [:, 0] only gets one channel
            if enable_conversion:
                outdata = callback(indata, block_num, 0, 0)
            else:
                outdata = indata
            spk_player.play(outdata)
            # stop audio thread when GUI window closed
            if stop_audio:
                break


audio_thread = threading.Thread(target=convert_audio)

# TODO: add a slider to add or subtract from the noise threshhold
# TODO: add option to set the RNG seed
# TODO: add option to browse files for reference audio for style encoder
# TODO: keep track of added style encoder reference audio paths
# TODO: only show style encoder reference audio paths associated with the current speaker
# TODO: add ability to save and load added reference audio paths and assign speaker names to buttons
layout = [[sg.Button('Conversion Enabled', key='EC', button_color="green")],
        [sg.Button('Noise Reduction Disabled', key='NR_ON', button_color="gray"),
            sg.Button('Record Noise', key='NR_REC'),
            sg.Checkbox('Stationary', key='NR_STAT', enable_events=True)],
        [sg.Text('Selected Voice:')],
        [sg.Button(str(i), key='VC_' + str(i), button_color="gray") for i in range(num_speakers)]]

window = sg.Window('Voice Conversion', layout)

audio_thread.start()

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        stop_audio = True
        break
    if event == 'EC':
        if enable_conversion:
            window['EC'].update('Conversion Disabled', button_color="red")
            enable_conversion = False
        else:
            window['EC'].update('Conversion Enabled', button_color="green")
            enable_conversion = True
    elif isinstance(event, str) and event[:3] == 'VC_':
        # Re-enable previous speaker's button
        window['VC_' + str(speaker)].update(disabled=False, button_color="gray")
        speaker = int(event[3:])
        ref = compute_style(ref_path, starganv2, speaker, rng)
        # Disable current speaker's button
        window['VC_' + str(speaker)].update(disabled=True, button_color="green")
    elif event == 'NR_ON':
        if enable_noisereduction:
            window['NR_ON'].update('Noise Reduction Disabled', button_color="gray")
            enable_noisereduction = False
        else:
            window['NR_ON'].update('Noise Reduction Enabled', button_color="blue")
            enable_noisereduction = True
    elif event == 'NR_STAT':
        enable_stationary_nr = values['NR_STAT']
    elif event == 'NR_REC':
        with input_device.recorder(samplerate=args.samplerate, blocksize=args.blocksize*2) as mic_recorder:
            # TODO: print this on the GUI instead
            # print("Recording noisy audio for noise reduction, please do not speak yet...")
            noise = mic_recorder.record(numframes=args.blocksize)[:, 0].squeeze() # [:, 0] only gets one channel
            # print("Done recording noise!")
            noise_threshold = torch.max(log_norm(preprocess(noise)))

