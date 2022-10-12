#!/usr/bin/env python3

import argparse
from pickle import TRUE
import librosa
import torchaudio
import yaml
import noisereduce as nr
import PySimpleGUI as sg
import threading
import time as Time
# import scipy.io

import sounddevice as sd

from models import *
from Utils.JDC.model import JDCNet
# MAX_WAV_VALUE = 32768.0


parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("-i", "--input-device", type=str, help="input device name", default=None)
# parser.add_argument("-o", "--output-device", type=str, help="output device name", default=None)
# parser.add_argument("-c", "--channels", type=int, default=1,
#                     help="number of channels") # not currently used
parser.add_argument("-t", "--dtype", help="audio data type")
parser.add_argument("-s", "--samplerate", type=float, help="sampling rate", default=24000)
parser.add_argument("-b", "--blocksize", type=int, help="block size in frames, automatically determined from latency when 0", default=0)
parser.add_argument("-l", "--latency", type=float, help="desired latency between input and output", default=0.5)
parser.add_argument("-w", "--wave-buffer-size", type=float, help="wave buffer size in frames", default=24000)
parser.add_argument("-bl", "--blend-length", type=float, help="number of frames to crossfade between each block", default=400)
parser.add_argument("-ngh", "--noise-gate-hold", type=float, help="time in ms to hold open the noise gate", default=120.0)
parser.add_argument("-oms", "--out-mel-shift", type=int, 
                    help="shift converted mel back by this many mels, prevents artifacts at end of converted audio", default=0)
parser.add_argument("-ws", "--wave-shift", type=int, 
                    help="shift vocoder output back by this many samples, prevents artifacts at end of vocoder output", default=12000)
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
def next_waves(wave, wave_cut, frames):
    wave_left = wave.shape[0] - frames - args.blend_length - args.wave_shift
    wave_right = wave.shape[0] - args.blend_length - args.wave_shift
    wave_return = wave[wave_left:wave_right].squeeze()
    wave_cut_next = wave[wave_right:-args.wave_shift].squeeze()

    # TODO: improve blend smoothness, or find some way to do phase alignment?
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

wave_buffer = np.zeros(args.wave_buffer_size)
wave_cut = None
noise = np.zeros(args.wave_buffer_size)
noise_threshold = 0.0
enable_conversion = True
enable_noisereduction = False
enable_stationary_nr = False
stop_audio = False

block_num = 0 # for bug testing

# Find the minimum mel bin amplitude by processing a wave of zeroes and taking the first element
minimum_mel = preprocess(np.zeros(args.samplerate))[0, 0, 0]

# TODO: implement a proper noise gate
# Create a filter to smoothen the silence detection with. Should be an odd number of entries.
silence_filter = torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]).to('cuda')
# Size of the kernel for max pooling, acting as the 'pre-open' and 'hold' time on a noise gate
# noise_gate_hold is in ms, so /1000 to convert to seconds, then * sample rate to get # samples, then / 300 to get # of mel frames.
# The * 2 + 1 is to center the kernel with the noise gate length on either side
silence_hold_size = int(np.ceil(args.noise_gate_hold / 1000. * args.samplerate / 300.) * 2 + 1)


def callback(indata, outdata, frames, time, status):
    global wave_buffer
    global wave_cut
    global block_num

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

        # Add part of a reversed copy of the wave buffer to the end of the wave buffer
        # that will be removed after vocoder output in order to prevent artifacts at end of vocoder output
        # Also, shorten cut the same amount from the beginning of wave_buffer as was appended, to maintain buffer
        wave_buffer_shifted = np.append(wave_buffer[args.wave_shift:], np.flip(wave_buffer)[:args.wave_shift])

        # If conversion is not enabled, just pass noise reduced audio to output.
        if not enable_conversion:
            outdata[:] = np.expand_dims(audio, axis=1)
            return

        mel = preprocess(wave_buffer_shifted).to('cuda') 
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
        
        out = convert(mel, starganv2, F0_model, ref, speaker)
        
        silence_mask = silence_mask.T[1:, ...]
        mel_silenced = out.masked_fill_(silence_mask, minimum_mel)

        # TODO: implement 'attack' and 'release' times as in a noise gate
        c = mel_silenced.squeeze().to('cuda')

        # Use vocoder to convert back to wave
        wave = generator(c)
        wave = wave.cpu().numpy()
        wave.dtype = np.float32

        # for bug testing, save wave here for each block, with file name according to block number
        # block_num = block_num + 1
        # scipy.io.wavfile.write("out/" + str(block_num) + ".wav", args.samplerate, wave)

        if wave_cut is None:
            wave_left = wave.shape[0] - frames - args.blend_length - args.wave_shift
            wave_right = wave.shape[0] - args.blend_length - args.wave_shift
            wave_cut = wave[wave_right:-args.wave_shift, ...].squeeze()
            wave = wave[wave_left:wave_right, ...]
        else:
            wave, wave_cut = next_waves(wave, wave_cut, frames)

    out = np.expand_dims(wave.squeeze(), axis=1)
    outdata[:] = out


audio_stream = sd.Stream(samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency, callback=callback)
audio_devices = sd.query_devices()
audio_hostapis = sd.query_hostapis()
# Add in the index into each host API dictionary
for index in range(len(audio_hostapis)):
    audio_hostapis[index]['index'] = index

# Separate audio devices into a nested list with outer list corresponding to which host API it uses, 
# inner list is a list of input devices or list of output devices,
# according to whether the device has any input or output channels
input_devices = []
output_devices = []
for list_API in audio_hostapis:
    input_devices_for_this_api = []
    output_devices_for_this_api = []
    for device_ID in list_API['devices']:
        if audio_devices[int(device_ID)]['max_input_channels'] > 0: # this device is an input
            # add a key and value to each device dictionary with its original index in sd.query_devices(), which sounddevice uses as an ID
            audio_devices[int(device_ID)]['index'] = int(device_ID)
            input_devices_for_this_api.append(audio_devices[int(device_ID)])
        if audio_devices[int(device_ID)]['max_output_channels'] > 0: # this device is an output
            # add a key and value to each device dictionary with its original index in sd.query_devices(), which sounddevice uses as an ID
            audio_devices[int(device_ID)]['index'] = int(device_ID)
            output_devices_for_this_api.append(audio_devices[int(device_ID)])
    input_devices.append(input_devices_for_this_api)
    output_devices.append(output_devices_for_this_api)


audio_device_menu_layout = ['', [
    'Host API', [str(api['name']) + '::ID  ' + str(api['index']) + '_API' for api in audio_hostapis],
    'Input Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_INPUT' for device in input_devices[sd.default.hostapi]],
    'Output Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_OUTPUT' for device in output_devices[sd.default.hostapi]]]]

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
        [sg.Button(str(i), key='VC_' + str(i), button_color="gray") for i in range(num_speakers)],
        [sg.ButtonMenu('Device Settings', audio_device_menu_layout, key='DEVICE_SETTINGS')]]

window = sg.Window('Voice Conversion', layout)

# Start processing audio.
audio_stream.start()

# Start the GUI
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        audio_stream.abort()
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
        noise[:] = wave_buffer
        noise_threshold = torch.max(log_norm(preprocess(noise)))
    elif event == 'DEVICE_SETTINGS':
        # make sure the value of DEVICE_SETTINGS is a string before comparing it to key strings
        if not isinstance(values['DEVICE_SETTINGS'], str):
            print("Value of DEVICE_SETTINGS event is not a string, when it should be.")
            break
        
        # Now, check which menu was selected from Host API, Input Device, or Output Device.
        if values['DEVICE_SETTINGS'][-4:] == '_API':
            # Not great implementation, but the menu requires values to be string
            api_ID = int(values['DEVICE_SETTINGS'][-7:-4])
            # When the Host API is changed, the Input Device and Output Device menus need to be updated.
            audio_device_menu_layout = ['', [
                'Host API', [str(api['name']) + '::ID  ' + str(api['index']) + '_API' for api in audio_hostapis],
                'Input Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_INPUT' for device in input_devices[api_ID]],
                'Output Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_OUTPUT' for device in output_devices[api_ID]]]]
            window['DEVICE_SETTINGS'].update(audio_device_menu_layout)
            # The current input and output devices should be changed to the default for that Host API?
            input_ID = int(audio_hostapis[api_ID]['default_input_device'])
            output_ID = int(audio_hostapis[api_ID]['default_output_device'])
            # For some reason some of the default devices are invalid. Try to instead find a valid device for that host API if it's invalid.
            try:
                sd.check_input_settings(device = input_ID, samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency)
            except:
                found_valid = False
                for device in input_devices[api_ID]:
                    try:
                        input_ID = device['index']
                        sd.check_input_settings(device = input_ID, samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency)
                    except:
                        # print('input device', input_ID, 'causes exception')
                        continue
                    else: # If one of the devices work, then keep it and stop searching.
                        # print('input device', input_ID, 'found that does not cause exception')
                        found_valid = True
                    if found_valid:
                        break
                if not found_valid:
                    print('No valid input device found for this host API :(')
            
            try:
                sd.check_output_settings(device = output_ID, samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency)
            except:
                found_valid = False
                for device in output_devices[api_ID]:
                    try:
                        output_ID = device['index']
                        sd.check_output_settings(device = output_ID, samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency)
                    except:
                        # print('output device', output_ID, 'causes exception')
                        continue
                    else: # If one of the devices work, then keep it and stop searching.
                        # print('output device', output_ID, 'found that does not cause exception')
                        found_valid = True
                    if found_valid:
                        break
                if not found_valid:
                    print('No valid output device found for this host API :(')
                # print('output device works fine')
            # print(input_ID, '|', output_ID)
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency, callback=callback)
            audio_stream.start()

        elif values['DEVICE_SETTINGS'][-6:] == '_INPUT':
            # Get the index of the device from the key string in values via a hardcoded position splice
            input_ID = int(values['DEVICE_SETTINGS'][-9:-6])
            # Get the current output device index to keep after stream is restarted
            output_ID = audio_stream.device[1]
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency, callback=callback)
            audio_stream.start()

        elif values['DEVICE_SETTINGS'][-7:] == '_OUTPUT':
            # Get the index of the device from the key string in values via a hardcoded position splice
            output_ID = int(values['DEVICE_SETTINGS'][-10:-7])
            # Get the current input device index to keep after stream is restarted
            input_ID = audio_stream.device[0]
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=args.samplerate, blocksize=args.blocksize, channels=1, latency=args.latency, callback=callback)
            audio_stream.start()
