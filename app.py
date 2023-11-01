import os
import wave
import torch
import argparse
import datetime
import subprocess
import contextlib
import numpy as np
import pyannote.audio
from tqdm import tqdm
from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.core import Segment
from faster_whisper import WhisperModel
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


MODEL = "large-v2"

def get_list(array_segments):
  segments = []
  for segment in tqdm(array_segments, desc='Generating Transcript'):
    formatted_segment = {
          'id': segment.id,
          'seek': segment.seek,
          'start': segment.start,
          'end': segment.end,
          'text': segment.text,
          'tokens': segment.tokens,
          'temperature': segment.temperature,
          'avg_logprob': segment.avg_logprob,
          'compression_ratio': segment.compression_ratio,
          'no_speech_prob': segment.no_speech_prob
      }
    segments.append(formatted_segment)
  return segments

def convert_to_mono(audio_file):
    
    audio = AudioSegment.from_file(audio_file)

    if audio_file[-3:] != 'wav':
      subprocess.call(['ffmpeg', '-i', audio_file, 'audio.wav', '-y'])
      audio_file = 'audio.wav'

    if audio.channels > 1:

        audio = audio.set_channels(1)

        mono_audio_file = 'mono_' + audio_file
        audio.export(mono_audio_file, format="wav")
        return mono_audio_file
    return audio_file

def get_duration(path):
  with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    return duration

def segment_embedding(segments, duration, num_speakers, embedding_model, path):
  audio = Audio()


  embeddings = np.zeros(shape=(len(segments), 192))
  for i, segment in tqdm(enumerate(segments), desc='Generating Segments for Speakers'):
    start = segment['start']
    end = min(duration, segment['end'])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    embeddings[i] = embedding_model(waveform[None])

  embeddings = np.nan_to_num(embeddings)

  clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
  labels = clustering.labels_
  for i in range(len(segments)):

    segments[i]['speaker'] = 'SPEAKER ' + str(labels[i] + 1)
  return segments

def time(secs):
  return datetime.timedelta(seconds=round(secs))

def get_transcript(time, segments):
  f = open("transcript.txt", "w")

  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
    f.write(segment["text"][1:] + ' ')
  f.close()

  file_paths = ['audio.wav', 'mono_audio.wav']

  for file_path in file_paths:
    if os.path.exists(file_path):
        os.remove(file_path)
  return 'Transcript is Successfully Generated'


def main():
  print(args.model)
  print(args.file_name)
  print(args.num_speakers)
  embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda"))
  model = args.model if args.model else MODEL
   
  fast_model = WhisperModel(model, device="cuda", compute_type="float32")
  segments, _ = fast_model.transcribe(args.file_name, beam_size=1)

  path = convert_to_mono(args.file_name)
  duration = get_duration(args.file_name)
  segments = get_list(segments)
  generated_segments = segment_embedding(segments, duration,
                                         num_speakers = args.num_speakers,
                                         embedding_model=embedding_model,
                                         path=args.filename)
  print(get_transcript(time, generated_segments))
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Transcribe Audio with speakers Segmentation')
  parser.add_argument('-model', type=str, help='Whisper Modle')
  parser.add_argument('-file_name', type=str, help='Path of your audio file')
  parser.add_argument('-num_speakers', type=int, help='Number of Spekers in Audio')
    
  args = parser.parse_args()
  main()
    


