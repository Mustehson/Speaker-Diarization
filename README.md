# Speaker Diarization

Speaker diarization is an implementation of the 'faster_whisper', which is used for transcribing audio files faster. In this implementation, we employ Agglomerative Clustering to group speakers within the audio and generate a text file that includes the speaker diarization results. This repository provides a powerful tool for segmenting and identifying distinct speakers in audio recordings, allowing for transcription and analysis of multi-speaker content.

## Features

- Efficient transcription of audio recordings.
- Automatic clustering of speakers using Agglomerative Clustering.
- Output of speaker diarization results to a text file.

## Usage

You can use this tool by providing your audio file as input. The tool will process the audio, identify and label different speakers, and create a text transcript with speaker diarization information.

```bash
python app.py -model <model_name> -file_name <path_to_audio_file> -num_speakers <num_speakers>
```
1. -model: Specifies the Whisper model to use. If not provided, the default model 'large-v2' is used.
2. -file_name: Indicates the path to the audio file that you want to process.
3. -num_speakers: Specifies the number of speakers in the audio.

## Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies and libraries.
3. Run the script with your audio file.
