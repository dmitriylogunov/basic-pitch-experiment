# Basic Pitch Experiment

A C# implementation for musical note detection using Spotify's Basic Pitch ONNX model.

## Overview

This project demonstrates how to:
- Load and process audio files using NAudio
- Run inference with the Basic Pitch ONNX model
- Extract musical notes with timing and pitch information
- Generate MIDI files from detected notes
- Save detailed note detection results

## Project Structure

```
├── Models/          # Data models
├── Audio/           # Audio processing (loading, normalization)
├── ML/              # Machine learning operations (ONNX inference)
├── Midi/            # MIDI file generation
├── Utils/           # Utility functions (note conversion, file operations)
└── Program.cs       # Main entry point
```

## Requirements

- .NET 6.0 or later
- Basic Pitch ONNX model file
- NuGet packages:
  - Microsoft.ML.OnnxRuntime
  - NAudio
  - Melanchall.DryWetMidi
  - MathNet.Numerics

## Setup

1. Clone the repository
2. Download the Basic Pitch ONNX model:
   ```bash
   git clone https://github.com/spotify/basic-pitch.git
   ```
3. The model path is expected at: `./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx`

## Usage

```bash
dotnet run
```

The application will:
1. Load `guitar_sample.wav`
2. Process it through the Basic Pitch model
3. Generate `output.mid` (MIDI file)
4. Generate `detected_notes.txt` (detailed note information)

## How It Works

1. **Audio Loading**: Converts audio to 22,050 Hz mono format
2. **Normalization**: Ensures audio levels are optimal for processing
3. **Model Inference**: Runs the Basic Pitch ONNX model
4. **Note Detection**: Extracts notes from model outputs using confidence thresholds
5. **MIDI Generation**: Converts detected notes to standard MIDI format

## License

This project is for experimental purposes. Basic Pitch is created by Spotify Research.