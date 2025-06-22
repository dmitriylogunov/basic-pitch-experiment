# Basic Pitch Model Abstraction

This directory contains the refactored Basic Pitch ONNX model implementation with a clean abstraction layer.

## Overview

The refactoring introduces a clear separation between inputs, processing, and outputs:

- **`IModelInference`**: Interface defining the contract for music transcription models
- **`BasicPitchModel`**: Implementation of the Basic Pitch ONNX model
- **`ModelInput`**: Structured input parameters including audio data and configuration
- **`ModelOutput`**: Comprehensive output including detected notes and raw activations
- **`ModelInference`**: Legacy static wrapper for backward compatibility

## Key Components

### ModelInput
```csharp
var input = new ModelInput
{
    AudioData = audioData,      // float[] at 22050 Hz
    SampleRate = 22050,        // Must be 22050 Hz
    Parameters = new InferenceParameters
    {
        NoteThreshold = 0.3f,      // Note detection threshold (0-1)
        OnsetThreshold = 0.5f,     // Onset detection threshold (0-1)
        MinNoteLength = 0.127f,    // Minimum note duration in seconds
        OverlappingFrames = 30,    // Frames to overlap between windows
        AutoApplySigmoid = true    // Apply sigmoid to raw outputs
    }
};
```

### ModelOutput
```csharp
public class ModelOutput
{
    List<DetectedNote> Notes;           // Detected musical notes
    float[,] NoteActivations;          // Raw [frames, 88 keys]
    float[,] OnsetActivations;         // Raw [frames, 88 keys]
    float[,] PitchContour;            // Raw [frames, 264 bins]
    int FrameCount;                    // Number of time frames
    float FrameRate;                   // Frames per second (86)
    InferenceStatistics Statistics;    // Processing statistics
}
```

## Usage Example

```csharp
// Simple usage
using var model = new BasicPitchModel("path/to/nmp.onnx");
var input = new ModelInput 
{ 
    AudioData = audioData, 
    SampleRate = 22050 
};
var output = model.ProcessAudio(input);

// Process detected notes
foreach (var note in output.Notes)
{
    Console.WriteLine($"Note {note.MidiNote} at {note.StartTime}s");
}
```

## Model Architecture

The Basic Pitch model:
- **Input**: Audio windows of 43,844 samples (~2 seconds at 22050 Hz)
- **Processing**: Sliding window approach with configurable overlap
- **Outputs**:
  - Note activations: Confidence for each of 88 piano keys per frame
  - Onset activations: Onset detection per key
  - Pitch contour: 264 frequency bins for fine-grained pitch tracking

## Key Features

1. **Clean Abstraction**: Clear separation of inputs, processing, and outputs
2. **Automatic Sigmoid**: Detects and applies sigmoid transformation when needed
3. **Comprehensive Statistics**: Processing time, activation ranges, window counts
4. **Raw Data Access**: Access to raw model outputs for custom processing
5. **Backward Compatibility**: Legacy `ModelInference` class still works

## Technical Details

- Sample Rate: 22050 Hz (required)
- Window Size: 43,844 samples
- Frame Rate: 86 FPS (22050 / 256 hop size)
- Piano Keys: 88 (A0 to C8, MIDI 21-108)
- Overlap: 30 frames default (configurable)

## Migration from Legacy Code

Old code:
```csharp
var notes = ModelInference.ProcessFullAudio(session, audioData, sampleRate);
```

New code:
```csharp
using var model = new BasicPitchModel(modelPath);
var output = model.ProcessAudio(new ModelInput 
{ 
    AudioData = audioData, 
    SampleRate = sampleRate 
});
var notes = output.Notes;
```