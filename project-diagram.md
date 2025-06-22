# Basic Pitch Experiment - Architecture Diagram

## Project Structure and Data Flow

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[MainWindow.xaml]
        UICode[MainWindow.xaml.cs]
        UI --> UICode
    end

    subgraph "Audio Processing"
        AudioFile[Audio File<br/>WAV/MP3]
        AudioLoader[Audio Loader<br/>NAudio]
        AudioData[Float Array<br/>22050 Hz]
        
        AudioFile --> AudioLoader
        AudioLoader --> AudioData
    end

    subgraph "ML Layer - BasicPitchModel"
        IModel[IModelInference<br/>Interface]
        BPModel[BasicPitchModel.cs]
        ModelInput[ModelInput<br/>- AudioData<br/>- SampleRate<br/>- Parameters]
        
        IModel -.implements.-> BPModel
        AudioData --> ModelInput
        ModelInput --> BPModel
    end

    subgraph "ONNX Processing"
        WindowProc[ProcessAudioWindows<br/>- 2 sec windows<br/>- 30 frame overlap]
        SingleWindow[ProcessSingleWindow<br/>- Creates tensor<br/>- Runs inference]
        ONNXSession[ONNX Runtime<br/>Session]
        ONNXModel[basic_pitch.onnx<br/>Model File]
        
        BPModel --> WindowProc
        WindowProc --> SingleWindow
        SingleWindow --> ONNXSession
        ONNXModel --> ONNXSession
    end

    subgraph "Model Outputs"
        Tensors[Raw Tensors<br/>- Note: 1 x frames x 88<br/>- Onset: 1 x frames x 88<br/>- Contour: 1 x frames x 264]
        WindowResult[WindowResult<br/>- Flattened arrays<br/>- Frame count<br/>- Key count]
        
        ONNXSession --> Tensors
        Tensors --> WindowResult
    end

    subgraph "Result Merging"
        MergeWindows[MergeWindowResults<br/>- Combines windows<br/>- Handles overlap<br/>- Index calculation]
        MergedArrays[Merged 2D Arrays<br/>- Notes: frames x 88<br/>- Onsets: frames x 88<br/>- Contour: frames x 264]
        
        WindowResult --> MergeWindows
        MergeWindows --> MergedArrays
    end

    subgraph "Note Detection"
        NoteDetection[Note Detection<br/>- Threshold filtering<br/>- Onset alignment<br/>- Duration check]
        DetectedNotes[DetectedNote List<br/>- Start/End time<br/>- MIDI pitch<br/>- Confidence]
        
        MergedArrays --> NoteDetection
        NoteDetection --> DetectedNotes
    end

    subgraph "Output Generation"
        ModelOutput[ModelOutput<br/>- Notes<br/>- Activations<br/>- Statistics]
        MusicNotation[Music Notation<br/>Display]
        ProcessingLog[Processing Log<br/>Window]
        
        DetectedNotes --> ModelOutput
        ModelOutput --> MusicNotation
        ModelOutput --> ProcessingLog
    end

    UICode --> AudioLoader
    UICode --> BPModel
    ModelOutput --> UICode

    style UI fill:#f9f,stroke:#333,stroke-width:2px
    style ONNXModel fill:#9f9,stroke:#333,stroke-width:2px
    style BPModel fill:#99f,stroke:#333,stroke-width:2px
    style MergeWindows fill:#ff9,stroke:#333,stroke-width:2px
```

## Data Flow Detail

```mermaid
sequenceDiagram
    participant User
    participant UI as MainWindow
    participant Audio as AudioLoader
    participant Model as BasicPitchModel
    participant ONNX as ONNX Runtime
    participant Merge as MergeResults

    User->>UI: Load Audio File
    UI->>Audio: Read WAV/MP3
    Audio->>UI: Float[] (22050 Hz)
    
    User->>UI: Click Process
    UI->>Model: ProcessAudio(input)
    
    loop For each 2-second window
        Model->>Model: Extract window<br/>(43844 samples)
        Model->>ONNX: Run inference
        ONNX->>Model: Tensors (note, onset, contour)
        Model->>Model: Store WindowResult
    end
    
    Model->>Merge: MergeWindowResults
    Note over Merge: Handle overlapping frames<br/>Calculate indices: f * keys + k
    Merge->>Model: Merged 2D arrays
    
    Model->>Model: Detect notes<br/>(threshold, duration)
    Model->>UI: ModelOutput
    UI->>User: Display results
```

## Key Components

```mermaid
classDiagram
    class IModelInference {
        <<interface>>
        +ProcessAudio(ModelInput) ModelOutput
    }
    
    class ModelInput {
        +float[] AudioData
        +int SampleRate
        +InferenceParameters Parameters
    }
    
    class InferenceParameters {
        +float NoteThreshold
        +float OnsetThreshold
        +float MinNoteLength
        +int OverlappingFrames
        +bool AutoApplySigmoid
    }
    
    class BasicPitchModel {
        -const int FFT_HOP = 256
        -const int MODEL_SAMPLE_RATE = 22050
        -const int AUDIO_WINDOW_LENGTH = 43844
        -const int PIANO_KEYS = 88
        -const int CONTOUR_BINS = 264
        
        +ProcessAudio(ModelInput) ModelOutput
        -ProcessAudioWindows(float[], params) List~WindowResult~
        -ProcessSingleWindow(float[], params) WindowResult
        -MergeWindowResults(List~WindowResult~, int) MergedResults
    }
    
    class WindowResult {
        +float[] NoteActivations
        +float[] OnsetActivations
        +float[] ContourActivations
        +int FrameCount
        +int KeyCount
        +int ContourBinCount
        +int StartSample
        +bool NeedsSigmoid
    }
    
    class ModelOutput {
        +List~DetectedNote~ Notes
        +float[][] NoteActivations
        +float[][] OnsetActivations
        +float[][] PitchContour
        +int FrameCount
        +float FrameRate
    }
    
    class DetectedNote {
        +float StartTime
        +float EndTime
        +int Pitch
        +float Confidence
        +float Frequency
        +string NoteName
    }
    
    IModelInference <|.. BasicPitchModel : implements
    BasicPitchModel --> WindowResult : creates
    BasicPitchModel --> ModelOutput : returns
    ModelInput --> InferenceParameters : contains
    ModelOutput --> DetectedNote : contains
```

## Index Calculation Flow

```mermaid
graph LR
    A[Audio Data<br/>573120 samples] --> B[ProcessAudioWindows]
    B --> C{Calculate Indices}
    
    subgraph "Index Calculation"
        E[frame * keysPerFrame + key]
        F[Flattened 1D Array]
        G[Uses tensor dimensions<br/>for correct indexing]
    end
    
    C --> E
    E --> F
    F --> G
    G --> H[Successfully merged<br/>window results]
    
    style H fill:#9f9,stroke:#333,stroke-width:2px
```

## Processing Parameters

```mermaid
graph TD
    subgraph "Window Processing"
        A[Window Size: 43844 samples<br/>~2 seconds @ 22050Hz]
        B[Overlap: 30 frames<br/>= 7680 samples]
        C[Hop Size: 36164 samples]
    end
    
    subgraph "Model Constants"
        D[FFT Hop: 256]
        E[Frame Rate: 86 fps]
        F[Piano Keys: 88]
        G[Contour Bins: 264]
    end
    
    A --> B
    B --> C
    D --> E
```

## File Structure

```mermaid
graph TD
    Root[basic-pitch-experiment/]
    Root --> ML[ML/]
    Root --> Examples[Examples/]
    Root --> Main[MainWindow.xaml/.cs]
    Root --> Claude[CLAUDE.md]
    
    ML --> Interface[IModelInference.cs]
    ML --> Model[BasicPitchModel.cs]
    ML --> Legacy[ModelInference.cs]
    ML --> README[README.md]
    
    Examples --> ExampleCode[BasicPitchExample.cs]
    
    Root --> ONNX[basic_pitch.onnx<br/>~10MB model file]
    
    style ONNX fill:#9f9,stroke:#333,stroke-width:2px
    style Model fill:#99f,stroke:#333,stroke-width:2px
```

## Technical Notes

1. **Window Processing**: Audio is processed in ~2 second windows with 30-frame overlap
2. **Tensor Handling**: 3D tensors from ONNX are flattened to 1D arrays for processing
3. **Index Calculation**: Uses formula `frame * keysPerFrame + key` to access flattened data
4. **Model Output**: Generates piano roll representation with 88 keys and pitch contour with 264 frequency bins