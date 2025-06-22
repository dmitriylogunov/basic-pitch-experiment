# Instructions for Claude

## Build Commands
- **IMPORTANT**: Skip running `dotnet build` or `dotnet run` unless explicitly requested by the user
- The user will handle building and running the project themselves

## Project Notes
- This is a Basic Pitch ONNX model experiment for audio transcription
- The model requires audio at 22050 Hz sample rate
- Main abstraction is in `ML/BasicPitchModel.cs` with clean input/output interfaces