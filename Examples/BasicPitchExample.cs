using BasicPitchExperimentApp.ML;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;
using System;
using System.Linq;

namespace BasicPitchExperimentApp.Examples
{
    /// <summary>
    /// Example demonstrating how to use the Basic Pitch model abstraction
    /// </summary>
    public class BasicPitchExample
    {
        public static void Example1_SimpleUsage()
        {
            // Load the model
            string modelPath = "./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx";
            using var model = new BasicPitchModel(modelPath);
            
            // Prepare audio data (must be 22050 Hz sample rate)
            float[] audioData = LoadAudioAt22050Hz("path/to/audio.wav");
            
            // Create input with default parameters
            var input = new ModelInput
            {
                AudioData = audioData,
                SampleRate = 22050
            };
            
            // Process audio
            var output = model.ProcessAudio(input);
            
            // Use the detected notes
            Console.WriteLine($"Detected {output.Notes.Count} notes:");
            foreach (var note in output.Notes.Take(5))
            {
                Console.WriteLine($"  MIDI {note.MidiNote} at {note.StartTime:F2}s for {note.Duration:F2}s");
            }
            
            // Access raw activations if needed
            Console.WriteLine($"Processed {output.FrameCount} frames at {output.FrameRate} FPS");
            if (output.NoteActivations != null)
            {
                Console.WriteLine($"Note activation shape: [{output.NoteActivations.GetLength(0)}, {output.NoteActivations.GetLength(1)}]");
            }
        }
        
        public static void Example2_CustomParameters()
        {
            string modelPath = "./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx";
            using var model = new BasicPitchModel(modelPath);
            
            float[] audioData = LoadAudioAt22050Hz("path/to/audio.wav");
            
            // Create input with custom parameters
            var input = new ModelInput
            {
                AudioData = audioData,
                SampleRate = 22050,
                Parameters = new InferenceParameters
                {
                    NoteThreshold = 0.5f,      // Higher threshold for more confident detections
                    OnsetThreshold = 0.5f,     // Onset detection threshold
                    MinNoteLength = 0.2f,      // Filter out notes shorter than 200ms
                    OverlappingFrames = 30,    // Window overlap for smoother detection
                    AutoApplySigmoid = true    // Automatically apply sigmoid if needed
                }
            };
            
            var output = model.ProcessAudio(input);
            
            // Check processing statistics
            var stats = output.Statistics;
            Console.WriteLine($"Processing Statistics:");
            Console.WriteLine($"  Windows processed: {stats.WindowsProcessed}");
            Console.WriteLine($"  Processing time: {stats.ProcessingTimeMs}ms");
            Console.WriteLine($"  Activation range: [{stats.MinActivation:F3}, {stats.MaxActivation:F3}]");
            Console.WriteLine($"  Average activation: {stats.AverageActivation:F3}");
            Console.WriteLine($"  Sigmoid applied: {stats.SigmoidApplied}");
        }
        
        public static void Example3_RawActivations()
        {
            string modelPath = "./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx";
            using var model = new BasicPitchModel(modelPath);
            
            float[] audioData = LoadAudioAt22050Hz("path/to/audio.wav");
            
            var input = new ModelInput
            {
                AudioData = audioData,
                SampleRate = 22050
            };
            
            var output = model.ProcessAudio(input);
            
            // Access raw model outputs for custom processing
            if (output.NoteActivations == null)
            {
                Console.WriteLine("No raw activations available");
                return;
            }
            
            int frames = output.NoteActivations.GetLength(0);
            int pianoKeys = output.NoteActivations.GetLength(1);
            
            // Find the most active piano key
            float maxActivation = 0;
            int maxKey = 0;
            int maxFrame = 0;
            
            for (int f = 0; f < frames; f++)
            {
                for (int k = 0; k < pianoKeys; k++)
                {
                    float activation = output.NoteActivations[f, k];
                    if (activation > maxActivation)
                    {
                        maxActivation = activation;
                        maxKey = k;
                        maxFrame = f;
                    }
                }
            }
            
            int midiNote = 21 + maxKey; // A0 = MIDI 21
            float timeInSeconds = maxFrame / output.FrameRate;
            
            Console.WriteLine($"Strongest activation:");
            Console.WriteLine($"  MIDI note: {midiNote} ({NoteUtils.GetNoteName(midiNote)})");
            Console.WriteLine($"  Time: {timeInSeconds:F2}s");
            Console.WriteLine($"  Confidence: {maxActivation:F3}");
            
            // Access onset and contour data
            if (output.OnsetActivations != null)
            {
                Console.WriteLine($"Onset activation at same point: {output.OnsetActivations[maxFrame, maxKey]:F3}");
            }
            
            if (output.PitchContour != null)
            {
                Console.WriteLine($"Pitch contour has {output.PitchContour.GetLength(1)} frequency bins");
            }
        }
        
        private static float[] LoadAudioAt22050Hz(string path)
        {
            // Placeholder - in real usage, use NAudio or similar to load and resample audio
            throw new NotImplementedException("Implement audio loading with resampling to 22050 Hz");
        }
    }
}