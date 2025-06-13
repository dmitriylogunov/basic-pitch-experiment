using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BasicPitchApp.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BasicPitchApp.ML
{
    /// <summary>
    /// Handles ONNX model operations including preprocessing, inference, and post-processing
    /// </summary>
    public static class ModelInference
    {
        // Constants for Basic Pitch model configuration
        private const int WINDOW_SIZE = 2048;
        private const int HOP_LENGTH = 256;
        private const int N_BINS_PER_SEMITONE = 3;
        private const float MIN_NOTE_LENGTH = 0.127f;
        private const float ONSET_THRESHOLD = 0.5f;
        private const float FRAME_THRESHOLD = 0.3f;

        /// <summary>
        /// Preprocesses audio data to match the expected model input shape
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes the raw audio samples (numbers representing sound waves)
        /// - Organizes them into the format the AI model expects
        /// - The model wants chunks of exactly 43,844 samples each
        /// - Returns a "tensor" (multi-dimensional array) that the AI can process
        /// 
        /// WHY WE NEED THIS:
        /// - AI models are very picky about input format
        /// - Like how a recipe needs ingredients in specific amounts and order
        /// - The model was trained on data in this exact format
        /// 
        /// TENSOR CONCEPT:
        /// - A tensor is like a multi-dimensional spreadsheet
        /// - Our tensor has shape [chunks, 43844, 1]
        /// - chunks = how many pieces we split the audio into
        /// - 43844 = samples per chunk (determined by the model's requirements)
        /// - 1 = one channel (mono audio)
        /// </summary>
        /// <param name="audioData">Raw audio samples</param>
        /// <returns>Tensor containing the preprocessed audio data</returns>
        public static DenseTensor<float> PreprocessAudio(float[] audioData)
        {
            // The AI model expects exactly 43,844 audio samples per input chunk
            // This number was determined when the model was trained
            int expectedFeatures = 43844;
            
            // Calculate how many chunks we need to fit all our audio data
            // If audio is longer than 43,844 samples, we'll need multiple chunks
            int numChunks = (int)Math.Ceiling((double)audioData.Length / expectedFeatures);
            
            // Create a 3D tensor to hold our data
            // Think of it as a box with dimensions: [chunks, samples_per_chunk, channels]
            var tensor = new DenseTensor<float>(new[] { numChunks, expectedFeatures, 1 });
            
            // Fill the tensor with our audio data, chunk by chunk
            for (int chunk = 0; chunk < numChunks; chunk++)
            {
                // Calculate where to start reading from the original audio
                int startIdx = chunk * expectedFeatures;
                
                // Fill this chunk with audio samples
                for (int i = 0; i < expectedFeatures; i++)
                {
                    if (startIdx + i < audioData.Length)
                    {
                        // Copy the audio sample directly
                        // Each sample represents the sound wave amplitude at that moment
                        tensor[chunk, i, 0] = audioData[startIdx + i];
                    }
                    else
                    {
                        // If we run out of audio data, fill with silence (zero)
                        // This is called "padding" - like adding blank pages to a book
                        tensor[chunk, i, 0] = 0.0f;
                    }
                }
            }
            
            return tensor;
        }

        /// <summary>
        /// Runs inference on the ONNX model with the preprocessed audio data
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes our prepared audio data and feeds it to the AI model
        /// - The AI model analyzes the audio and predicts what notes are present
        /// - Returns the model's "thoughts" as numerical predictions
        /// 
        /// INFERENCE CONCEPT:
        /// - "Inference" means using a trained AI model to make predictions
        /// - Like showing a photo to someone and asking "what do you see?"
        /// - The model doesn't learn here - it just uses what it already knows
        /// 
        /// MODEL INPUTS/OUTPUTS:
        /// - Input: Our audio data in the format the model expects
        /// - Output: Numbers indicating how confident the model is about each note
        /// - Higher numbers = more confident that note is present
        /// </summary>
        /// <param name="session">ONNX inference session</param>
        /// <param name="inputTensor">Preprocessed audio data</param>
        /// <returns>Dictionary containing model outputs</returns>
        public static Dictionary<string, DenseTensor<float>> RunInference(InferenceSession session, DenseTensor<float> inputTensor)
        {
            // Find out what the model calls its input
            // Different models might call their input "audio", "input", "data", etc.
            var inputName = session.InputMetadata.Keys.First();
            var inputMetadata = session.InputMetadata[inputName];
            
            // Print debug information so we can see if everything matches up
            Console.WriteLine($"Using input name: {inputName}");
            Console.WriteLine($"Expected input shape: {string.Join("x", inputMetadata.Dimensions)}");
            Console.WriteLine($"Provided tensor shape: {string.Join("x", inputTensor.Dimensions.ToArray())}");
            
            // Package our data for the model
            // We need to tell the model: "here's your input data, and it's called [inputName]"
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            // Ask the AI model to analyze our audio data
            // This is where the magic happens - the model processes the audio
            using var results = session.Run(inputs);
            
            // Collect the model's predictions
            // The model might output multiple types of information
            var outputs = new Dictionary<string, DenseTensor<float>>();
            foreach (var result in results)
            {
                // Make sure this output contains numerical predictions (floats)
                if (result.Value is Tensor<float> tensor)
                {
                    // Store the predictions with their name for later use
                    outputs[result.Name] = tensor.ToDenseTensor();
                    Console.WriteLine($"Output '{result.Name}': {string.Join('x', tensor.Dimensions.ToArray())}");
                    
                    // DEBUG: Show some sample values from each output
                    var denseTensor = tensor.ToDenseTensor();
                    var maxVal = denseTensor.ToArray().Max();
                    var minVal = denseTensor.ToArray().Min();
                    var avgVal = denseTensor.ToArray().Average();
                    Console.WriteLine($"  -> Value range: {minVal:F4} to {maxVal:F4}, average: {avgVal:F4}");
                }
            }
            
            return outputs;
        }

        /// <summary>
        /// Processes the model outputs to extract individual notes with timing and pitch information
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes the AI model's raw predictions (just numbers)
        /// - Converts them into understandable musical information
        /// - Finds where notes start and stop
        /// - Determines what pitch (musical note) each detection represents
        /// 
        /// HOW NOTE DETECTION WORKS:
        /// 1. The model outputs confidence scores for each possible note at each time
        /// 2. We look for scores above our threshold (high confidence)
        /// 3. We group consecutive high-confidence detections into notes
        /// 4. We filter out notes that are too short to be meaningful
        /// 
        /// MUSICAL CONCEPTS:
        /// - Pitch: How high or low a note sounds (like C, D, E, F, G, A, B)
        /// - Timing: When a note starts and stops
        /// - MIDI Note Numbers: A standard way to represent pitches (C4 = 60)
        /// </summary>
        /// <param name="modelOutputs">Raw outputs from the ONNX model</param>
        /// <param name="audioLength">Length of original audio in samples</param>
        /// <param name="sampleRate">Sample rate of the audio</param>
        /// <returns>List of detected notes</returns>
        public static List<DetectedNote> ProcessModelOutputs(Dictionary<string, DenseTensor<float>> modelOutputs, int audioLength, int sampleRate)
        {
            // Create a list to store all the notes we find
            var notes = new List<DetectedNote>();
            
            // Basic Pitch model typically provides several types of information:
            // - note_activations: How confident it is that each note is playing
            // - onset_activations: How confident it is that a note is starting  
            // - contour_activations: Information about pitch bending/vibrato
            // 
            // Let's try to find the best output to use for note detection
            // Look for outputs with 88 pitches (piano keys) or 264 pitches (with harmonics)
            DenseTensor<float>? primaryOutput = null;
            string outputName = "";
            
            // Try to find the output with the most pitch information
            foreach (var output in modelOutputs)
            {
                var tensor = output.Value;
                Console.WriteLine($"Checking output '{output.Key}' with shape: {string.Join('x', tensor.Dimensions.ToArray())}");
                
                // Look for outputs that have pitch dimension (88 or 264)
                if (tensor.Dimensions.Length >= 2)
                {
                    var lastDim = tensor.Dimensions[tensor.Dimensions.Length - 1];
                    if (lastDim == 88 || lastDim == 264)
                    {
                        primaryOutput = tensor;
                        outputName = output.Key;
                        Console.WriteLine($"  -> Selected this output for note detection (has {lastDim} pitch classes)");
                        break;
                    }
                }
            }
            
            // Fallback to first output if no suitable one found
            if (primaryOutput == null)
            {
                primaryOutput = modelOutputs.First().Value;
                outputName = modelOutputs.First().Key;
                Console.WriteLine($"  -> Using fallback output: {outputName}");
            }
            
            Console.WriteLine($"Processing output tensor '{outputName}' with shape: {string.Join('x', primaryOutput.Dimensions.ToArray())}");
            
            // Figure out the structure of the model's output
            // The output is organized as [time_frames, pitch_classes, ...]
            // time_frames = how many time slices the audio was divided into
            // pitch_classes = how many different notes the model can detect
            int numFrames, numPitches;
            if (primaryOutput.Dimensions.Length == 3)
            {
                // 3D output: [frames, pitches, channels]
                numFrames = primaryOutput.Dimensions[0];
                numPitches = primaryOutput.Dimensions[1];
            }
            else if (primaryOutput.Dimensions.Length == 2)
            {
                // 2D output: [frames, pitches]
                numFrames = primaryOutput.Dimensions[0];
                numPitches = primaryOutput.Dimensions[1];
            }
            else
            {
                // Fallback for unexpected formats
                numFrames = primaryOutput.Dimensions[0];
                numPitches = Math.Min(264, primaryOutput.Dimensions.Length > 1 ? primaryOutput.Dimensions[1] : 1);
            }
            
            // Calculate how much real time each frame represents
            // If we have 1000 frames and 10 seconds of audio, each frame = 0.01 seconds
            float frameToTime = (float)audioLength / sampleRate / numFrames;
            
            // DEBUG: Track detection statistics
            int totalDetections = 0;
            float maxActivationSeen = 0.0f;
            int detectionsAboveThreshold = 0;
            
            // Look for notes by examining each possible pitch
            // For each pitch (like C, C#, D, etc.), check if it's active over time
            for (int pitch = 0; pitch < numPitches; pitch++)
            {
                // Store time frames where this pitch is detected with high confidence
                var pitchActivations = new List<(int frame, float activation)>();
                
                // Look at every time frame for this specific pitch
                for (int frame = 0; frame < numFrames; frame++)
                {
                    // Get the model's confidence that this pitch is active at this time
                    float activation;
                    if (primaryOutput.Dimensions.Length == 3)
                    {
                        // For 3D output, get the main channel (index 0)
                        activation = primaryOutput[frame, pitch, 0];
                    }
                    else if (primaryOutput.Dimensions.Length == 2)
                    {
                        // For 2D output, get the value directly
                        activation = primaryOutput[frame, pitch];
                    }
                    else
                    {
                        // Fallback for unexpected formats
                        activation = 0.0f;
                    }
                    
                    // DEBUG: Track activation statistics
                    totalDetections++;
                    if (activation > maxActivationSeen) maxActivationSeen = activation;
                    
                    // If the model is confident enough that this note is playing...
                    if (activation > FRAME_THRESHOLD)
                    {
                        // Remember this detection
                        pitchActivations.Add((frame, activation));
                        detectionsAboveThreshold++;
                    }
                }
                
                // Turn the detected moments into actual musical notes
                if (pitchActivations.Count > 0)
                {
                    // Group nearby detections into continuous notes
                    // If we detect C at frame 10, 11, 12, that's probably one C note
                    var noteSegments = GroupConsecutiveActivations(pitchActivations);
                    
                    // Convert each segment into a musical note
                    foreach (var segment in noteSegments)
                    {
                        // Calculate when this note starts and stops (in seconds)
                        float startTime = segment.startFrame * frameToTime;
                        float endTime = segment.endFrame * frameToTime;
                        float duration = endTime - startTime;
                        
                        // Ignore notes that are too short (probably just noise)
                        if (duration >= MIN_NOTE_LENGTH)
                        {
                            // Convert our internal pitch number to standard MIDI note number
                            // MIDI notes: C4=60, C#4=61, D4=62, etc.
                            int midiNote = PitchIndexToMidiNote(pitch);
                            
                            // Create a complete note with all its information
                            notes.Add(new DetectedNote
                            {
                                MidiNote = midiNote,           // Standard note number (60 = middle C)
                                StartTime = startTime,         // When note begins (seconds)
                                EndTime = endTime,             // When note ends (seconds)
                                Duration = duration,           // How long note lasts (seconds)
                                Confidence = segment.avgConfidence, // How sure we are about this note
                                Frequency = MidiNoteToFrequency(midiNote) // Frequency in Hz (440 = A4)
                            });
                        }
                    }
                }
            }
            
            // DEBUG: Print detection statistics
            Console.WriteLine($"DEBUG: Detection Statistics:");
            Console.WriteLine($"  Total predictions checked: {totalDetections}");
            Console.WriteLine($"  Maximum activation value seen: {maxActivationSeen:F4}");
            Console.WriteLine($"  Current threshold: {FRAME_THRESHOLD}");
            Console.WriteLine($"  Detections above threshold: {detectionsAboveThreshold}");
            Console.WriteLine($"  Final notes after filtering: {notes.Count}");
            
            // Arrange notes in chronological order (earliest first)
            // This makes the output easier to read and understand
            notes.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
            
            return notes;
        }

        /// <summary>
        /// Groups consecutive frame activations into note segments
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes individual detections that are close together in time
        /// - Groups them into continuous notes
        /// - Example: detections at frames 10,11,12,15,16 become two notes:
        ///   - Note 1: frames 10-12
        ///   - Note 2: frames 15-16
        /// 
        /// WHY WE NEED THIS:
        /// - The AI model analyzes audio in small time slices
        /// - A single musical note might be detected across multiple slices
        /// - We need to connect these detections into complete notes
        /// </summary>
        /// <param name="activations">List of frame activations for a specific pitch</param>
        /// <returns>List of note segments with start/end frames and confidence</returns>
        private static List<(int startFrame, int endFrame, float avgConfidence)> GroupConsecutiveActivations(
            List<(int frame, float activation)> activations)
        {
            // List to store the grouped note segments
            var segments = new List<(int startFrame, int endFrame, float avgConfidence)>();
            
            // If no detections, return empty list
            if (activations.Count == 0) return segments;
            
            // Start with the first detection
            int currentStart = activations[0].frame;      // When current note segment starts
            int currentEnd = activations[0].frame;        // When current note segment ends
            float confidenceSum = activations[0].activation; // Sum of confidence scores
            int confidenceCount = 1;                      // How many detections in this segment
            
            // Look at each detection after the first one
            for (int i = 1; i < activations.Count; i++)
            {
                // Check if this detection is right after the previous one
                if (activations[i].frame == currentEnd + 1)
                {
                    // Consecutive frame - this detection continues the current note
                    currentEnd = activations[i].frame;
                    confidenceSum += activations[i].activation;
                    confidenceCount++;
                }
                else
                {
                    // Gap found - the current note has ended, start a new one
                    // Save the completed note segment
                    segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
                    
                    // Start tracking a new note segment
                    currentStart = activations[i].frame;
                    currentEnd = activations[i].frame;
                    confidenceSum = activations[i].activation;
                    confidenceCount = 1;
                }
            }
            
            // Don't forget to add the final note segment
            segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
            
            return segments;
        }

        /// <summary>
        /// Converts a pitch index from the model output to a MIDI note number
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - The AI model uses its own numbering system for pitches
        /// - MIDI uses a standard numbering system (60 = middle C)
        /// - This function translates between the two systems
        /// 
        /// MIDI NOTE NUMBERS:
        /// - 21 = A0 (lowest piano key)
        /// - 60 = C4 (middle C)
        /// - 69 = A4 (440 Hz, tuning reference)
        /// - 108 = C8 (highest piano key)
        /// 
        /// BASIC PITCH MODEL:
        /// - Covers 88 piano keys starting from A0
        /// - Uses 3 frequency bins per semitone for fine-tuned detection
        /// </summary>
        /// <param name="pitchIndex">Index from model output</param>
        /// <returns>MIDI note number (0-127)</returns>
        private static int PitchIndexToMidiNote(int pitchIndex)
        {
            // Basic Pitch model covers the 88 keys of a piano, starting from A0 (MIDI note 21)
            // The model uses 3 detection bins per semitone for better accuracy
            // So we divide by 3 to get the actual semitone number
            int noteOffset = pitchIndex / N_BINS_PER_SEMITONE;
            
            // Add the offset to the starting note (A0 = 21) and make sure it's valid
            return Math.Max(0, Math.Min(127, 21 + noteOffset)); // Keep within MIDI range (0-127)
        }

        /// <summary>
        /// Converts a MIDI note number to frequency in Hz
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes a MIDI note number (like 60 for middle C)
        /// - Calculates the actual sound frequency in Hertz
        /// 
        /// FREQUENCY CONCEPTS:
        /// - Frequency = how many sound waves per second
        /// - Higher frequency = higher pitch
        /// - A4 (MIDI 69) = 440 Hz by musical convention
        /// - Each octave doubles the frequency (A5 = 880 Hz)
        /// - Each semitone multiplies by 2^(1/12) ≈ 1.059
        /// </summary>
        /// <param name="midiNote">MIDI note number</param>
        /// <returns>Frequency in Hz</returns>
        private static float MidiNoteToFrequency(int midiNote)
        {
            // Formula: freq = 440 * 2^((note - 69) / 12)
            // 440 = frequency of A4 (MIDI note 69)
            // Each semitone is 2^(1/12) times the previous frequency
            return 440.0f * (float)Math.Pow(2.0, (midiNote - 69) / 12.0);
        }
    }
}