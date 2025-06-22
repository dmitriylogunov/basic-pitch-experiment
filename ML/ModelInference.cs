using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BasicPitchExperimentApp.ML
{
    /// <summary>
    /// Static wrapper for ONNX model operations - maintains backward compatibility
    /// while using the new BasicPitchModel abstraction internally
    /// </summary>
    public static class ModelInference
    {
        // Constants for Basic Pitch model configuration
        private const int FFT_HOP = 256;
        private const int AUDIO_SAMPLE_RATE = 22050;
        private const int AUDIO_N_SAMPLES = 43844; // 2 seconds * 22050 - 256
        private const int ANNOTATIONS_FPS = 86; // AUDIO_SAMPLE_RATE / FFT_HOP
        private const int N_BINS_PER_SEMITONE = 3;
        private const float MIN_NOTE_LENGTH = 0.127f;
        private const float ONSET_THRESHOLD = 0.5f;
        private const float FRAME_THRESHOLD = 0.3f;
        private const int N_FREQ_BINS_NOTES = 88;
        private const int N_FREQ_BINS_CONTOURS = 264;
        private const int N_OVERLAPPING_FRAMES = 30; // Number of frames to overlap between windows
        private const int OVERLAP_LENGTH = N_OVERLAPPING_FRAMES * FFT_HOP; // 7680 samples
        private const int HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LENGTH; // 36164 samples

        /// <summary>
        /// Processes the entire audio file using sliding windows and returns all detected notes
        /// This method maintains backward compatibility while using the new model abstraction
        /// </summary>
        /// <param name="session">ONNX inference session</param>
        /// <param name="audioData">Complete audio data</param>
        /// <param name="sampleRate">Sample rate of the audio</param>
        /// <returns>List of all detected notes from the entire audio</returns>
        public static List<DetectedNote> ProcessFullAudio(InferenceSession session, float[] audioData, int sampleRate)
        {
            // Create a temporary BasicPitchModel instance using the existing session
            // Note: This is a compatibility layer - ideally, use BasicPitchModel directly
            var modelInput = new ModelInput
            {
                AudioData = audioData,
                SampleRate = sampleRate,
                Parameters = new InferenceParameters
                {
                    NoteThreshold = FRAME_THRESHOLD,
                    OnsetThreshold = ONSET_THRESHOLD,
                    MinNoteLength = MIN_NOTE_LENGTH,
                    OverlappingFrames = N_OVERLAPPING_FRAMES,
                    AutoApplySigmoid = true
                }
            };
            
            // For backward compatibility, we'll keep the original implementation
            // but show how it maps to the new abstraction
            var allNotes = new List<DetectedNote>();
            
            // Add padding at the beginning (half of overlap length)
            int halfOverlap = OVERLAP_LENGTH / 2;
            var paddedAudio = new float[audioData.Length + halfOverlap];
            Array.Copy(audioData, 0, paddedAudio, halfOverlap, audioData.Length);
            
            // Process audio in sliding windows
            int windowCount = 0;
            var allOutputs = new Dictionary<string, List<float[]>>
            {
                ["note"] = new List<float[]>(),
                ["onset"] = new List<float[]>(),
                ["contour"] = new List<float[]>()
            };
            
            Console.WriteLine($"Processing audio in {AUDIO_N_SAMPLES} sample windows with {HOP_SIZE} sample hops");
            
            for (int i = 0; i < paddedAudio.Length; i += HOP_SIZE)
            {
                // Extract window
                var window = new float[AUDIO_N_SAMPLES];
                for (int j = 0; j < AUDIO_N_SAMPLES; j++)
                {
                    if (i + j < paddedAudio.Length)
                    {
                        window[j] = paddedAudio[i + j];
                    }
                    else
                    {
                        window[j] = 0.0f; // Zero-pad if we run out of audio
                    }
                }
                
                // Preprocess window
                var windowTensor = PreprocessAudioWindow(window);
                
                // Run inference on this window
                var windowOutputs = RunInference(session, windowTensor);
                
                // Store outputs for later unwrapping
                foreach (var kvp in windowOutputs)
                {
                    // Convert tensor to array for storage
                    var tensor = kvp.Value;
                    var array = new float[tensor.Length];
                    tensor.Buffer.Span.CopyTo(array);
                    allOutputs[kvp.Key].Add(array);
                }
                
                windowCount++;
                if (windowCount % 10 == 0)
                {
                    Console.WriteLine($"  Processed {windowCount} windows...");
                }
            }
            
            Console.WriteLine($"Total windows processed: {windowCount}");
            
            // Unwrap and merge outputs from all windows
            var mergedOutputs = UnwrapOutputs(allOutputs, audioData.Length, N_OVERLAPPING_FRAMES);
            
            // Process the merged outputs to extract notes
            return ProcessModelOutputs(mergedOutputs, audioData.Length, sampleRate);
        }
        
        /// <summary>
        /// Preprocesses a single audio window to match the expected model input shape
        /// </summary>
        /// <param name="audioWindow">Audio samples for one window</param>
        /// <returns>Tensor containing the preprocessed audio data</returns>
        private static DenseTensor<float> PreprocessAudioWindow(float[] audioWindow)
        {
            // Create tensor with shape [1, 43844, 1]
            var tensor = new DenseTensor<float>(new[] { 1, AUDIO_N_SAMPLES, 1 });
            
            // Fill the tensor with audio data
            for (int i = 0; i < AUDIO_N_SAMPLES; i++)
            {
                tensor[0, i, 0] = audioWindow[i];
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
            // Basic Pitch ONNX model expects input named "serving_default_input_2:0"
            var inputName = "serving_default_input_2:0";
            if (!session.InputMetadata.ContainsKey(inputName))
            {
                // Fallback to first input name if specific name not found
                inputName = session.InputMetadata.Keys.First();
                Console.WriteLine($"Warning: Expected input name 'serving_default_input_2:0' not found, using '{inputName}'");
            }
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
            
            // Basic Pitch ONNX model has specific output names
            var outputNames = new[] {
                "StatefulPartitionedCall:1", // note
                "StatefulPartitionedCall:2", // onset  
                "StatefulPartitionedCall:0"  // contour
            };
            
            // Run inference with specific output names
            using var results = session.Run(inputs, outputNames);
            
            // Map outputs to their semantic names
            var outputs = new Dictionary<string, DenseTensor<float>>();
            var outputMapping = new Dictionary<string, string>
            {
                ["StatefulPartitionedCall:1"] = "note",
                ["StatefulPartitionedCall:2"] = "onset",
                ["StatefulPartitionedCall:0"] = "contour"
            };
            
            foreach (var result in results)
            {
                if (result.Value is Tensor<float> tensor)
                {
                    var semanticName = outputMapping.ContainsKey(result.Name) ? outputMapping[result.Name] : result.Name;
                    outputs[semanticName] = tensor.ToDenseTensor();
                    Console.WriteLine($"Output '{semanticName}' (tensor: {result.Name}): {string.Join('x', tensor.Dimensions.ToArray())}");
                    
                    // DEBUG: Show value statistics
                    var denseTensor = tensor.ToDenseTensor();
                    var values = denseTensor.ToArray();
                    var maxVal = values.Max();
                    var minVal = values.Min();
                    var avgVal = values.Average();
                    var aboveThreshold = values.Count(v => v > FRAME_THRESHOLD);
                    Console.WriteLine($"  -> Range: [{minVal:F4}, {maxVal:F4}], Mean: {avgVal:F4}, Values > {FRAME_THRESHOLD}: {aboveThreshold}/{values.Length}");
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
            
            // Basic Pitch model outputs:
            // - note: Note activations (shape: [frames, 88])
            // - onset: Onset activations (shape: [frames, 88])  
            // - contour: Pitch contour (shape: [frames, 264])
            
            // Use the 'note' output for primary note detection
            if (!modelOutputs.ContainsKey("note"))
            {
                Console.WriteLine("ERROR: 'note' output not found in model outputs!");
                Console.WriteLine($"Available outputs: {string.Join(", ", modelOutputs.Keys)}");
                return notes;
            }
            
            var noteOutput = modelOutputs["note"];
            var onsetOutput = modelOutputs.ContainsKey("onset") ? modelOutputs["onset"] : null;
            
            Console.WriteLine($"Using 'note' output with shape: {string.Join('x', noteOutput.Dimensions.ToArray())}");
            if (onsetOutput != null)
            {
                Console.WriteLine($"Also using 'onset' output with shape: {string.Join('x', onsetOutput.Dimensions.ToArray())}");
            }
            
            // Basic Pitch outputs are 2D: [frames, pitches]
            // Note output: [frames, 88] - one value per piano key
            // Onset output: [frames, 88] - onset detection per piano key
            int numFrames = noteOutput.Dimensions[0];
            int numPitches = noteOutput.Dimensions[1];
            
            Console.WriteLine($"Processing {numFrames} time frames with {numPitches} pitch classes");
            
            // Calculate timing: Basic Pitch uses fixed frame rate
            // Each frame represents 1/ANNOTATIONS_FPS seconds
            float frameToTime = 1.0f / ANNOTATIONS_FPS;
            
            // Calculate actual number of frames for the original audio
            int expectedFrames = (int)Math.Floor(audioLength * (ANNOTATIONS_FPS / (float)sampleRate));
            Console.WriteLine($"Frame timing: {frameToTime:F4} seconds per frame");
            Console.WriteLine($"Expected frames for audio length: {expectedFrames}, actual frames: {numFrames}");
            
            // DEBUG: Track detection statistics
            int totalDetections = 0;
            float maxActivationSeen = 0.0f;
            float minActivationSeen = float.MaxValue;
            int detectionsAboveThreshold = 0;
            bool needsSigmoid = false;
            
            // Look for notes by examining each possible pitch
            // For each pitch (like C, C#, D, etc.), check if it's active over time
            for (int pitch = 0; pitch < numPitches; pitch++)
            {
                // Store time frames where this pitch is detected with high confidence
                var pitchActivations = new List<(int frame, float activation)>();
                
                // Look at every time frame for this specific pitch
                for (int frame = 0; frame < numFrames; frame++)
                {
                    // Get note activation value (2D tensor: [frame, pitch])
                    float rawActivation = noteOutput[frame, pitch];
                    float activation = rawActivation;
                    
                    // Track if we see values outside [0,1] - indicates need for sigmoid
                    if (rawActivation < 0 || rawActivation > 1)
                    {
                        needsSigmoid = true;
                    }
                    
                    // IMPORTANT: Check if activation needs sigmoid transformation
                    // If values are outside [0,1] range, they likely need sigmoid
                    if (rawActivation < -1 || rawActivation > 1)
                    {
                        // Apply sigmoid: 1 / (1 + e^(-x))
                        activation = 1.0f / (1.0f + (float)Math.Exp(-rawActivation));
                    }
                    
                    // Get onset activation if available
                    float onsetActivation = 0.0f;
                    if (onsetOutput != null && frame < onsetOutput.Dimensions[0] && pitch < onsetOutput.Dimensions[1])
                    {
                        onsetActivation = onsetOutput[frame, pitch];
                        // Apply sigmoid if needed
                        if (onsetActivation < 0 || onsetActivation > 1)
                        {
                            onsetActivation = 1.0f / (1.0f + (float)Math.Exp(-onsetActivation));
                        }
                    }
                    
                    // DEBUG: Track activation statistics
                    totalDetections++;
                    if (activation > maxActivationSeen) maxActivationSeen = activation;
                    if (activation < minActivationSeen) minActivationSeen = activation;
                    
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
            Console.WriteLine($"\nDEBUG: Detection Statistics:");
            Console.WriteLine($"  Total predictions checked: {totalDetections}");
            Console.WriteLine($"  Activation value range: [{minActivationSeen:F4}, {maxActivationSeen:F4}]");
            Console.WriteLine($"  Values outside [0,1] range: {(needsSigmoid ? "YES - sigmoid applied" : "NO")}");
            Console.WriteLine($"  Current threshold: {FRAME_THRESHOLD}");
            Console.WriteLine($"  Detections above threshold: {detectionsAboveThreshold}");
            Console.WriteLine($"  Final notes after filtering: {notes.Count}");
            
            // Suggest adjustments if no notes detected
            if (notes.Count == 0 && maxActivationSeen < FRAME_THRESHOLD)
            {
                Console.WriteLine($"\nSUGGESTION: Maximum activation ({maxActivationSeen:F4}) is below threshold ({FRAME_THRESHOLD})");
                Console.WriteLine($"Try reducing threshold to {maxActivationSeen * 0.8f:F4}");
            }
            
            // Print final statistics
            if (notes.Count == 0)
            {
                Console.WriteLine("\nWARNING: No notes detected! Possible issues:");
                Console.WriteLine("1. Activation values might need sigmoid transformation");
                Console.WriteLine("2. Threshold values might be too high");
                Console.WriteLine("3. Audio preprocessing might not match expected format");
                Console.WriteLine($"\nTry reducing FRAME_THRESHOLD from {FRAME_THRESHOLD} to a lower value like 0.1");
            }
            else
            {
                Console.WriteLine($"\nSuccessfully detected {notes.Count} notes");
                // Show first few notes for debugging
                int showCount = Math.Min(5, notes.Count);
                Console.WriteLine($"First {showCount} notes:");
                for (int i = 0; i < showCount; i++)
                {
                    var note = notes[i];
                    Console.WriteLine($"  - MIDI {note.MidiNote} ({NoteUtils.GetNoteName(note.MidiNote)}), " +
                                    $"Time: {note.StartTime:F2}-{note.EndTime:F2}s, " +
                                    $"Confidence: {note.Confidence:F3}");
                }
            }
            
            // Sort notes by start time
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
            // Basic Pitch 'note' output has 88 bins, one per piano key
            // Starting from A0 (MIDI note 21) to C8 (MIDI note 108)
            // No division needed - direct mapping
            return Math.Max(0, Math.Min(127, 21 + pitchIndex));
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
        
        /// <summary>
        /// Unwraps and merges outputs from multiple overlapping windows
        /// </summary>
        /// <param name="allOutputs">Dictionary containing lists of outputs from each window</param>
        /// <param name="audioOriginalLength">Original audio length in samples</param>
        /// <param name="nOverlappingFrames">Number of overlapping frames between windows</param>
        /// <returns>Merged outputs as tensors</returns>
        private static Dictionary<string, DenseTensor<float>> UnwrapOutputs(
            Dictionary<string, List<float[]>> allOutputs, 
            int audioOriginalLength, 
            int nOverlappingFrames)
        {
            var mergedOutputs = new Dictionary<string, DenseTensor<float>>();
            int nOlap = nOverlappingFrames / 2;
            
            foreach (var kvp in allOutputs)
            {
                var outputName = kvp.Key;
                var windowOutputs = kvp.Value;
                
                if (windowOutputs.Count == 0) continue;
                
                // Determine dimensions from first window
                int numFeatures = outputName == "contour" ? N_FREQ_BINS_CONTOURS : N_FREQ_BINS_NOTES;
                
                // Calculate total frames after removing overlaps
                var processedFrames = new List<float[]>();
                
                for (int w = 0; w < windowOutputs.Count; w++)
                {
                    var windowData = windowOutputs[w];
                    int windowFrames = windowData.Length / numFeatures;
                    
                    // For each frame in this window
                    for (int f = 0; f < windowFrames; f++)
                    {
                        // Skip overlapping frames at beginning and end (except for first/last windows)
                        bool skipFrame = false;
                        if (w > 0 && f < nOlap) skipFrame = true;
                        if (w < windowOutputs.Count - 1 && f >= windowFrames - nOlap) skipFrame = true;
                        
                        if (!skipFrame)
                        {
                            // Extract this frame's data
                            var frameData = new float[numFeatures];
                            for (int feat = 0; feat < numFeatures; feat++)
                            {
                                frameData[feat] = windowData[f * numFeatures + feat];
                            }
                            processedFrames.Add(frameData);
                        }
                    }
                }
                
                // Calculate expected number of frames for original audio length
                int nOutputFramesOriginal = (int)Math.Floor(audioOriginalLength * (ANNOTATIONS_FPS / (float)AUDIO_SAMPLE_RATE));
                
                // Create merged tensor and copy data
                int actualFrames = Math.Min(processedFrames.Count, nOutputFramesOriginal);
                var mergedTensor = new DenseTensor<float>(new[] { actualFrames, numFeatures });
                
                for (int f = 0; f < actualFrames; f++)
                {
                    for (int feat = 0; feat < numFeatures; feat++)
                    {
                        mergedTensor[f, feat] = processedFrames[f][feat];
                    }
                }
                
                mergedOutputs[outputName] = mergedTensor;
                Console.WriteLine($"Merged {outputName} output: {actualFrames} frames x {numFeatures} features");
            }
            
            return mergedOutputs;
        }
    }
}