using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace BasicPitchExperimentApp.ML
{
    /// <summary>
    /// Implementation of the Basic Pitch ONNX model for music transcription
    /// </summary>
    public class BasicPitchModel : IModelInference, IDisposable
    {
        // Model constants
        private const int FFT_HOP = 256;
        private const int MODEL_SAMPLE_RATE = 22050;
        private const int AUDIO_WINDOW_LENGTH = 43844; // ~2 seconds at 22050 Hz
        private const int ANNOTATIONS_FPS = 86; // MODEL_SAMPLE_RATE / FFT_HOP
        private const int PIANO_KEYS = 88;
        private const int CONTOUR_BINS = 264;
        
        // Model I/O names
        private const string INPUT_NAME = "serving_default_input_2:0";
        private const string OUTPUT_NOTE = "StatefulPartitionedCall:1";
        private const string OUTPUT_ONSET = "StatefulPartitionedCall:2";
        private const string OUTPUT_CONTOUR = "StatefulPartitionedCall:0";

        private readonly InferenceSession _session;
        private bool _disposed;

        /// <summary>
        /// Creates a new Basic Pitch model instance
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file</param>
        public BasicPitchModel(string modelPath)
        {
            _session = new InferenceSession(modelPath);
            ValidateModel();
        }

        /// <summary>
        /// Creates a new Basic Pitch model instance with custom session options
        /// </summary>
        /// <param name="modelPath">Path to the ONNX model file</param>
        /// <param name="sessionOptions">ONNX Runtime session options</param>
        public BasicPitchModel(string modelPath, SessionOptions sessionOptions)
        {
            _session = new InferenceSession(modelPath, sessionOptions);
            ValidateModel();
        }

        /// <summary>
        /// Processes audio data and returns detected musical notes
        /// </summary>
        public ModelOutput ProcessAudio(ModelInput input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.AudioData == null) throw new ArgumentNullException(nameof(input.AudioData));
            if (input.SampleRate != MODEL_SAMPLE_RATE)
            {
                throw new ArgumentException($"Audio must be resampled to {MODEL_SAMPLE_RATE} Hz. Current: {input.SampleRate} Hz");
            }

            Console.WriteLine($"Processing audio: {input.AudioData.Length} samples ({input.AudioData.Length / (float)MODEL_SAMPLE_RATE:F2} seconds)");
            var stopwatch = Stopwatch.StartNew();
            var output = new ModelOutput
            {
                FrameRate = ANNOTATIONS_FPS,
                Statistics = new InferenceStatistics()
            };

            try
            {
                // Process audio in sliding windows
                var windowResults = ProcessAudioWindows(input.AudioData, input.Parameters);
                
                // Merge results from all windows
                var mergedResults = MergeWindowResults(windowResults, input.AudioData.Length);
            
            // Store raw activations
            output.NoteActivations = mergedResults.NoteActivations;
            output.OnsetActivations = mergedResults.OnsetActivations;
            output.PitchContour = mergedResults.PitchContour;
            output.FrameCount = mergedResults.FrameCount;
            
            // Detect notes from activations
            output.Notes = DetectNotes(mergedResults, input.Parameters);
            
            // Update statistics
            output.Statistics.WindowsProcessed = windowResults.Count;
            output.Statistics.MaxActivation = mergedResults.MaxActivation;
            output.Statistics.MinActivation = mergedResults.MinActivation;
            output.Statistics.AverageActivation = mergedResults.AverageActivation;
            output.Statistics.SigmoidApplied = mergedResults.SigmoidApplied;
            output.Statistics.DetectionsAboveThreshold = mergedResults.DetectionsAboveThreshold;
                output.Statistics.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;

                // Detect tempo from note onsets
                output.DetectedTempo = DetectTempo(output.Notes);
                Console.WriteLine($"Detected tempo: {output.DetectedTempo} BPM");

                return output;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in ProcessAudio: {ex.GetType().Name} - {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        private void ValidateModel()
        {
            // Validate input
            var inputName = _session.InputMetadata.ContainsKey(INPUT_NAME) 
                ? INPUT_NAME 
                : _session.InputMetadata.Keys.First();
                
            var inputMeta = _session.InputMetadata[inputName];
            if (inputMeta.Dimensions[1] != AUDIO_WINDOW_LENGTH)
            {
                throw new InvalidOperationException($"Model expects input length {AUDIO_WINDOW_LENGTH}, but found {inputMeta.Dimensions[1]}");
            }

            // Validate outputs
            var outputNames = _session.OutputMetadata.Keys.ToHashSet();
            if (!outputNames.Contains(OUTPUT_NOTE) || !outputNames.Contains(OUTPUT_ONSET) || !outputNames.Contains(OUTPUT_CONTOUR))
            {
                throw new InvalidOperationException("Model outputs do not match expected Basic Pitch format");
            }
        }

        private List<WindowResult> ProcessAudioWindows(float[] audioData, InferenceParameters parameters)
        {
            var results = new List<WindowResult>();
            try
            {
                checked
                {
                    Console.WriteLine($"DEBUG: parameters.OverlappingFrames = {parameters.OverlappingFrames}");
                    Console.WriteLine($"DEBUG: FFT_HOP = {FFT_HOP}");
                    
                    int overlapLength = parameters.OverlappingFrames * FFT_HOP;
                    Console.WriteLine($"DEBUG: overlapLength = {overlapLength}");
                    
                    int hopSize = AUDIO_WINDOW_LENGTH - overlapLength;
                    Console.WriteLine($"DEBUG: AUDIO_WINDOW_LENGTH = {AUDIO_WINDOW_LENGTH}, hopSize = {hopSize}");
                    
                    if (hopSize <= 0)
                    {
                        throw new ArgumentException($"Invalid hop size: {hopSize}. AUDIO_WINDOW_LENGTH ({AUDIO_WINDOW_LENGTH}) must be greater than overlapLength ({overlapLength})");
                    }
                    
                    int halfOverlap = overlapLength / 2;
                    Console.WriteLine($"DEBUG: halfOverlap = {halfOverlap}");
                
                Console.WriteLine($"Window processing: overlapLength={overlapLength}, hopSize={hopSize}, halfOverlap={halfOverlap}");
                Console.WriteLine($"Audio length: {audioData.Length}, padding size: {halfOverlap}");
                
                // Check for potential overflow
                if (audioData.Length > int.MaxValue - halfOverlap)
                {
                    throw new OverflowException($"Audio too large: {audioData.Length} + {halfOverlap} would overflow");
                }
                
                // Add padding at the beginning
                var paddedAudio = new float[audioData.Length + halfOverlap];
                Array.Copy(audioData, 0, paddedAudio, halfOverlap, audioData.Length);
                
                // Process windows
                int windowIndex = 0;
                Console.WriteLine($"DEBUG: Starting window processing. paddedAudio.Length = {paddedAudio.Length}, hopSize = {hopSize}");
                
            for (int i = 0; i < paddedAudio.Length; )
            {
                Console.WriteLine($"DEBUG: Processing window {windowIndex} at position i = {i}");
                
                // Extract window
                var window = new float[AUDIO_WINDOW_LENGTH];
                int windowLength = Math.Min(AUDIO_WINDOW_LENGTH, paddedAudio.Length - i);
                for (int j = 0; j < windowLength; j++)
                {
                    window[j] = paddedAudio[i + j];
                }
                
                // Run inference
                var windowResult = ProcessSingleWindow(window, parameters);
                windowResult.StartSample = i;
                results.Add(windowResult);
                
                windowIndex++;
                if (windowIndex % 10 == 0)
                {
                    Console.WriteLine($"Processed {windowIndex} windows...");
                }
                
                // Stop if we've processed all audio
                if (i + AUDIO_WINDOW_LENGTH >= paddedAudio.Length) break;
                
                // Safe increment to avoid overflow
                if (i > int.MaxValue - hopSize)
                {
                    break;
                }
                i += hopSize;
            }
                
                return results;
                }
            }
            catch (OverflowException ex)
            {
                Console.WriteLine($"Overflow in ProcessAudioWindows: {ex.Message}");
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in ProcessAudioWindows: {ex.GetType().Name} - {ex.Message}");
                throw;
            }
        }

        private WindowResult ProcessSingleWindow(float[] audioWindow, InferenceParameters parameters)
        {
            // Create input tensor [1, 43844, 1]
            var inputTensor = new DenseTensor<float>(new[] { 1, AUDIO_WINDOW_LENGTH, 1 });
            for (int i = 0; i < AUDIO_WINDOW_LENGTH; i++)
            {
                inputTensor[0, i, 0] = audioWindow[i];
            }
            
            // Prepare inputs
            var inputName = _session.InputMetadata.ContainsKey(INPUT_NAME) 
                ? INPUT_NAME 
                : _session.InputMetadata.Keys.First();
                
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            
            // Run inference
            var outputNames = new[] { OUTPUT_NOTE, OUTPUT_ONSET, OUTPUT_CONTOUR };
            using var results = _session.Run(inputs, outputNames);
            
            // Extract outputs
            var windowResult = new WindowResult();
            foreach (var result in results)
            {
                if (result.Value is Tensor<float> tensor)
                {
                    var values = tensor.ToArray();
                    
                    switch (result.Name)
                    {
                        case OUTPUT_NOTE:
                            windowResult.NoteActivations = values;
                            windowResult.FrameCount = tensor.Dimensions[1]; // Second dimension is frames
                            windowResult.KeyCount = tensor.Dimensions[2]; // Third dimension is keys
                            // Debug: Log tensor dimensions
                            Console.WriteLine($"Note tensor dimensions: [{string.Join(" x ", tensor.Dimensions.ToArray())}], array length: {values.Length}");
                            break;
                        case OUTPUT_ONSET:
                            windowResult.OnsetActivations = values;
                            Console.WriteLine($"Onset tensor dimensions: [{string.Join(" x ", tensor.Dimensions.ToArray())}], array length: {values.Length}");
                            break;
                        case OUTPUT_CONTOUR:
                            windowResult.ContourActivations = values;
                            if (tensor.Dimensions.Length >= 3)
                            {
                                windowResult.ContourBinCount = tensor.Dimensions[2]; // Third dimension is bins
                            }
                            Console.WriteLine($"Contour tensor dimensions: [{string.Join(" x ", tensor.Dimensions.ToArray())}], array length: {values.Length}");
                            break;
                    }
                    
                    // Check if sigmoid is needed
                    if (parameters.AutoApplySigmoid && values.Any(v => v < 0 || v > 1))
                    {
                        windowResult.NeedsSigmoid = true;
                    }
                }
            }
            
            return windowResult;
        }

        private MergedResults MergeWindowResults(List<WindowResult> windowResults, int originalAudioLength)
        {
            if (windowResults.Count == 0)
                return new MergedResults();
                
            // Debug window data structure
            Console.WriteLine($"DEBUG: MergeWindowResults - windowResults.Count = {windowResults.Count}");
            
            var firstWindow = windowResults[0];
            Console.WriteLine($"DEBUG: First window - FrameCount:{firstWindow.FrameCount}, " +
                $"NoteActivations length:{firstWindow.NoteActivations?.Length ?? 0}, " +
                $"OnsetActivations length:{firstWindow.OnsetActivations?.Length ?? 0}, " +
                $"ContourActivations length:{firstWindow.ContourActivations?.Length ?? 0}");
                
            int framesPerWindow = firstWindow.FrameCount;
            int overlapFrames = 15; // Half of the overlapping frames parameter
            
            Console.WriteLine($"Merging windows: count={windowResults.Count}, framesPerWindow={framesPerWindow}");
            
            // Calculate total frames using long to avoid overflow
            long totalFramesLong = 0;
            for (int i = 0; i < windowResults.Count; i++)
            {
                if (i == 0)
                    totalFramesLong += framesPerWindow - overlapFrames;
                else if (i == windowResults.Count - 1)
                    totalFramesLong += framesPerWindow - overlapFrames;
                else
                    totalFramesLong += framesPerWindow - 2 * overlapFrames;
            }
            
            if (totalFramesLong > int.MaxValue)
            {
                throw new OverflowException($"Total frames {totalFramesLong} exceeds int.MaxValue");
            }
            
            int totalFrames = (int)totalFramesLong;
            
            // Limit to expected frames for original audio
            // Use double arithmetic to avoid overflow
            double framesPerSecond = (double)ANNOTATIONS_FPS / MODEL_SAMPLE_RATE * originalAudioLength;
            int expectedFrames = (int)Math.Min(framesPerSecond, int.MaxValue);
            totalFrames = Math.Min(totalFrames, expectedFrames);
            
            Console.WriteLine($"Merging {windowResults.Count} windows: totalFrames={totalFrames}, expectedFrames={expectedFrames}");
            
            // Initialize merged arrays
            var merged = new MergedResults
            {
                NoteActivations = new float[totalFrames, PIANO_KEYS],
                OnsetActivations = new float[totalFrames, PIANO_KEYS],
                PitchContour = new float[totalFrames, CONTOUR_BINS],
                FrameCount = totalFrames
            };
            
            // Merge windows
            int currentFrame = 0;
            for (int w = 0; w < windowResults.Count; w++)
            {
                var window = windowResults[w];
                int startFrame = (w == 0) ? 0 : overlapFrames;
                int endFrame = (w == windowResults.Count - 1) ? window.FrameCount : window.FrameCount - overlapFrames;
                
                // Copy frames
                for (int f = startFrame; f < endFrame && currentFrame < totalFrames; f++)
                {
                    // Copy note activations
                    int keysPerFrame = window.KeyCount > 0 ? window.KeyCount : PIANO_KEYS;
                    for (int k = 0; k < PIANO_KEYS && k < keysPerFrame; k++)
                    {
                        // Calculate index safely
                        long indexLong = (long)f * keysPerFrame + k;
                        
                        // Debug first iteration
                        if (w == 0 && f == startFrame && k == 0)
                        {
                            Console.WriteLine($"DEBUG: First note copy - window:{w}, frame:{f}, key:{k}, keysPerFrame:{keysPerFrame}, indexLong:{indexLong}, array length:{window.NoteActivations?.Length ?? 0}");
                        }
                        
                        if (window.NoteActivations == null || indexLong >= window.NoteActivations.Length)
                        {
                            Console.WriteLine($"ERROR: Note index out of bounds - frame:{f}, key:{k}, index:{indexLong}, array length:{window.NoteActivations?.Length ?? 0}");
                            continue;
                        }
                        int index = (int)indexLong;
                        
                        float value = window.NoteActivations[index];
                        if (window.NeedsSigmoid && (value < 0 || value > 1))
                        {
                            value = Sigmoid(value);
                            merged.SigmoidApplied = true;
                        }
                        merged.NoteActivations[currentFrame, k] = value;
                        UpdateStatistics(merged, value);
                    }
                    
                    // Copy onset activations
                    if (window.OnsetActivations != null && window.OnsetActivations.Length > 0)
                    {
                        for (int k = 0; k < PIANO_KEYS && k < keysPerFrame; k++)
                        {
                            long indexLong = (long)f * keysPerFrame + k;
                            if (indexLong >= window.OnsetActivations.Length)
                            {
                                continue;
                            }
                            int index = (int)indexLong;
                            
                            float value = window.OnsetActivations[index];
                            if (window.NeedsSigmoid && (value < 0 || value > 1))
                            {
                                value = Sigmoid(value);
                            }
                            merged.OnsetActivations[currentFrame, k] = value;
                        }
                    }
                    
                    // Copy contour
                    if (window.ContourActivations != null && window.ContourActivations.Length > 0)
                    {
                        int binsPerFrame = window.ContourBinCount > 0 ? window.ContourBinCount : CONTOUR_BINS;
                        for (int b = 0; b < CONTOUR_BINS && b < binsPerFrame; b++)
                        {
                            long indexLong = (long)f * binsPerFrame + b;
                            if (indexLong >= window.ContourActivations.Length)
                            {
                                continue;
                            }
                            int index = (int)indexLong;
                            
                            float value = window.ContourActivations[index];
                            if (window.NeedsSigmoid && (value < 0 || value > 1))
                            {
                                value = Sigmoid(value);
                            }
                            merged.PitchContour[currentFrame, b] = value;
                        }
                    }
                    
                    currentFrame++;
                }
            }
            
            // Calculate average
            if (merged.ActivationCount > 0)
            {
                merged.AverageActivation = merged.ActivationSum / merged.ActivationCount;
            }
            
            return merged;
        }

        private List<DetectedNote> DetectNotes(MergedResults mergedResults, InferenceParameters parameters)
        {
            var notes = new List<DetectedNote>();
            
            for (int pitch = 0; pitch < PIANO_KEYS; pitch++)
            {
                var activations = new List<(int frame, float activation)>();
                var onsets = new List<(int frame, float onset)>();
                
                // Find frames where this pitch is active and detect onsets
                for (int frame = 0; frame < mergedResults.FrameCount; frame++)
                {
                    float activation = mergedResults.NoteActivations[frame, pitch];
                    if (activation > parameters.NoteThreshold)
                    {
                        activations.Add((frame, activation));
                        mergedResults.DetectionsAboveThreshold++;
                    }
                    
                    // Collect onset information if available
                    if (mergedResults.OnsetActivations != null && mergedResults.OnsetActivations.GetLength(0) > frame)
                    {
                        float onset = mergedResults.OnsetActivations[frame, pitch];
                        if (onset > parameters.OnsetThreshold)
                        {
                            onsets.Add((frame, onset));
                        }
                    }
                }
                
                // Group consecutive activations, considering onsets for segmentation
                var segments = GroupConsecutiveActivationsWithOnsets(activations, onsets, parameters);
                
                // Log onset usage for debugging
                if (onsets.Count > 0 && segments.Count > 1)
                {
                    int midiNote = 21 + pitch;
                    Console.WriteLine($"Note {GetNoteName(midiNote)} (MIDI {midiNote}): {onsets.Count} onsets detected, {segments.Count} segments created");
                }
                
                // Convert segments to notes
                foreach (var segment in segments)
                {
                    float startTime = segment.startFrame / (float)ANNOTATIONS_FPS;
                    float endTime = segment.endFrame / (float)ANNOTATIONS_FPS;
                    float duration = endTime - startTime;
                    
                    if (duration >= parameters.MinNoteLength)
                    {
                        int midiNote = 21 + pitch; // A0 = MIDI 21
                        notes.Add(new DetectedNote
                        {
                            MidiNote = midiNote,
                            StartTime = startTime,
                            EndTime = endTime,
                            Duration = duration,
                            Confidence = segment.avgConfidence,
                            Frequency = MidiNoteToFrequency(midiNote)
                        });
                    }
                }
            }
            
            // Sort by start time
            notes.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
            return notes;
        }

        private List<(int startFrame, int endFrame, float avgConfidence)> GroupConsecutiveActivationsWithOnsets(
            List<(int frame, float activation)> activations,
            List<(int frame, float onset)> onsets,
            InferenceParameters parameters)
        {
            if (activations.Count == 0)
                return new List<(int, int, float)>();
            
            var segments = new List<(int startFrame, int endFrame, float avgConfidence)>();
            int currentStart = activations[0].frame;
            int currentEnd = activations[0].frame;
            float confidenceSum = activations[0].activation;
            int confidenceCount = 1;
            
            // Create a HashSet of onset frames for quick lookup
            var onsetFrames = new HashSet<int>(onsets.Select(o => o.frame));
            
            for (int i = 1; i < activations.Count; i++)
            {
                bool shouldSplit = false;
                
                // Check if there's a gap in activation
                if (activations[i].frame != currentEnd + 1)
                {
                    shouldSplit = true;
                }
                // Check if there's an onset within the continuous activation
                else if (parameters.UseOnsetForNoteSplitting && onsetFrames.Count > 0)
                {
                    // Look for onsets between the current segment start and this frame
                    // Allow a small window around the onset for timing variations
                    for (int checkFrame = currentEnd; checkFrame <= activations[i].frame + 2; checkFrame++)
                    {
                        if (onsetFrames.Contains(checkFrame) && checkFrame > currentStart + parameters.MinFramesBetweenOnsets)
                        {
                            // Found an onset within continuous activation, split here
                            shouldSplit = true;
                            Console.WriteLine($"Splitting note at onset frame {checkFrame} (segment started at {currentStart})");
                            break;
                        }
                    }
                }
                
                if (shouldSplit)
                {
                    // End current segment
                    segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
                    
                    // Start new segment
                    currentStart = activations[i].frame;
                    currentEnd = activations[i].frame;
                    confidenceSum = activations[i].activation;
                    confidenceCount = 1;
                }
                else
                {
                    // Continue current segment
                    currentEnd = activations[i].frame;
                    confidenceSum += activations[i].activation;
                    confidenceCount++;
                }
            }
            
            // Add final segment
            segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
            return segments;
        }

        private List<(int startFrame, int endFrame, float avgConfidence)> GroupConsecutiveActivations(
            List<(int frame, float activation)> activations)
        {
            var segments = new List<(int startFrame, int endFrame, float avgConfidence)>();
            if (activations.Count == 0) return segments;
            
            int currentStart = activations[0].frame;
            int currentEnd = activations[0].frame;
            float confidenceSum = activations[0].activation;
            int confidenceCount = 1;
            
            for (int i = 1; i < activations.Count; i++)
            {
                if (activations[i].frame == currentEnd + 1)
                {
                    currentEnd = activations[i].frame;
                    confidenceSum += activations[i].activation;
                    confidenceCount++;
                }
                else
                {
                    segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
                    currentStart = activations[i].frame;
                    currentEnd = activations[i].frame;
                    confidenceSum = activations[i].activation;
                    confidenceCount = 1;
                }
            }
            
            segments.Add((currentStart, currentEnd, confidenceSum / confidenceCount));
            return segments;
        }

        private static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        private static float MidiNoteToFrequency(int midiNote)
        {
            return 440.0f * (float)Math.Pow(2.0, (midiNote - 69) / 12.0);
        }
        
        private static string GetNoteName(int midiNote)
        {
            string[] noteNames = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
            int octave = (midiNote / 12) - 1;
            int noteIndex = midiNote % 12;
            return $"{noteNames[noteIndex]}{octave}";
        }

        private static void UpdateStatistics(MergedResults merged, float value)
        {
            if (value > merged.MaxActivation) merged.MaxActivation = value;
            if (value < merged.MinActivation) merged.MinActivation = value;
            merged.ActivationSum += value;
            merged.ActivationCount++;
        }

        private static int DetectTempo(List<DetectedNote> notes)
        {
            if (notes.Count < 2)
                return 120; // Default tempo
            
            // Get all note onset times
            var onsetTimes = notes.Select(n => n.StartTime).OrderBy(t => t).ToList();
            
            // Calculate intervals between consecutive onsets
            var intervals = new List<float>();
            for (int i = 1; i < onsetTimes.Count; i++)
            {
                float interval = onsetTimes[i] - onsetTimes[i - 1];
                if (interval > 0.05f && interval < 2.0f) // Filter out very short or very long intervals
                {
                    intervals.Add(interval);
                }
            }
            
            if (intervals.Count == 0)
                return 120;
            
            // Find common beat intervals using histogram approach
            var beatCandidates = new Dictionary<int, int>(); // BPM -> count
            
            foreach (var interval in intervals)
            {
                // Test various beat divisions (quarter, eighth, sixteenth notes)
                for (int division = 1; division <= 4; division *= 2)
                {
                    float beatInterval = interval * division;
                    int bpm = (int)Math.Round(60.0f / beatInterval);
                    
                    // Only consider reasonable tempo range
                    if (bpm >= 40 && bpm <= 200)
                    {
                        // Allow some tolerance for tempo variations
                        for (int offset = -2; offset <= 2; offset++)
                        {
                            int candidateBpm = bpm + offset;
                            if (candidateBpm >= 40 && candidateBpm <= 200)
                            {
                                if (!beatCandidates.ContainsKey(candidateBpm))
                                    beatCandidates[candidateBpm] = 0;
                                beatCandidates[candidateBpm]++;
                            }
                        }
                    }
                }
            }
            
            // Find the most common BPM
            if (beatCandidates.Count == 0)
                return 120;
            
            var detectedBpm = beatCandidates.OrderByDescending(kvp => kvp.Value).First().Key;
            
            // Prefer common tempos if they're close
            int[] commonTempos = { 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
            foreach (var commonTempo in commonTempos)
            {
                if (Math.Abs(detectedBpm - commonTempo) <= 3)
                {
                    Console.WriteLine($"Tempo detection: {detectedBpm} BPM -> snapped to common tempo {commonTempo} BPM");
                    return commonTempo;
                }
            }
            
            Console.WriteLine($"Tempo detection: {detectedBpm} BPM (confidence: {beatCandidates[detectedBpm]} votes)");
            return detectedBpm;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                }
                _disposed = true;
            }
        }

        // Internal classes
        private class WindowResult
        {
            public float[] NoteActivations { get; set; } = Array.Empty<float>();
            public float[]? OnsetActivations { get; set; }
            public float[]? ContourActivations { get; set; }
            public int FrameCount { get; set; }
            public int KeyCount { get; set; }
            public int ContourBinCount { get; set; }
            public int StartSample { get; set; }
            public bool NeedsSigmoid { get; set; }
        }

        private class MergedResults
        {
            public float[,] NoteActivations { get; set; } = new float[0, 0];
            public float[,] OnsetActivations { get; set; } = new float[0, 0];
            public float[,] PitchContour { get; set; } = new float[0, 0];
            public int FrameCount { get; set; }
            public float MaxActivation { get; set; } = float.MinValue;
            public float MinActivation { get; set; } = float.MaxValue;
            public float AverageActivation { get; set; }
            public float ActivationSum { get; set; }
            public int ActivationCount { get; set; }
            public bool SigmoidApplied { get; set; }
            public int DetectionsAboveThreshold { get; set; }
        }
    }
}