using BasicPitchExperimentApp.Models;
using System.Collections.Generic;

namespace BasicPitchExperimentApp.ML
{
    /// <summary>
    /// Interface for music transcription model inference
    /// </summary>
    public interface IModelInference
    {
        /// <summary>
        /// Processes audio data and returns detected musical notes
        /// </summary>
        /// <param name="input">Input parameters for the model</param>
        /// <returns>Output containing detected notes and metadata</returns>
        ModelOutput ProcessAudio(ModelInput input);
    }

    /// <summary>
    /// Input parameters for the model inference
    /// </summary>
    public class ModelInput
    {
        /// <summary>
        /// Raw audio data as float array
        /// </summary>
        public float[] AudioData { get; set; } = Array.Empty<float>();

        /// <summary>
        /// Sample rate of the audio (e.g., 22050 Hz)
        /// </summary>
        public int SampleRate { get; set; }

        /// <summary>
        /// Configuration parameters for note detection
        /// </summary>
        public InferenceParameters Parameters { get; set; } = new InferenceParameters();
    }

    /// <summary>
    /// Configuration parameters for the inference process
    /// </summary>
    public class InferenceParameters
    {
        /// <summary>
        /// Minimum confidence threshold for note detection (0.0 to 1.0)
        /// </summary>
        public float NoteThreshold { get; set; } = 0.3f;

        /// <summary>
        /// Minimum confidence threshold for onset detection (0.0 to 1.0)
        /// </summary>
        public float OnsetThreshold { get; set; } = 0.5f;

        /// <summary>
        /// Minimum note duration in seconds
        /// </summary>
        public float MinNoteLength { get; set; } = 0.127f;

        /// <summary>
        /// Number of frames to overlap between processing windows
        /// </summary>
        public int OverlappingFrames { get; set; } = 30;

        /// <summary>
        /// Whether to apply sigmoid transformation to raw model outputs
        /// </summary>
        public bool AutoApplySigmoid { get; set; } = true;

        /// <summary>
        /// Whether to use onset detection to split consecutive same notes
        /// </summary>
        public bool UseOnsetForNoteSplitting { get; set; } = true;

        /// <summary>
        /// Minimum frames between notes for onset-based splitting
        /// </summary>
        public int MinFramesBetweenOnsets { get; set; } = 3;
    }

    /// <summary>
    /// Output from the model inference
    /// </summary>
    public class ModelOutput
    {
        /// <summary>
        /// List of detected musical notes
        /// </summary>
        public List<DetectedNote> Notes { get; set; } = new List<DetectedNote>();

        /// <summary>
        /// Raw note activations from the model [frames, 88 piano keys]
        /// </summary>
        public float[,]? NoteActivations { get; set; }

        /// <summary>
        /// Raw onset activations from the model [frames, 88 piano keys]
        /// </summary>
        public float[,]? OnsetActivations { get; set; }

        /// <summary>
        /// Raw pitch contour from the model [frames, 264 frequency bins]
        /// </summary>
        public float[,]? PitchContour { get; set; }

        /// <summary>
        /// Number of time frames processed
        /// </summary>
        public int FrameCount { get; set; }

        /// <summary>
        /// Frame rate in frames per second
        /// </summary>
        public float FrameRate { get; set; }

        /// <summary>
        /// Statistics about the inference process
        /// </summary>
        public InferenceStatistics Statistics { get; set; } = new InferenceStatistics();
    }

    /// <summary>
    /// Statistics collected during inference
    /// </summary>
    public class InferenceStatistics
    {
        /// <summary>
        /// Total number of audio windows processed
        /// </summary>
        public int WindowsProcessed { get; set; }

        /// <summary>
        /// Maximum activation value seen
        /// </summary>
        public float MaxActivation { get; set; }

        /// <summary>
        /// Minimum activation value seen
        /// </summary>
        public float MinActivation { get; set; }

        /// <summary>
        /// Average activation value
        /// </summary>
        public float AverageActivation { get; set; }

        /// <summary>
        /// Whether sigmoid was applied to outputs
        /// </summary>
        public bool SigmoidApplied { get; set; }

        /// <summary>
        /// Number of detections above threshold before filtering
        /// </summary>
        public int DetectionsAboveThreshold { get; set; }

        /// <summary>
        /// Processing time in milliseconds
        /// </summary>
        public double ProcessingTimeMs { get; set; }
    }
}