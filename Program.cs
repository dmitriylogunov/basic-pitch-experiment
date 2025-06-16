// Libraries for machine learning model inference
using Microsoft.ML.OnnxRuntime;          // For running ONNX (machine learning) models

// Libraries for audio file processing
using NAudio.MediaFoundation;            // For audio format conversion and resampling

// Import our modular components
using BasicPitchExperimentApp.Audio;
using BasicPitchExperimentApp.ML;
using BasicPitchExperimentApp.Midi;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;

using System;

namespace BasicPitchExperimentApp
{
    /// <summary>
    /// Console application that uses Basic Pitch ONNX model to detect notes from audio
    /// and generate MIDI files with detected note information
    /// 
    /// WHAT THIS PROGRAM DOES:
    /// 1. Loads an audio file (like a guitar recording)
    /// 2. Uses an AI model to detect what musical notes are being played
    /// 3. Creates a MIDI file that represents those notes
    /// 4. Saves detailed information about the detected notes
    /// 
    /// BASIC PITCH MODEL:
    /// Basic Pitch is an AI model trained by Spotify to detect musical notes in audio.
    /// It can identify what notes are being played and when they start/stop.
    /// </summary>
    class Program
    {
        // Constants for Basic Pitch model configuration
        // These values control how the audio is processed and analyzed
        private const int SAMPLE_RATE = 22050;           // How many audio samples per second (like frame rate for audio)

        static void Main(string[] args)
        {
            Console.WriteLine("=== Basic Pitch ONNX Note Detection ===");
            Console.WriteLine("Loading model and processing audio...\n");
            
            // Check for command line arguments
            if (args.Length > 0 && args[0] == "--debug")
            {
                Console.WriteLine("DEBUG MODE: Using lower thresholds for testing");
            }

            // Initialize MediaFoundation for audio processing
            // This is required by NAudio to handle different audio formats
            MediaFoundationApi.Startup();

            try
            {
                // STEP 1: Load the AI model
                // ONNX is a standard format for machine learning models
                // The model file contains the trained AI that can detect musical notes
                using var session = new InferenceSession("./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx");
                
                // STEP 2: Load the audio file we want to analyze
                Console.WriteLine("Loading audio file: guitar_sample.wav");
                float[] audioData = AudioProcessor.LoadAudioFile("guitar_sample.wav", SAMPLE_RATE);
                Console.WriteLine($"Audio loaded: {audioData.Length} samples, {audioData.Length / (float)SAMPLE_RATE:F2} seconds");

                // Check and normalize audio if needed
                AudioProcessor.CheckAudioNormalization(audioData);

                // STEP 3: Prepare the audio data for the AI model
                // The model expects data in a specific format, so we need to convert it
                Console.WriteLine("Preprocessing audio data...");
                var spectrogramTensor = ModelInference.PreprocessAudio(audioData);
                
                // STEP 4: Ask the AI model to analyze the audio and detect notes
                Console.WriteLine("Running ONNX model inference...");
                var modelOutputs = ModelInference.RunInference(session, spectrogramTensor);
                
                // STEP 5: Convert the AI model's raw output into understandable note information
                Console.WriteLine("Processing model outputs...");
                var detectedNotes = ModelInference.ProcessModelOutputs(modelOutputs, audioData.Length, SAMPLE_RATE);
                
                // STEP 6: Create a MIDI file from the detected notes
                // MIDI is a standard format that music software can understand
                Console.WriteLine($"Generating MIDI file with {detectedNotes.Count} detected notes...");
                MidiGenerator.GenerateMidiFile(detectedNotes, "output.mid");
                
                // STEP 7: Save a human-readable summary of what we found
                Console.WriteLine("Saving note detection results...");
                FileUtils.SaveNotesToTextFile(detectedNotes, "detected_notes.txt");
                
                // Show the user what we accomplished
                Console.WriteLine("\n=== Processing Complete ===");
                Console.WriteLine($"Generated files:");
                Console.WriteLine($"- output.mid (MIDI file)");
                Console.WriteLine($"- detected_notes.txt (note details)");
                Console.WriteLine($"Total notes detected: {detectedNotes.Count}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}