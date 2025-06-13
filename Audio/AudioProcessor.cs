using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NAudio.MediaFoundation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BasicPitchApp.Audio
{
    /// <summary>
    /// Handles audio file loading, conversion, and normalization
    /// </summary>
    public static class AudioProcessor
    {
        /// <summary>
        /// Loads an audio file and converts it to the required sample rate and format
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Opens an audio file (WAV, MP3, etc.)
        /// - Converts it to the sample rate the AI model expects (22,050 Hz)
        /// - Converts stereo to mono if needed (AI model works with single channel)
        /// - Returns the audio as an array of numbers representing sound waves
        /// 
        /// AUDIO CONCEPTS:
        /// - Sample Rate: How many measurements of sound per second (like FPS for audio)
        /// - Mono vs Stereo: Mono = 1 channel, Stereo = 2 channels (left/right)
        /// - Audio Samples: Individual measurements of sound wave amplitude
        /// </summary>
        /// <param name="filePath">Path to the audio file</param>
        /// <param name="targetSampleRate">Target sample rate for conversion</param>
        /// <returns>Array of audio samples as floats</returns>
        public static float[] LoadAudioFile(string filePath, int targetSampleRate)
        {
            // Open the audio file for reading
            using var audioFile = new AudioFileReader(filePath);
            
            // Check if we need to change the sample rate
            // The AI model expects exactly 22,050 samples per second
            ISampleProvider sampleProvider = audioFile;
            if (audioFile.WaveFormat.SampleRate != targetSampleRate)
            {
                Console.WriteLine($"Resampling from {audioFile.WaveFormat.SampleRate}Hz to {targetSampleRate}Hz");
                // Resample = change how many measurements per second we have
                // Like converting a 60fps video to 30fps
                sampleProvider = new MediaFoundationResampler(audioFile, targetSampleRate).ToSampleProvider();
            }
            
            // Check if we need to convert from stereo to mono
            // The AI model works with single-channel audio only
            if (sampleProvider.WaveFormat.Channels > 1)
            {
                Console.WriteLine("Converting stereo to mono");
                // ToMono() averages the left and right channels into one
                sampleProvider = sampleProvider.ToMono();
            }
            
            // Read all the audio data into memory
            // Audio files can be large, so we read them in chunks
            var samples = new List<float>();
            var buffer = new float[4096];  // Read 4096 samples at a time
            int samplesRead;
            
            // Keep reading until we've got all the audio data
            while ((samplesRead = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
            {
                // Add each sample to our collection
                // Each sample is a number representing the sound wave at that moment
                for (int i = 0; i < samplesRead; i++)
                {
                    samples.Add(buffer[i]);
                }
            }
            
            return samples.ToArray();
        }

        /// <summary>
        /// Checks and normalizes audio data to ensure it's loud enough for processing
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Analyzes the audio data to find the minimum and maximum values
        /// - Checks if the audio is too quiet (below 50% of full range)
        /// - If quiet, normalizes it to use 95% of the available range
        /// 
        /// WHY THIS IS IMPORTANT:
        /// - AI models work better with properly normalized audio
        /// - Too quiet audio might not be detected properly
        /// - Normalizing ensures consistent detection quality
        /// </summary>
        /// <param name="audioData">Audio samples to check and normalize</param>
        public static void CheckAudioNormalization(float[] audioData)
        {
            float min = audioData.Min();
            float max = audioData.Max();
            float absMax = Math.Max(Math.Abs(min), Math.Abs(max));
            
            Console.WriteLine($"Min: {min}, Max: {max}, AbsMax: {absMax}");

            // If absMax is much less than 1.0, your audio is too quiet
            if (absMax < 0.5f)
            {
                Console.WriteLine("Audio seems too quiet!");

                // Normalize to use full range
                float scale = 0.95f / absMax;
                for (int i = 0; i < audioData.Length; i++)
                {
                    audioData[i] *= scale;
                }
            }
            else
            {
                Console.WriteLine("Normalization check... OK");
            }
        }
    }
}