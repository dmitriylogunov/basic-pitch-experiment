using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using Microsoft.Win32;
using Microsoft.ML.OnnxRuntime;
using NAudio.MediaFoundation;
using NAudio.Wave;
using BasicPitchExperimentApp.Audio;
using BasicPitchExperimentApp.ML;
using BasicPitchExperimentApp.Midi;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;
using BasicPitchExperimentApp.UI;

namespace BasicPitchExperimentApp
{
    public partial class MainWindow : Window
    {
        private const int SAMPLE_RATE = 22050;
        private InferenceSession? session;
        private float[]? loadedAudioData;
        private string? currentAudioFile;
        private List<DetectedNote>? detectedNotes;
        private IWavePlayer? wavePlayer;
        private string? currentMidiFile = "output.mid";
        private MusicNotationRenderer? notationRenderer;

        public MainWindow()
        {
            InitializeComponent();
            MediaFoundationApi.Startup();
            LoadModel();
            
            LengthMatchCombo.SelectionChanged += LengthMatchCombo_SelectionChanged;
            notationRenderer = new MusicNotationRenderer(NotationCanvas);
        }

        private async void LoadModel()
        {
            try
            {
                LogMessage("Loading Basic Pitch model...");
                await Task.Run(() =>
                {
                    session = new InferenceSession("./basic-pitch/basic_pitch/saved_models/icassp_2022/nmp.onnx");
                });
                LogMessage("Model loaded successfully!");
            }
            catch (Exception ex)
            {
                LogMessage($"Error loading model: {ex.Message}");
                MessageBox.Show($"Failed to load model: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BrowseButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Audio files (*.wav;*.mp3;*.m4a;*.flac)|*.wav;*.mp3;*.m4a;*.flac|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                FilePathTextBox.Text = openFileDialog.FileName;
                currentAudioFile = openFileDialog.FileName;
                ProcessButton.IsEnabled = true;
                LogMessage($"Selected file: {Path.GetFileName(currentAudioFile)}");
            }
        }

        private async void ProcessButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentAudioFile) || session == null) return;

            try
            {
                ProcessButton.IsEnabled = false;
                RegenerateButton.IsEnabled = false;
                ProcessingProgressBar.Visibility = Visibility.Visible;
                ProcessingProgressBar.IsIndeterminate = true;

                LogMessage($"Loading audio file: {Path.GetFileName(currentAudioFile)}");
                
                await Task.Run(() =>
                {
                    loadedAudioData = AudioProcessor.LoadAudioFile(currentAudioFile, SAMPLE_RATE);
                });

                var audioDuration = loadedAudioData.Length / (float)SAMPLE_RATE;
                LogMessage($"Audio loaded: {loadedAudioData.Length} samples, {audioDuration:F2} seconds");
                
                await Dispatcher.InvokeAsync(() =>
                {
                    DurationTextBox.Text = audioDuration.ToString("F2");
                });

                await ProcessAudioWithCurrentSettings();
            }
            catch (Exception ex)
            {
                LogMessage($"Error processing audio: {ex.Message}");
                MessageBox.Show($"Failed to process audio: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                ProcessButton.IsEnabled = true;
                ProcessingProgressBar.Visibility = Visibility.Collapsed;
            }
        }

        private async Task ProcessAudioWithCurrentSettings()
        {
            if (loadedAudioData == null || session == null) return;

            try
            {
                LogMessage("Processing audio with Basic Pitch model...");
                
                // Get current threshold values
                var noteThreshold = (float)NoteThresholdSlider.Value;
                var onsetThreshold = (float)OnsetThresholdSlider.Value;
                var minNoteDuration = (float)(MinNoteDurationSlider.Value / 1000.0); // Convert ms to seconds
                
                LogMessage($"Using thresholds - Note: {noteThreshold:F2}, Onset: {onsetThreshold:F2}, Min Duration: {minNoteDuration:F3}s");

                // Create modified inference parameters
                var modelParams = new ModelParameters
                {
                    NoteThreshold = noteThreshold,
                    OnsetThreshold = onsetThreshold,
                    MinNoteLength = minNoteDuration
                };

                await Task.Run(() =>
                {
                    // Process with custom parameters
                    detectedNotes = ProcessFullAudioWithParams(session, loadedAudioData, SAMPLE_RATE, modelParams);
                });

                LogMessage($"Detected {detectedNotes.Count} notes");

                // Generate MIDI with current settings
                await GenerateMidiWithCurrentSettings();
                
                // Update notation display
                UpdateNotationDisplay();
                
                RegenerateButton.IsEnabled = true;
                PlayButton.IsEnabled = true;
                SaveMidiButton.IsEnabled = true;
            }
            catch (Exception ex)
            {
                LogMessage($"Error during processing: {ex.Message}");
                throw;
            }
        }

        private async Task GenerateMidiWithCurrentSettings()
        {
            if (detectedNotes == null || detectedNotes.Count == 0) return;

            try
            {
                // Get BPM setting
                int bpm = 120;
                if (int.TryParse(BpmTextBox.Text, out int parsedBpm) && parsedBpm > 0)
                {
                    bpm = parsedBpm;
                }

                // Get length matching mode
                var lengthMode = (LengthMatchCombo.SelectedItem as ComboBoxItem)?.Content.ToString();
                float? targetDuration = null;

                if (lengthMode == "Custom (sec)")
                {
                    if (float.TryParse(DurationTextBox.Text, out float duration) && duration > 0)
                    {
                        targetDuration = duration;
                    }
                }
                else if (lengthMode == "Stretch to BPM" && loadedAudioData != null)
                {
                    // Calculate stretch factor based on BPM
                    var originalDuration = loadedAudioData.Length / (float)SAMPLE_RATE;
                    targetDuration = CalculateDurationForBPM(detectedNotes, bpm);
                }

                LogMessage($"Generating MIDI file - BPM: {bpm}, Mode: {lengthMode ?? "Original"}");
                
                await Task.Run(() =>
                {
                    // Generate MIDI with parameters
                    if (!string.IsNullOrEmpty(currentMidiFile))
                    {
                        GenerateMidiWithParameters(detectedNotes, currentMidiFile, bpm, targetDuration);
                    }
                });

                LogMessage($"MIDI file generated: {currentMidiFile ?? "output.mid"}");
            }
            catch (Exception ex)
            {
                LogMessage($"Error generating MIDI: {ex.Message}");
                throw;
            }
        }

        private void GenerateMidiWithParameters(List<DetectedNote> notes, string filename, int bpm, float? targetDuration)
        {
            // Create a copy of notes to potentially modify timing
            var processedNotes = new List<DetectedNote>(notes);

            // Apply time stretching if needed
            if (targetDuration.HasValue && notes.Count > 0)
            {
                var originalDuration = notes.Max(n => n.EndTime);
                var stretchFactor = targetDuration.Value / originalDuration;
                
                processedNotes = notes.Select(n => new DetectedNote
                {
                    MidiNote = n.MidiNote,
                    StartTime = n.StartTime * stretchFactor,
                    EndTime = n.EndTime * stretchFactor,
                    Duration = n.Duration * stretchFactor,
                    Confidence = n.Confidence,
                    Frequency = n.Frequency
                }).ToList();
            }

            // Use existing MIDI generator with BPM
            MidiGenerator.GenerateMidiFile(processedNotes, filename, bpm);
        }

        private float CalculateDurationForBPM(List<DetectedNote> notes, int bpm)
        {
            // Simple calculation - could be enhanced
            // For now, just ensure the duration fits nicely with the BPM
            var originalDuration = notes.Max(n => n.EndTime);
            var beatsPerSecond = bpm / 60.0f;
            var totalBeats = Math.Ceiling(originalDuration * beatsPerSecond);
            return (float)(totalBeats / beatsPerSecond);
        }

        private async void RegenerateButton_Click(object sender, RoutedEventArgs e)
        {
            if (loadedAudioData == null || session == null) return;

            try
            {
                RegenerateButton.IsEnabled = false;
                ProcessingProgressBar.Visibility = Visibility.Visible;
                ProcessingProgressBar.IsIndeterminate = true;

                LogMessage("Regenerating with new parameters...");
                await ProcessAudioWithCurrentSettings();
            }
            finally
            {
                RegenerateButton.IsEnabled = true;
                ProcessingProgressBar.Visibility = Visibility.Collapsed;
            }
        }

        private void PlayButton_Click(object sender, RoutedEventArgs e)
        {
            if (!File.Exists(currentMidiFile)) return;

            try
            {
                StopPlayback();

                // For MIDI playback, we'll use NAudio's MIDI capabilities
                LogMessage("Playing MIDI file...");
                
                // Simple approach - convert MIDI to WAV for playback
                // You could also use a MIDI synthesizer library
                PlayButton.IsEnabled = false;
                StopButton.IsEnabled = true;

                // TODO: Implement proper MIDI playback
                LogMessage("MIDI playback functionality to be implemented");
                MessageBox.Show("MIDI playback will be implemented with a synthesizer library.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                LogMessage($"Error playing MIDI: {ex.Message}");
                MessageBox.Show($"Failed to play MIDI: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void StopButton_Click(object sender, RoutedEventArgs e)
        {
            StopPlayback();
        }

        private void StopPlayback()
        {
            try
            {
                wavePlayer?.Stop();
                wavePlayer?.Dispose();
                wavePlayer = null;
                
                PlayButton.IsEnabled = true;
                StopButton.IsEnabled = false;
                LogMessage("Playback stopped");
            }
            catch (Exception ex)
            {
                LogMessage($"Error stopping playback: {ex.Message}");
            }
        }

        private void SaveMidiButton_Click(object sender, RoutedEventArgs e)
        {
            if (!File.Exists(currentMidiFile)) return;

            var saveFileDialog = new SaveFileDialog
            {
                Filter = "MIDI files (*.mid)|*.mid|All files (*.*)|*.*",
                DefaultExt = "mid",
                FileName = "converted_audio.mid"
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                try
                {
                    File.Copy(currentMidiFile, saveFileDialog.FileName, true);
                    LogMessage($"MIDI file saved to: {saveFileDialog.FileName}");
                    MessageBox.Show("MIDI file saved successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    LogMessage($"Error saving MIDI file: {ex.Message}");
                    MessageBox.Show($"Failed to save MIDI file: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void LengthMatchCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (DurationTextBox == null) return;
            
            var selectedItem = (LengthMatchCombo.SelectedItem as ComboBoxItem)?.Content.ToString();
            DurationTextBox.IsEnabled = selectedItem == "Custom (sec)";
        }
        
        private void GuitarNotationCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            UpdateNotationDisplay();
        }
        
        private void UpdateNotationDisplay()
        {
            Dispatcher.Invoke(() =>
            {
                if (notationRenderer != null && detectedNotes != null)
                {
                    bool guitarNotation = GuitarNotationCheckBox?.IsChecked ?? true;
                    notationRenderer.RenderNotes(detectedNotes, guitarNotation);
                }
            });
        }

        private void LogMessage(string message)
        {
            Dispatcher.Invoke(() =>
            {
                LogTextBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}\n");
                LogTextBox.ScrollToEnd();
            });
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);
            StopPlayback();
            session?.Dispose();
            MediaFoundationApi.Shutdown();
        }

        // Modified version of ProcessFullAudio that accepts custom parameters
        private List<DetectedNote> ProcessFullAudioWithParams(InferenceSession session, float[] audioData, int sampleRate, ModelParameters parameters)
        {
            // This is a simplified version - you'd need to modify ModelInference.cs to accept these parameters
            // For now, we'll use the existing method
            var notes = ModelInference.ProcessFullAudio(session, audioData, sampleRate);
            
            // Apply additional filtering based on parameters
            return notes.Where(n => 
                n.Confidence >= parameters.NoteThreshold && 
                n.Duration >= parameters.MinNoteLength
            ).ToList();
        }
    }

    // Helper class for model parameters
    public class ModelParameters
    {
        public float NoteThreshold { get; set; } = 0.5f;
        public float OnsetThreshold { get; set; } = 0.5f;
        public float MinNoteLength { get; set; } = 0.127f;
    }
}