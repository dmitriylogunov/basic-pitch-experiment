using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.Interaction;
using BasicPitchExperimentApp.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BasicPitchExperimentApp.Midi
{
    /// <summary>
    /// Handles MIDI file generation from detected notes
    /// </summary>
    public static class MidiGenerator
    {
        /// <summary>
        /// Generates a MIDI file from the detected notes
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes our list of detected notes
        /// - Creates a standard MIDI file that music software can read
        /// - Converts timing from seconds to MIDI "ticks"
        /// - Creates "Note On" and "Note Off" events for each note
        /// 
        /// MIDI CONCEPTS:
        /// - MIDI = Musical Instrument Digital Interface
        /// - MIDI files don't contain audio, just instructions
        /// - Like sheet music for computers
        /// - Ticks = MIDI's way of measuring time (like frame rate)
        /// - Note On = start playing a note
        /// - Note Off = stop playing a note
        /// </summary>
        /// <param name="notes">List of detected notes</param>
        /// <param name="outputPath">Path for the output MIDI file</param>
        public static void GenerateMidiFile(List<DetectedNote> notes, string outputPath, int bpm = 120)
        {
            Console.WriteLine($"Generating MIDI file with BPM: {bpm}");
            Console.WriteLine($"Number of notes: {notes.Count}");
            
            // Create a new MIDI file structure
            var midiFile = new MidiFile();  // The complete MIDI file
            var track = new TrackChunk();   // One track to hold all our notes
            
            // Create a list to hold all events with their absolute timing
            var timedEvents = new List<(long absoluteTime, MidiEvent midiEvent)>();
            
            // Set the tempo (speed) of the music
            // Convert BPM to microseconds per quarter note
            // Formula: 60,000,000 / BPM = microseconds per quarter note
            int microsecondsPerQuarterNote = 60_000_000 / bpm;
            var tempoEvent = new SetTempoEvent(microsecondsPerQuarterNote);
            timedEvents.Add((0, tempoEvent));  // Tempo event at time 0
            
            // Convert each detected note into MIDI events
            int noteIndex = 0;
            foreach (var note in notes)
            {
                // Convert time from seconds to MIDI ticks
                // 480 ticks per quarter note is a common standard
                // Calculate quarter notes per second based on actual BPM
                double quarterNotesPerSecond = bpm / 60.0;
                long startTicks = (long)(note.StartTime * 480 * quarterNotesPerSecond);
                long endTicks = (long)(note.EndTime * 480 * quarterNotesPerSecond);
                
                if (noteIndex < 5) // Log first 5 notes for debugging
                {
                    Console.WriteLine($"Note {noteIndex}: MIDI {note.MidiNote}, " +
                        $"Start: {note.StartTime:F3}s -> {startTicks} ticks, " +
                        $"End: {note.EndTime:F3}s -> {endTicks} ticks");
                }
                noteIndex++;
                
                // Create a "Note On" event (start playing the note)
                var noteOn = new NoteOnEvent(
                    (SevenBitNumber)note.MidiNote,  // Which note to play
                    (SevenBitNumber)80              // How hard to play it (velocity)
                );
                
                // Create a "Note Off" event (stop playing the note)
                var noteOff = new NoteOffEvent(
                    (SevenBitNumber)note.MidiNote,  // Which note to stop
                    (SevenBitNumber)0               // Release velocity (usually 0)
                );
                
                // Add both events with their absolute timing
                timedEvents.Add((startTicks, noteOn));
                timedEvents.Add((endTicks, noteOff));
            }
            
            // Sort all events by their absolute time
            timedEvents.Sort((a, b) => a.absoluteTime.CompareTo(b.absoluteTime));
            
            // Convert from absolute timing to relative timing and add to track
            // MIDI uses "delta time" = time since the previous event
            long previousTime = 0;
            foreach (var (absoluteTime, midiEvent) in timedEvents)
            {
                midiEvent.DeltaTime = absoluteTime - previousTime;  // Time since last event
                track.Events.Add(midiEvent);
                previousTime = absoluteTime;
            }
            
            // The library automatically adds an "End of Track" event when saving
            // This tells MIDI players that the song is finished
            
            // Add our track to the MIDI file
            midiFile.Chunks.Add(track);
            
            // Set the time division (ticks per quarter note)
            // This is CRITICAL for proper playback speed
            midiFile.TimeDivision = new TicksPerQuarterNoteTimeDivision(480);
            
            // Save the complete MIDI file to disk
            // Delete existing file if it exists to avoid overwrite errors
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
            midiFile.Write(outputPath);
            
            Console.WriteLine($"MIDI file saved: {outputPath}");
            Console.WriteLine($"Time division: 480 ticks per quarter note");
            Console.WriteLine($"Tempo: {bpm} BPM ({microsecondsPerQuarterNote} μs per quarter note)");
        }
    }
}