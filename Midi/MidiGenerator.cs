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
        public static void GenerateMidiFile(List<DetectedNote> notes, string outputPath)
        {
            // Create a new MIDI file structure
            var midiFile = new MidiFile();  // The complete MIDI file
            var track = new TrackChunk();   // One track to hold all our notes
            
            // Set the tempo (speed) of the music
            // 120 BPM = 120 quarter notes per minute = 0.5 seconds per quarter note
            // 500,000 microseconds = 0.5 seconds
            track.Events.Add(new SetTempoEvent(500000));
            
            // Convert each detected note into MIDI events
            foreach (var note in notes)
            {
                // Convert time from seconds to MIDI ticks
                // 480 ticks per quarter note is a common standard
                // At 120 BPM, we have 2 quarter notes per second, so multiply by 2
                long startTicks = (long)(note.StartTime * 480 * 2);
                long endTicks = (long)(note.EndTime * 480 * 2);
                
                // Create a "Note On" event (start playing the note)
                var noteOn = new NoteOnEvent(
                    (SevenBitNumber)note.MidiNote,  // Which note to play
                    (SevenBitNumber)80              // How hard to play it (velocity)
                )
                {
                    DeltaTime = startTicks  // When to start playing
                };
                
                // Create a "Note Off" event (stop playing the note)
                var noteOff = new NoteOffEvent(
                    (SevenBitNumber)note.MidiNote,  // Which note to stop
                    (SevenBitNumber)0               // Release velocity (usually 0)
                )
                {
                    DeltaTime = endTicks - startTicks  // How long after Note On to stop
                };
                
                // Add both events to our track
                track.Events.Add(noteOn);
                track.Events.Add(noteOff);
            }
            
            // Sort all events by when they should happen
            // MIDI events must be in chronological order
            var sortedEvents = track.Events.OrderBy(e => e.DeltaTime).ToList();
            track.Events.Clear();
            foreach (var evt in sortedEvents)
            {
                track.Events.Add(evt);
            }
            
            // Convert from absolute timing to relative timing
            // MIDI uses "delta time" = time since the previous event
            // Like directions: "go 5 miles, then turn left, then go 3 more miles"
            long previousTime = 0;
            foreach (var midiEvent in track.Events)
            {
                long absoluteTime = midiEvent.DeltaTime;  // When this event happens
                midiEvent.DeltaTime = absoluteTime - previousTime;  // Time since last event
                previousTime = absoluteTime;
            }
            
            // The library automatically adds an "End of Track" event when saving
            // This tells MIDI players that the song is finished
            
            // Add our track to the MIDI file
            midiFile.Chunks.Add(track);
            
            // Save the complete MIDI file to disk
            // Delete existing file if it exists to avoid overwrite errors
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
            midiFile.Write(outputPath);
        }
    }
}