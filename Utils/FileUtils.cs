using BasicPitchApp.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BasicPitchApp.Utils
{
    /// <summary>
    /// Utility functions for file operations
    /// </summary>
    public static class FileUtils
    {
        /// <summary>
        /// Saves detailed information about detected notes to a text file
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Creates a human-readable report of all detected notes
        /// - Shows note names, timing, frequencies, and confidence scores
        /// - Includes summary statistics for analysis
        /// 
        /// WHY THIS IS USEFUL:
        /// - MIDI files are for music software, this is for humans
        /// - Helps verify that the detection worked correctly
        /// - Provides detailed analysis data for research or debugging
        /// </summary>
        /// <param name="notes">List of detected notes</param>
        /// <param name="outputPath">Path for the output text file</param>
        public static void SaveNotesToTextFile(List<DetectedNote> notes, string outputPath)
        {
            // Build the text report using StringBuilder (efficient for multiple lines)
            var sb = new StringBuilder();
            
            // Add header information
            sb.AppendLine("=== Basic Pitch Note Detection Results ===");
            sb.AppendLine($"Generated on: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Total notes detected: {notes.Count}");
            sb.AppendLine();
            
            // Add column headers to explain the data format
            sb.AppendLine("Format: Note | MIDI# | Start(s) | End(s) | Duration(s) | Frequency(Hz) | Confidence");
            sb.AppendLine(new string('-', 80));  // Separator line
            
            // Add a line for each detected note
            foreach (var note in notes)
            {
                // Convert MIDI number to note name (like "C4", "F#5")
                string noteName = NoteUtils.GetNoteName(note.MidiNote);
                
                // Format all the information in neat columns
                sb.AppendLine($"{noteName,-4} | {note.MidiNote,5} | {note.StartTime,8:F3} | {note.EndTime,7:F3} | " +
                             $"{note.Duration,10:F3} | {note.Frequency,11:F2} | {note.Confidence:F3}");
            }
            
            // Add summary statistics section
            sb.AppendLine();
            sb.AppendLine("=== Summary Statistics ===");
            if (notes.Count > 0)
            {
                // Calculate and display interesting statistics about the detected notes
                sb.AppendLine($"Earliest note: {notes.Min(n => n.StartTime):F3}s");  // When first note starts
                sb.AppendLine($"Latest note: {notes.Max(n => n.EndTime):F3}s");      // When last note ends
                sb.AppendLine($"Average duration: {notes.Average(n => n.Duration):F3}s"); // Typical note length
                sb.AppendLine($"Lowest note: {NoteUtils.GetNoteName(notes.Min(n => n.MidiNote))} (MIDI {notes.Min(n => n.MidiNote)})");   // Deepest pitch
                sb.AppendLine($"Highest note: {NoteUtils.GetNoteName(notes.Max(n => n.MidiNote))} (MIDI {notes.Max(n => n.MidiNote)})"); // Highest pitch
                sb.AppendLine($"Average confidence: {notes.Average(n => n.Confidence):F3}");  // How sure we are overall
            }
            
            File.WriteAllText(outputPath, sb.ToString());
        }
    }
}