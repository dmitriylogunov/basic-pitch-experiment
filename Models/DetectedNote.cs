namespace BasicPitchApp.Models
{
    /// <summary>
    /// Represents a detected musical note with timing and pitch information
    /// 
    /// WHAT THIS CLASS STORES:
    /// - All the information we know about one detected musical note
    /// - Think of it as a digital version of a note on sheet music
    /// - Includes when it plays, what pitch it is, and how confident we are
    /// 
    /// WHY WE NEED THIS:
    /// - Organizes all note information in one place
    /// - Makes it easy to pass note data between functions
    /// - Provides a clear structure for our detection results
    /// </summary>
    public class DetectedNote
    {
        /// <summary>MIDI note number (0-127) - Standard way to represent pitch</summary>
        public int MidiNote { get; set; }
        
        /// <summary>Start time in seconds - When this note begins playing</summary>
        public float StartTime { get; set; }
        
        /// <summary>End time in seconds - When this note stops playing</summary>
        public float EndTime { get; set; }
        
        /// <summary>Duration in seconds - How long this note lasts</summary>
        public float Duration { get; set; }
        
        /// <summary>Confidence score from the model - How sure the AI is about this note (0-1)</summary>
        public float Confidence { get; set; }
        
        /// <summary>Frequency in Hz - The actual sound frequency of this note</summary>
        public float Frequency { get; set; }
    }
}