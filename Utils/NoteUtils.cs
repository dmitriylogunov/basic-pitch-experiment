namespace BasicPitchApp.Utils
{
    /// <summary>
    /// Utility functions for musical note operations
    /// </summary>
    public static class NoteUtils
    {
        /// <summary>
        /// Converts a MIDI note number to note name (e.g., C4, F#5)
        /// 
        /// WHAT THIS FUNCTION DOES:
        /// - Takes a MIDI number (like 60) and converts it to a note name (like "C4")
        /// - Uses the standard naming convention: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        /// - Includes the octave number (C4 = middle C)
        /// 
        /// MUSICAL THEORY:
        /// - There are 12 different note names that repeat in each octave
        /// - # means "sharp" (raised by a semitone)
        /// - Octave numbers increase every 12 semitones
        /// - MIDI note 60 = C4 (middle C) by convention
        /// </summary>
        /// <param name="midiNote">MIDI note number</param>
        /// <returns>Note name string</returns>
        public static string GetNoteName(int midiNote)
        {
            // The 12 note names that repeat in each octave
            string[] noteNames = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
            
            // Calculate which octave this note is in
            // MIDI octaves start at C, and MIDI note 0 is C(-1)
            int octave = (midiNote / 12) - 1;
            
            // Figure out which of the 12 notes this is
            int noteIndex = midiNote % 12;
            
            // Combine note name and octave number
            return $"{noteNames[noteIndex]}{octave}";
        }
    }
}