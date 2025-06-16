using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using BasicPitchExperimentApp.Models;
using BasicPitchExperimentApp.Utils;

namespace BasicPitchExperimentApp.UI
{
    public class MusicNotationRenderer
    {
        private readonly Canvas canvas;
        private readonly double staffHeight = 120;
        private readonly double staffLineSpacing = 10;
        private readonly double leftMargin = 40;
        private readonly double rightMargin = 20;
        private readonly double topMargin = 30;
        private readonly int pixelsPerSecond = 100;
        
        // Define staff positions for treble and bass clefs
        private readonly int[] trebleStaffLines = { 64, 67, 71, 74, 77 }; // E4, G4, B4, D5, F5
        private readonly int[] bassStaffLines = { 43, 47, 50, 53, 57 }; // G2, B2, D3, F3, A3
        
        public MusicNotationRenderer(Canvas notationCanvas)
        {
            canvas = notationCanvas;
        }
        
        public void RenderNotes(List<DetectedNote> notes)
        {
            canvas.Children.Clear();
            
            if (notes == null || notes.Count == 0)
            {
                DrawEmptyStaff();
                return;
            }
            
            // Calculate canvas width based on duration
            double maxTime = notes.Max(n => n.EndTime);
            double canvasWidth = leftMargin + rightMargin + (maxTime * pixelsPerSecond);
            canvas.Width = Math.Max(canvasWidth, canvas.ActualWidth);
            
            // Draw the staff
            DrawStaff();
            
            // Draw time grid
            DrawTimeGrid(maxTime);
            
            // Draw the notes
            foreach (var note in notes)
            {
                DrawNote(note);
            }
        }
        
        private void DrawEmptyStaff()
        {
            canvas.Width = canvas.ActualWidth > 0 ? canvas.ActualWidth : 800;
            DrawStaff();
        }
        
        private void DrawStaff()
        {
            // Draw treble clef staff
            double trebleY = topMargin;
            DrawStaffLines(trebleY, "Treble");
            
            // Draw bass clef staff
            double bassY = topMargin + staffHeight;
            DrawStaffLines(bassY, "Bass");
            
            // Draw clefs
            DrawClef(leftMargin - 25, trebleY + 2 * staffLineSpacing, "Treble");
            DrawClef(leftMargin - 25, bassY + 2 * staffLineSpacing, "Bass");
        }
        
        private void DrawStaffLines(double yOffset, string clefType)
        {
            for (int i = 0; i < 5; i++)
            {
                Line line = new Line
                {
                    X1 = leftMargin,
                    Y1 = yOffset + i * staffLineSpacing,
                    X2 = canvas.Width - rightMargin,
                    Y2 = yOffset + i * staffLineSpacing,
                    Stroke = Brushes.Black,
                    StrokeThickness = 1
                };
                canvas.Children.Add(line);
            }
        }
        
        private void DrawClef(double x, double y, string clefType)
        {
            TextBlock clefText = new TextBlock
            {
                Text = clefType == "Treble" ? "𝄞" : "𝄢",
                FontSize = 40,
                FontFamily = new FontFamily("Segoe UI Symbol"),
                Foreground = Brushes.Black
            };
            
            Canvas.SetLeft(clefText, x);
            Canvas.SetTop(clefText, y - 20);
            canvas.Children.Add(clefText);
        }
        
        private void DrawTimeGrid(double maxTime)
        {
            // Draw measure lines every second (or based on BPM)
            for (double time = 1.0; time <= maxTime; time += 1.0)
            {
                double x = leftMargin + time * pixelsPerSecond;
                
                Line measureLine = new Line
                {
                    X1 = x,
                    Y1 = topMargin,
                    X2 = x,
                    Y2 = topMargin + staffHeight * 2 - staffLineSpacing * 4,
                    Stroke = Brushes.LightGray,
                    StrokeThickness = 1,
                    StrokeDashArray = new DoubleCollection { 2, 2 }
                };
                canvas.Children.Add(measureLine);
                
                // Add time label
                TextBlock timeLabel = new TextBlock
                {
                    Text = $"{time:0}s",
                    FontSize = 10,
                    Foreground = Brushes.Gray
                };
                Canvas.SetLeft(timeLabel, x - 10);
                Canvas.SetTop(timeLabel, topMargin + staffHeight * 2 - staffLineSpacing * 3);
                canvas.Children.Add(timeLabel);
            }
        }
        
        private void DrawNote(DetectedNote note)
        {
            double x = leftMargin + note.StartTime * pixelsPerSecond;
            double noteWidth = Math.Max(5, (note.EndTime - note.StartTime) * pixelsPerSecond);
            
            // Determine which staff and position
            bool isTrebleClef = note.MidiNote >= 60; // Middle C and above
            double staffY = isTrebleClef ? topMargin : topMargin + staffHeight;
            
            // Calculate Y position on staff
            double y = CalculateNoteY(note.MidiNote, staffY, isTrebleClef);
            
            // Draw note head
            Ellipse noteHead = new Ellipse
            {
                Width = 8,
                Height = 6,
                Fill = Brushes.Black,
                Stroke = Brushes.Black,
                StrokeThickness = 1
            };
            
            Canvas.SetLeft(noteHead, x);
            Canvas.SetTop(noteHead, y - 3);
            canvas.Children.Add(noteHead);
            
            // Draw duration line
            Rectangle durationBar = new Rectangle
            {
                Width = noteWidth,
                Height = 4,
                Fill = new SolidColorBrush(Color.FromArgb(100, 0, 0, 255)),
                Stroke = Brushes.Blue,
                StrokeThickness = 0.5
            };
            
            Canvas.SetLeft(durationBar, x);
            Canvas.SetTop(durationBar, y - 2);
            canvas.Children.Add(durationBar);
            
            // Draw ledger lines if needed
            DrawLedgerLines(x, y, note.MidiNote, staffY, isTrebleClef);
            
            // Add note name label
            string noteName = NoteUtils.GetNoteName(note.MidiNote);
            TextBlock noteLabel = new TextBlock
            {
                Text = noteName,
                FontSize = 8,
                Foreground = Brushes.DarkBlue
            };
            Canvas.SetLeft(noteLabel, x);
            Canvas.SetTop(noteLabel, y + 10);
            canvas.Children.Add(noteLabel);
        }
        
        private double CalculateNoteY(int midiNote, double staffY, bool isTrebleClef)
        {
            // Reference notes for positioning
            int referenceNote = isTrebleClef ? 71 : 50; // B4 for treble, D3 for bass
            double referenceY = staffY + 2 * staffLineSpacing; // Middle line
            
            // Calculate semitone difference
            int semitoneDiff = midiNote - referenceNote;
            
            // Convert to staff position (each line/space is 2 semitones apart)
            double staffPositions = semitoneDiff / 2.0;
            
            // Each staff position is half a line spacing
            return referenceY - (staffPositions * staffLineSpacing / 2);
        }
        
        private void DrawLedgerLines(double x, double y, int midiNote, double staffY, bool isTrebleClef)
        {
            double ledgerLength = 15;
            double ledgerX = x - 3;
            
            if (isTrebleClef)
            {
                // Above treble staff
                if (midiNote > 77) // Above F5
                {
                    for (int ledgerNote = 81; ledgerNote <= midiNote; ledgerNote += 4)
                    {
                        double ledgerY = CalculateNoteY(ledgerNote, staffY, true);
                        DrawLedgerLine(ledgerX, ledgerY, ledgerLength);
                    }
                }
                // Below treble staff (including middle C)
                else if (midiNote < 64) // Below E4
                {
                    for (int ledgerNote = 60; ledgerNote >= midiNote; ledgerNote -= 4)
                    {
                        double ledgerY = CalculateNoteY(ledgerNote, staffY, true);
                        DrawLedgerLine(ledgerX, ledgerY, ledgerLength);
                    }
                }
            }
            else
            {
                // Above bass staff (including middle C)
                if (midiNote > 57) // Above A3
                {
                    for (int ledgerNote = 60; ledgerNote <= midiNote; ledgerNote += 4)
                    {
                        double ledgerY = CalculateNoteY(ledgerNote, staffY, false);
                        DrawLedgerLine(ledgerX, ledgerY, ledgerLength);
                    }
                }
                // Below bass staff
                else if (midiNote < 43) // Below G2
                {
                    for (int ledgerNote = 40; ledgerNote >= midiNote; ledgerNote -= 4)
                    {
                        double ledgerY = CalculateNoteY(ledgerNote, staffY, false);
                        DrawLedgerLine(ledgerX, ledgerY, ledgerLength);
                    }
                }
            }
        }
        
        private void DrawLedgerLine(double x, double y, double length)
        {
            Line ledger = new Line
            {
                X1 = x,
                Y1 = y,
                X2 = x + length,
                Y2 = y,
                Stroke = Brushes.Black,
                StrokeThickness = 1
            };
            canvas.Children.Add(ledger);
        }
    }
}