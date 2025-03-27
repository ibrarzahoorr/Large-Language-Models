#!/usr/bin/env python
"""
Simple Text-to-Video Generator using OpenCV
-------------------------------------------
This script creates videos from text input with subtitles 
and background images using only OpenCV.
"""

import os
import sys
import argparse
import textwrap
import random
import time
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

class SimpleTextToVideo:
    """Generate videos from text with basic subtitles and background images."""
    
    def __init__(self, 
                 output_size=(1280, 720),
                 font_path=None,
                 font_size=40,
                 subtitle_color=(255, 255, 255),
                 temp_dir="temp"):
        """
        Initialize the text to video generator.
        
        Args:
            output_size: Video resolution (width, height)
            font_path: Path to TTF font file (or None to use default)
            font_size: Font size for subtitles
            subtitle_color: Color of the subtitle text as RGB tuple
            temp_dir: Directory to store temporary files
        """
        self.output_size = output_size
        self.font_path = font_path
        self.font_size = font_size
        self.subtitle_color = subtitle_color
        self.temp_dir = temp_dir
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def parse_text(self, text, max_words_per_segment=15):
        """
        Parse input text into segments for subtitles.
        
        Args:
            text: Input text to parse
            max_words_per_segment: Maximum number of words per subtitle segment
            
        Returns:
            List of text segments
        """
        # Split text into sentences
        sentences = text.split('.')
        segments = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it's not the last sentence
            if sentence != sentences[-1].strip():
                sentence += '.'
                
            # If sentence is short enough, add it as a segment
            words = sentence.split()
            if len(words) <= max_words_per_segment:
                segments.append(sentence)
            else:
                # Split long sentences into multiple segments
                for i in range(0, len(words), max_words_per_segment):
                    segment = ' '.join(words[i:i + max_words_per_segment])
                    segments.append(segment)
        
        return segments
    
    def fetch_images(self, keywords, num_images=10):
        """
        Fetch image URLs from free sources.
        
        Args:
            keywords: List of keywords to search for images
            num_images: Number of images to fetch
            
        Returns:
            List of image URLs
        """
        # Simple placeholder image URLs using Lorem Picsum
        return [f"https://picsum.photos/{self.output_size[0]}/{self.output_size[1]}?random={i}" 
                for i in range(num_images)]
    
    def download_image(self, url):
        """
        Download an image from a URL and convert to OpenCV format.
        
        Args:
            url: URL of the image
            
        Returns:
            Image as numpy array
        """
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            img = img.resize(self.output_size)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error downloading image: {e}")
            # Return a colored blank image
            blank_image = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
            # Add some random colors to make it less plain
            cv2.rectangle(
                blank_image, 
                (0, 0), 
                (self.output_size[0], self.output_size[1]), 
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
                -1
            )
            return blank_image
    
    def add_subtitle_to_image(self, image, text):
        """
        Add subtitle text to an image.
        
        Args:
            image: OpenCV image as numpy array
            text: Text to add as subtitle
            
        Returns:
            Image with subtitle added
        """
        # Convert OpenCV image to PIL image for better text handling
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use specified font or fall back to default
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, self.font_size)
            else:
                # Try to use a system font
                try:
                    # Default fonts that might be available on different systems
                    system_fonts = [
                        "arial.ttf", "Arial.ttf",
                        "verdana.ttf", "Verdana.ttf",
                        "times.ttf", "Times New Roman.ttf",
                        "cour.ttf", "Courier New.ttf"
                    ]
                    
                    # Font directories on different platforms
                    font_dirs = []
                    if os.name == 'nt':  # Windows
                        font_dirs.append(os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'))
                    else:  # Linux/Mac
                        font_dirs.extend([
                            "/usr/share/fonts",
                            "/usr/local/share/fonts",
                            os.path.expanduser("~/.fonts"),
                            os.path.expanduser("~/Library/Fonts")  # Mac
                        ])
                    
                    # Try to find a usable font
                    font_found = False
                    for font_dir in font_dirs:
                        if os.path.exists(font_dir):
                            for font_name in system_fonts:
                                font_path = os.path.join(font_dir, font_name)
                                if os.path.exists(font_path):
                                    font = ImageFont.truetype(font_path, self.font_size)
                                    font_found = True
                                    break
                        if font_found:
                            break
                    
                    if not font_found:
                        # Last resort: use default PIL font
                        font = ImageFont.load_default()
                        
                except Exception:
                    # If all else fails, use default
                    font = ImageFont.load_default()
        except Exception:
            # If loading TTF fails, use default font
            font = ImageFont.load_default()
        
        # Wrap text to fit in image
        wrapper = textwrap.TextWrapper(width=int(self.output_size[0] * 0.8 / (self.font_size * 0.5)))
        wrapped_text = wrapper.fill(text)
        
        # Get text size for positioning
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position (centered at bottom)
        x = (self.output_size[0] - text_width) // 2
        y = self.output_size[1] - text_height - 50  # 50px from bottom
        
        # Draw text shadow/outline first for better visibility
        shadow_offset = 2
        shadow_color = (0, 0, 0)
        
        # Draw shadow/outline by drawing text multiple times with offsets
        for dx, dy in [(-shadow_offset, -shadow_offset), 
                       (-shadow_offset, 0), 
                       (-shadow_offset, shadow_offset),
                       (0, -shadow_offset), 
                       (0, shadow_offset),
                       (shadow_offset, -shadow_offset), 
                       (shadow_offset, 0), 
                       (shadow_offset, shadow_offset)]:
            draw.text((x + dx, y + dy), wrapped_text, font=font, fill=shadow_color)
        
        # Draw main text
        draw.text((x, y), wrapped_text, font=font, fill=self.subtitle_color)
        
        # Convert back to OpenCV image
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def create_video(self, text, keywords=None, segment_duration=5, output_filename="output.mp4"):
        """
        Create a video from text with subtitles and background images.
        
        Args:
            text: Input text to convert to video
            keywords: Keywords for image search (defaults to extracting from text)
            segment_duration: Duration of each text segment in seconds
            output_filename: Name of the output video file
            
        Returns:
            Path to the generated video file
        """
        # Parse text into segments
        segments = self.parse_text(text)
        
        # Extract keywords from text if not provided
        if not keywords:
            # Simple keyword extraction (exclude common words)
            common_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by"}
            words = [word.lower() for word in text.split() if len(word) > 3]
            keywords = list(set([word for word in words if word not in common_words])[:5])
        
        print("Using keywords:", keywords)
        
        # Fetch background images
        image_urls = self.fetch_images(keywords, num_images=len(segments))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' if MP4 doesn't work
        fps = 30
        output_path = os.path.abspath(output_filename)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        # For each segment, create video frames
        for i, (segment, image_url) in enumerate(zip(segments, image_urls)):
            print(f"Processing segment {i+1}/{len(segments)}: {segment[:30]}...")
            
            # Download background image
            background = self.download_image(image_url)
            
            # Add subtitle to image
            frame = self.add_subtitle_to_image(background, segment)
            
            # Write frames for the duration of the segment
            num_frames = int(segment_duration * fps)
            for _ in range(num_frames):
                video_writer.write(frame)
        
        # Release video writer
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
        return output_path

def display_menu():
    """Display interactive menu for the text-to-video generator."""
    print("\n" + "="*50)
    print(" SIMPLE TEXT TO VIDEO GENERATOR ".center(50, "="))
    print("="*50 + "\n")
    
    # Get script text
    print("Enter your script text below (type 'END' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    script_text = "\n".join(lines)
    
    # Get keywords
    print("\nEnter keywords for image search (comma-separated, or press Enter to extract from text):")
    keywords_input = input().strip()
    
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',')]
    else:
        # Extract keywords from text
        common_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", "is", "are", "was", "were"}
        words = [word.lower() for word in script_text.split() if len(word) > 3]
        keywords = list(set([word for word in words if word not in common_words])[:5])
        print(f"Extracted keywords: {', '.join(keywords)}")
    
    # Output filename
    print("\nEnter output filename (default: output.mp4):")
    output_filename = input().strip()
    if not output_filename:
        output_filename = "output.mp4"
    elif not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Subtitle options
    print("\nSubtitle color (default: white, options: white, yellow, green, red, blue):")
    subtitle_color_name = input().strip().lower()
    color_map = {
        'white': (255, 255, 255),
        'yellow': (255, 255, 0),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255)
    }
    subtitle_color = color_map.get(subtitle_color_name, (255, 255, 255))
    
    print("\nSegment duration in seconds (default: 5.0):")
    segment_duration_input = input().strip()
    try:
        segment_duration = float(segment_duration_input) if segment_duration_input else 5.0
    except ValueError:
        segment_duration = 5.0
        print("Invalid value, using default: 5.0 seconds")
    
    # Create generator with selected options
    generator = SimpleTextToVideo(
        font_size=40,
        subtitle_color=subtitle_color
    )
    
    # Display processing message
    print("\nGenerating video...")
    print("This may take several minutes depending on the script length.")
    print("Downloading images and processing text...")
    
    # Generate video
    output_path = generator.create_video(
        text=script_text,
        keywords=keywords,
        segment_duration=segment_duration,
        output_filename=output_filename
    )
    
    print(f"\nVideo generated successfully!")
    print(f"Saved as: {output_path}")
    
    return output_path

def main():
    """Main function to handle command line arguments or run interactive menu."""
    parser = argparse.ArgumentParser(description='Convert text to video with subtitles')
    parser.add_argument('--text', type=str, help='Input text or path to text file')
    parser.add_argument('--keywords', type=str, help='Keywords for image search (comma separated)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output filename')
    parser.add_argument('--segment-duration', type=float, default=5.0, 
                        help='Duration of each text segment in seconds')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode with guided prompts')
    
    args = parser.parse_args()
    
    # Check if interactive mode is requested
    if args.interactive or len(sys.argv) == 1:  # Run interactive if no args or explicitly requested
        display_menu()
        return
    
    # Command line mode
    # Get input text
    if args.text:
        if os.path.isfile(args.text):
            with open(args.text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.text
    else:
        text = input("Enter the text you want to convert to video: ")
    
    # Parse keywords
    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(',')]
    
    # Create generator
    generator = SimpleTextToVideo()
    
    # Generate video
    print(f"Generating video from text...")
    output_path = generator.create_video(
        text=text,
        keywords=keywords,
        segment_duration=args.segment_duration,
        output_filename=args.output
    )
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main()