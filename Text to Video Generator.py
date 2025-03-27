#!/usr/bin/env python
"""
Text to Video Generator
-----------------------
This script converts text input into a video with subtitles,
using free images from Unsplash or Pexels as background.
"""

import os
import textwrap
import argparse
import random
import re
from typing import List, Dict, Tuple
import time
import requests
import json
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from moviepy.editor import (
    TextClip, ImageClip, CompositeVideoClip, concatenate_videoclips
)
from moviepy.video.fx.all import fadein, fadeout

class TextToVideoGenerator:
    """Generate videos from text with beautiful subtitles and free background images."""
    
    def __init__(self, 
                 font: str = "Arial-Bold",  # Changed to a more common font
                 font_size: int = 50,
                 output_size: tuple = (1920, 1080),
                 subtitle_color: str = 'white',
                 temp_dir: str = "temp"):
        """
        Initialize the text to video generator.
        
        Args:
            font: Font name for subtitles
            font_size: Font size for subtitles
            output_size: Video resolution (width, height)
            subtitle_color: Color of the subtitle text
            temp_dir: Directory to store temporary files
        """
        self.font = font
        self.font_size = font_size
        self.output_size = output_size
        self.subtitle_color = subtitle_color
        self.temp_dir = temp_dir
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def parse_text(self, text: str, max_words_per_segment: int = 15) -> List[str]:
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
    
    def fetch_images(self, keywords: List[str], num_images: int = 10) -> List[str]:
        """
        Fetch image URLs from free sources without API keys.
        
        Args:
            keywords: List of keywords to search for images
            num_images: Number of images to fetch
            
        Returns:
            List of image URLs
        """
        image_urls = []
        query = "+".join(keywords)
        
        # Sources that don't require API keys
        sources = [
            "pixabay",
            "pexels_free",
            "unsplash_free",
            "picsum"
        ]
        
        try:
            # 1. Try Pixabay (no API key required for search, only for download)
            if "pixabay" in sources and len(image_urls) < num_images:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    search_url = f"https://pixabay.com/images/search/{query}/"
                    response = requests.get(search_url, headers=headers)
                    
                    # Basic parsing to extract image URLs
                    if response.status_code == 200:
                        # Find image URLs in the HTML response
                        import re
                        img_pattern = r'img srcset="(https://cdn\.pixabay\.com/photo/[^"]+)'
                        matches = re.findall(img_pattern, response.text)
                        
                        # Extract unique image URLs
                        for match in matches:
                            # Get highest resolution version
                            base_url = match.split(' ')[0]
                            if base_url not in image_urls:
                                image_urls.append(base_url)
                                if len(image_urls) >= num_images:
                                    break
                except Exception as e:
                    print(f"Error fetching from Pixabay: {e}")
            
            # 2. Try Pexels without API
            if "pexels_free" in sources and len(image_urls) < num_images:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    search_url = f"https://www.pexels.com/search/{query}/"
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        import re
                        # Pattern to match Pexels image URLs
                        img_pattern = r'src="(https://images\.pexels\.com/photos/[^"]+)'
                        matches = re.findall(img_pattern, response.text)
                        
                        for match in matches:
                            if match not in image_urls:
                                image_urls.append(match)
                                if len(image_urls) >= num_images:
                                    break
                except Exception as e:
                    print(f"Error fetching from Pexels: {e}")
            
            # 3. Try Unsplash without API
            if "unsplash_free" in sources and len(image_urls) < num_images:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    search_url = f"https://unsplash.com/s/photos/{query}"
                    response = requests.get(search_url, headers=headers)
                    
                    if response.status_code == 200:
                        import re
                        # Pattern to match Unsplash image URLs
                        img_pattern = r'srcSet="(https://images\.unsplash\.com/photo-[^?]+)'
                        matches = re.findall(img_pattern, response.text)
                        
                        for match in matches:
                            if match not in image_urls:
                                image_urls.append(match)
                                if len(image_urls) >= num_images:
                                    break
                except Exception as e:
                    print(f"Error fetching from Unsplash: {e}")
            
            # 4. Fallback to Picsum (Lorem Picsum) for guaranteed images
            remaining = num_images - len(image_urls)
            if remaining > 0:
                for i in range(remaining):
                    # Random high-quality images
                    image_urls.append(f"https://picsum.photos/{self.output_size[0]}/{self.output_size[1]}?random={i}")
            
            return image_urls
                
        except Exception as e:
            print(f"Error fetching images: {e}")
            # Fallback to placeholder images
            return [f"https://picsum.photos/{self.output_size[0]}/{self.output_size[1]}?random={i}" 
                    for i in range(num_images)]
    
    def download_image(self, url: str) -> np.ndarray:
        """
        Download an image from a URL and convert to OpenCV format.
        
        Args:
            url: URL of the image
            
        Returns:
            Image as numpy array
        """
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.resize(self.output_size)
            return np.array(img)
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
    
    def create_subtitle_clip(self, text: str, duration: float) -> TextClip:
        """
        Create a subtitle clip with the specified text.
        
        Args:
            text: Text for the subtitle
            duration: Duration of the subtitle in seconds
            
        Returns:
            TextClip object
        """
        # Wrap text to multiple lines if needed
        wrapped_text = textwrap.fill(text, width=40)
        
                    # Create text clip with shadow for better visibility
        try:
            # First attempt with specified font
            txt_clip = TextClip(
                wrapped_text,
                fontsize=self.font_size,
                font=self.font,
                color=self.subtitle_color,
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(self.output_size[0] - 100, None),
                align='center'
            ).set_duration(duration)
        except Exception as e:
            print(f"Error with specified font: {e}, trying with default font")
            # Fallback to a system default font if specified font fails
            try:
                txt_clip = TextClip(
                    wrapped_text,
                    fontsize=self.font_size,
                    color=self.subtitle_color,
                    stroke_color='black',
                    stroke_width=2,
                    method='caption',
                    size=(self.output_size[0] - 100, None),
                    align='center'
                ).set_duration(duration)
            except Exception as e2:
                print(f"Error with default font: {e2}, trying basic method")
                # Last resort - most basic text method
                txt_clip = TextClip(
                    wrapped_text,
                    fontsize=self.font_size,
                    color=self.subtitle_color,
                    method='label',
                    size=(self.output_size[0] - 100, None),
                    align='center'
                ).set_duration(duration)
        
        # Add fade in/out effects for smoother transitions
        txt_clip = txt_clip.fx(fadein, 0.5).fx(fadeout, 0.5)
        
        # Position at the bottom with some padding
        return txt_clip.set_position(('center', 'bottom'))
    
    def create_video(self, 
                    text: str, 
                    keywords: List[str] = None,
                    segment_duration: float = 5.0,
                    output_filename: str = "output.mp4") -> str:
        """
        Create a video from text with subtitles and background images.
        
        Args:
            text: Input text to convert to video
            keywords: Keywords for image search (defaults to important words from text)
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
        
        # Fetch background images
        image_urls = self.fetch_images(keywords, num_images=len(segments))
        
        # Create video clips for each segment
        video_clips = []
        
        for i, (segment, image_url) in enumerate(zip(segments, image_urls)):
            # Download and process image
            img = self.download_image(image_url)
            
            # Convert OpenCV image to MoviePy format
            img_clip = ImageClip(img).set_duration(segment_duration)
            
            # Add subtle zoom effect for visual interest
            zoom_factor = 1.05
            img_clip = img_clip.resize(lambda t: zoom_factor - (zoom_factor-1)*t/segment_duration)
            
            # Add fade in/out effects for smoother transitions
            img_clip = img_clip.fx(fadein, 0.5).fx(fadeout, 0.5)
            
            # Create subtitle clip
            subtitle = self.create_subtitle_clip(segment, segment_duration)
            
            # Combine image and subtitle
            video_clip = CompositeVideoClip([img_clip, subtitle])
            video_clips.append(video_clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(video_clips, method="compose")
        
        # Write output file
        output_path = os.path.join(os.getcwd(), output_filename)
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio=False,
            threads=4,
            preset='medium'
        )
        
        # Clean up MoviePy clips
        final_clip.close()
        for clip in video_clips:
            clip.close()
        
        return output_path


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Convert text to video with subtitles')
    parser.add_argument('--text', type=str, help='Input text or path to text file')
    parser.add_argument('--keywords', type=str, help='Keywords for image search (comma separated)')
    parser.add_argument('--font', type=str, default='Arial-Bold', help='Font for subtitles')
    parser.add_argument('--font-size', type=int, default=50, help='Font size for subtitles')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output filename')
    parser.add_argument('--segment-duration', type=float, default=5.0, 
                        help='Duration of each text segment in seconds')
    parser.add_argument('--subtitle-color', type=str, default='white', 
                        help='Color of subtitles (e.g., white, yellow)')
    
    args = parser.parse_args()
    
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
    generator = TextToVideoGenerator(
        font=args.font,
        font_size=args.font_size,
        subtitle_color=args.subtitle_color
    )
    
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
