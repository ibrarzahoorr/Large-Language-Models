#!/usr/bin/env python
"""
User-Friendly Text-to-Video Generator
------------------------------------
This script creates videos with animated text and effects
with a simplified, foolproof user input system.
"""

import os
import sys
import argparse
import textwrap
import random
import time
import math
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

class TextToVideoGenerator:
    """Generate videos with mixed animations and easy user input."""
    
    def __init__(self, 
                 output_size=(1280, 720),
                 font_size=60,
                 subtitle_color=(255, 255, 255),
                 temp_dir="temp"):
        """Initialize the text to video generator."""
        self.output_size = output_size
        self.font_size = font_size
        self.subtitle_color = subtitle_color
        self.temp_dir = temp_dir
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Cache for found font
        self.found_font = None
    
    def parse_text(self, text, max_words_per_segment=10):
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
    
    def load_video_frames(self, video_path, segment_duration, fps):
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of segment in seconds
            fps: Frames per second
            
        Returns:
            List of frames (numpy arrays)
        """
        frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                # Return a colored blank image sequence
                for _ in range(int(segment_duration * fps)):
                    blank_image = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
                    cv2.rectangle(
                        blank_image, 
                        (0, 0), 
                        (self.output_size[0], self.output_size[1]), 
                        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
                        -1
                    )
                    frames.append(blank_image)
                return frames
            
            # Calculate total frames needed
            total_frames_needed = int(segment_duration * fps)
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame step to match segment duration
            # if video is longer than segment
            if video_frame_count > total_frames_needed:
                step = video_frame_count / total_frames_needed
                frame_indices = [int(i * step) for i in range(total_frames_needed)]
            else:
                # If video is shorter, we'll loop it
                frame_indices = list(range(video_frame_count)) * (total_frames_needed // video_frame_count + 1)
                frame_indices = frame_indices[:total_frames_needed]
            
            for frame_idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame if needed
                    if frame.shape[0] != self.output_size[1] or frame.shape[1] != self.output_size[0]:
                        frame = cv2.resize(frame, self.output_size)
                    
                    frames.append(frame)
                else:
                    # If frame read failed, add a blank frame
                    blank_frame = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
                    frames.append(blank_frame)
            
            # Release video capture
            cap.release()
            
            return frames
            
        except Exception as e:
            print(f"Error loading video frames: {e}")
            # Return a colored blank image sequence
            for _ in range(int(segment_duration * fps)):
                blank_image = np.ones((self.output_size[1], self.output_size[0], 3), dtype=np.uint8) * 255
                cv2.rectangle(
                    blank_image, 
                    (0, 0), 
                    (self.output_size[0], self.output_size[1]), 
                    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 
                    -1
                )
                frames.append(blank_image)
            return frames
    
    def find_font(self):
        """Find a suitable font on the system."""
        if self.found_font:
            return self.found_font
            
        try:
            # Try to use a system font
            system_fonts = [
                "arial.ttf", "Arial.ttf",
                "impact.ttf", "Impact.ttf",  # Good for memes and bold statements
                "georgia.ttf", "Georgia.ttf",  # Elegant serif font
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
                            self.found_font = font
                            font_found = True
                            break
                if font_found:
                    break
            
            if not font_found:
                # Last resort: use default PIL font
                font = ImageFont.load_default()
                self.found_font = font
                    
        except Exception:
            # If all else fails, use default
            font = ImageFont.load_default()
            self.found_font = font
            
        return self.found_font
    
    def apply_effects(self, frame, frame_idx, total_frames, effect_type="zoom"):
        """
        Apply effects to a background frame.
        
        Args:
            frame: OpenCV image
            frame_idx: Current frame index
            total_frames: Total frames for this segment
            effect_type: Type of effect to apply
            
        Returns:
            Processed frame
        """
        # Convert to PIL for easier image processing
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        progress = frame_idx / total_frames
        
        if effect_type == "zoom":
            # Zoom effect
            zoom_factor = 1.0 + (0.1 * math.sin(progress * math.pi * 2))
            new_size = (int(pil_img.width * zoom_factor), int(pil_img.height * zoom_factor))
            zoomed = pil_img.resize(new_size, Image.LANCZOS)
            
            # Center crop
            left = (zoomed.width - pil_img.width) // 2
            top = (zoomed.height - pil_img.height) // 2
            right = left + pil_img.width
            bottom = top + pil_img.height
            
            # Make sure we don't go out of bounds
            if left < 0: left = 0
            if top < 0: top = 0
            if right > zoomed.width: right = zoomed.width
            if bottom > zoomed.height: bottom = zoomed.height
            
            processed = zoomed.crop((left, top, right, bottom))
            
            # Resize back to original if needed
            if processed.width != pil_img.width or processed.height != pil_img.height:
                processed = processed.resize(pil_img.size, Image.LANCZOS)
        
        elif effect_type == "tiktok":
            # TikTok-style effect
            zoom_factor = 1.0 + (0.05 * math.sin(progress * math.pi * 4))
            shake_x = int(5 * math.sin(progress * math.pi * 8))
            shake_y = int(3 * math.cos(progress * math.pi * 6))
            
            new_size = (int(pil_img.width * zoom_factor), int(pil_img.height * zoom_factor))
            zoomed = pil_img.resize(new_size, Image.LANCZOS)
            
            # Center crop with shake
            left = (zoomed.width - pil_img.width) // 2 + shake_x
            top = (zoomed.height - pil_img.height) // 2 + shake_y
            right = left + pil_img.width
            bottom = top + pil_img.height
            
            # Make sure we don't go out of bounds
            if left < 0: left, right = 0, pil_img.width
            if top < 0: top, bottom = 0, pil_img.height
            if right > zoomed.width: right, left = zoomed.width, zoomed.width - pil_img.width
            if bottom > zoomed.height: bottom, top = zoomed.height, zoomed.height - pil_img.height
            
            processed = zoomed.crop((left, top, right, bottom))
            
            # Ensure correct size
            if processed.width != pil_img.width or processed.height != pil_img.height:
                processed = processed.resize(pil_img.size, Image.LANCZOS)
        
        elif effect_type == "smooth":
            # Subtle smooth effect for professional look
            zoom_factor = 1.0 + (0.03 * (0.5 - abs(0.5 - progress)))  # Peaks in middle
            
            new_size = (int(pil_img.width * zoom_factor), int(pil_img.height * zoom_factor))
            zoomed = pil_img.resize(new_size, Image.LANCZOS)
            
            # Center crop
            left = (zoomed.width - pil_img.width) // 2
            top = (zoomed.height - pil_img.height) // 2
            right = left + pil_img.width
            bottom = top + pil_img.height
            
            # Ensure we stay in bounds
            if left < 0: left = 0
            if top < 0: top = 0
            if right > zoomed.width: right = zoomed.width
            if bottom > zoomed.height: bottom = zoomed.height
            
            processed = zoomed.crop((left, top, right, bottom))
            
            # Resize if needed
            if processed.width != pil_img.width or processed.height != pil_img.height:
                processed = processed.resize(pil_img.size, Image.LANCZOS)
        
        else:  # No effect or default
            processed = pil_img
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)
    
    def create_text_animation(self, text, frame_idx, total_frames, animation_type="fade"):
        """
        Create animated text overlay.
        
        Args:
            text: Text to animate
            frame_idx: Current frame index
            total_frames: Total frames for this segment
            animation_type: Type of text animation
            
        Returns:
            PIL Image with transparent background containing text
        """
        # Create transparent image
        text_img = Image.new('RGBA', self.output_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Get font
        font = self.find_font()
        
        # Wrap text to fit in image
        wrapper = textwrap.TextWrapper(width=int(self.output_size[0] * 0.8 / (self.font_size * 0.5)))
        wrapped_text = wrapper.fill(text)
        
        # Get text size for positioning
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate center position
        x = (self.output_size[0] - text_width) // 2
        y = (self.output_size[1] - text_height) // 2
        
        progress = frame_idx / total_frames
        
        # Animation variables
        text_color = self.subtitle_color
        outline_color = (0, 0, 0)
        alpha = 255  # Full opacity
        outline_width = 2
        scale = 1.0
        y_offset = 0
        
        # Apply animation effects
        if animation_type == "fade":
            # Fade in/out
            if progress < 0.2:  # First 20% - fade in
                alpha = int(255 * (progress / 0.2))
                scale = 0.9 + 0.1 * (progress / 0.2)  # Slightly grow while fading in
            elif progress > 0.8:  # Last 20% - fade out
                alpha = int(255 * ((1.0 - progress) / 0.2))
                scale = 0.9 + 0.1 * ((1.0 - progress) / 0.2)  # Slightly shrink while fading out
        
        elif animation_type == "pop":
            # Pop animation effect
            if progress < 0.3:
                scale = 0.7 + progress * 1.5  # Overshoot
            elif progress < 0.5:
                scale = 1.2 - (progress - 0.3) * 1  # Settle back
            
            # Add subtle bounce
            y_offset = int(10 * math.sin(progress * math.pi * 3))
            
            # Fade in/out
            if progress < 0.1:
                alpha = int(255 * (progress / 0.1))
            elif progress > 0.9:
                alpha = int(255 * ((1.0 - progress) / 0.1))
        
        elif animation_type == "type":
            # Typewriter effect - reveal characters gradually
            char_count = len(wrapped_text)
            visible_chars = int(char_count * min(1.0, progress * 1.5))
            wrapped_text = wrapped_text[:visible_chars]
            
            # Add fade in/out
            if progress < 0.1:
                alpha = int(255 * (progress / 0.1))
            elif progress > 0.9:
                alpha = int(255 * ((1.0 - progress) / 0.1))
        
        elif animation_type == "highlight":
            # Professional highlight effect
            
            # Subtle scale change
            scale = 1.0 + 0.05 * math.sin(progress * math.pi)
            
            # Subtle vertical motion
            y_offset = int(5 * math.sin(progress * math.pi * 2))
            
            # Color emphasis at certain points
            if 0.4 < progress < 0.6:
                # Brightness boost at middle
                boost_factor = 1.0 - abs((progress - 0.5) / 0.1) * 2  # 0 to 1 range
                r, g, b = text_color
                r = min(255, int(r + 50 * boost_factor))
                g = min(255, int(g + 50 * boost_factor))
                b = min(255, int(b + 50 * boost_factor))
                text_color = (r, g, b)
            
            # Fade in/out
            if progress < 0.1:
                alpha = int(255 * (progress / 0.1))
            elif progress > 0.9:
                alpha = int(255 * ((1.0 - progress) / 0.1))
        
        else:  # Default or "mixed"
            # Combine multiple effects
            
            # Scale effect - slightly grow and shrink
            scale = 1.0 + 0.1 * math.sin(progress * math.pi * 2)
            
            # Vertical motion
            y_offset = int(8 * math.sin(progress * math.pi * 3))
            
            # Fade in/out
            if progress < 0.1:
                alpha = int(255 * (progress / 0.1))
            elif progress > 0.9:
                alpha = int(255 * ((1.0 - progress) / 0.1))
        
        # Apply scale transformation
        if scale != 1.0:
            scaled_font_size = int(self.font_size * scale)
            try:
                scaled_font = ImageFont.truetype(font.path, scaled_font_size)
            except:
                # Fallback if we can't get the font path
                scaled_font = font
            
            # Recalculate position with new size
            bbox = draw.textbbox((0, 0), wrapped_text, font=scaled_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.output_size[0] - text_width) // 2
            y = (self.output_size[1] - text_height) // 2
            
            font = scaled_font
        
        # Apply y-offset
        y += y_offset
        
        # Draw outline shadow for better visibility
        for dx, dy in [(-outline_width, -outline_width), 
                    (-outline_width, 0), 
                    (-outline_width, outline_width),
                    (0, -outline_width), 
                    (0, outline_width),
                    (outline_width, -outline_width), 
                    (outline_width, 0), 
                    (outline_width, outline_width)]:
            draw.text((x + dx, y + dy), wrapped_text, font=font, fill=(*outline_color, alpha))
        
        # Draw main text
        draw.text((x, y), wrapped_text, font=font, fill=(*text_color, alpha))
        
        return text_img
    
    def blend_images(self, background, overlay):
        """
        Blend a background OpenCV image with a RGBA PIL overlay image.
        
        Args:
            background: OpenCV BGR image
            overlay: PIL RGBA image
            
        Returns:
            Blended OpenCV BGR image
        """
        # Convert background to RGBA PIL image
        bg_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)).convert('RGBA')
        
        # Blend the images
        blended = Image.alpha_composite(bg_pil, overlay)
        
        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(blended.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    def create_video(self, text, keywords=None, segment_duration=5, 
                     output_filename="output.mp4", 
                     bg_effect="zoom", text_animation="fade",
                     use_video_backgrounds=False, video_paths=None):
        """
        Create a video from text with animations.
        
        Args:
            text: Input text to convert to video
            keywords: Keywords for image search (defaults to extracting from text)
            segment_duration: Duration of each text segment in seconds
            output_filename: Name of the output video file
            bg_effect: Background effect to use
            text_animation: Text animation type to use
            use_video_backgrounds: Whether to use video backgrounds instead of images
            video_paths: List of paths to video files to use as backgrounds
            
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
            
            # Filter out common words
            filtered_words = [word for word in words if word not in common_words]
            
            # Get unique words
            keywords = list(set(filtered_words))
            
            # Take up to 5 keywords
            if len(keywords) > 5:
                keywords = keywords[:5]
        
        print("Using keywords:", keywords)
        
        # Get backgrounds (videos or images)
        if use_video_backgrounds and video_paths:
            # Use provided video paths
            if len(video_paths) < len(segments):
                # If we don't have enough videos, repeat them
                video_paths = (video_paths * ((len(segments) // len(video_paths)) + 1))[:len(segments)]
            backgrounds = video_paths
            is_video = [True] * len(backgrounds)
        else:
            # Use images
            image_urls = self.fetch_images(keywords, num_images=len(segments))
            backgrounds = image_urls
            is_video = [False] * len(backgrounds)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' if MP4 doesn't work
        fps = 30
        output_path = os.path.abspath(output_filename)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        # Process each segment
        for i, (segment, background, is_vid) in enumerate(zip(segments, backgrounds, is_video)):
            print(f"Processing segment {i+1}/{len(segments)}")
            print(f"  Text: {segment[:30]}...")
            
            # Get background frames
            if is_vid:
                # Load video frames
                print(f"  Loading video background: {background}")
                bg_frames = self.load_video_frames(background, segment_duration, fps)
            else:
                # Download and duplicate image
                print(f"  Downloading image background")
                bg_image = self.download_image(background)
                num_frames = int(segment_duration * fps)
                bg_frames = [bg_image.copy() for _ in range(num_frames)]
            
            # Process frames
            for frame_idx, bg_frame in enumerate(bg_frames):
                num_frames = len(bg_frames)
                
                # 1. Apply background effects
                processed_bg = self.apply_effects(
                    bg_frame, 
                    effect_type=bg_effect,
                    frame_idx=frame_idx,
                    total_frames=num_frames
                )
                
                # 2. Create text animation
                text_overlay = self.create_text_animation(
                    segment,
                    frame_idx=frame_idx,
                    total_frames=num_frames,
                    animation_type=text_animation
                )
                
                # 3. Blend background and text
                final_frame = self.blend_images(processed_bg, text_overlay)
                
                # 4. Add frame to video
                video_writer.write(final_frame)
                
                # Progress indicator (update every 10%)
                if frame_idx % (num_frames // 10 or 1) == 0 or frame_idx == num_frames - 1:
                    progress_pct = int((frame_idx + 1) / num_frames * 100)
                    print(f"  Progress: {progress_pct}%", end="\r")
            
            print()  # New line after progress indicator
        
        # Release video writer
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
        return output_path


def get_user_text_input():
    """Get text input from the user with clear instructions."""
    print("\n" + "="*70)
    print(" TEXT TO VIDEO GENERATOR ".center(70, "="))
    print("="*70)
    
    print("\nPlease enter your text for the video below:")
    print("(Type your text line by line, then type 'END' or press Ctrl+D when finished)")
    print("-" * 50)
    
    lines = []
    try:
        while True:
            line = input("> ")
            if line.strip().upper() == 'END':
                break
            lines.append(line)
    except EOFError:  # Handle Ctrl+D
        pass
    
    text = "\n".join(lines)
    
    if not text.strip():
        print("\nNo text entered. Using example text...")
        text = "This is an example video.\nCreated with the Text to Video Generator.\nIt converts your text into animated videos."
    
    return text


def run_interactive_mode():
    """Run the generator in interactive mode with clear user prompts."""
    print("\n" + "="*70)
    print(" TEXT TO VIDEO GENERATOR ".center(70, "="))
    print("="*70)
    
    # Step 1: Get text input
    print("\nSTEP 1: Enter your text for the video")
    print("Type your text line by line, then type 'END' or press Ctrl+D when finished")
    print("-" * 50)
    
    lines = []
    try:
        while True:
            line = input("> ")
            if line.strip().upper() == 'END':
                break
            lines.append(line)
    except EOFError:  # Handle Ctrl+D
        pass
    
    script_text = "\n".join(lines)
    
    if not script_text.strip():
        print("\nNo text entered. Using example text...")
        script_text = "This is an example video.\nCreated with the Text to Video Generator.\nIt converts your text into animated videos."
    
    # Step 2: Background options
    print("\nSTEP 2: Choose background type")
    print("1. Images from the web (default, easiest option)")
    print("2. Your own video files")
    
    bg_choice = input("Enter your choice (1-2): ").strip()
    use_video_backgrounds = bg_choice == "2"
    video_paths = []
    
    if use_video_backgrounds:
        print("\nEnter paths to video files (one per line)")
        print("Type 'DONE' when finished")
        print("-" * 50)
        
        while True:
            video_path = input("> ")
            if video_path.strip().upper() == 'DONE':
                break
            if os.path.exists(video_path):
                video_paths.append(video_path)
            else:
                print(f"Warning: File '{video_path}' not found. Please enter a valid path.")
        
        if not video_paths:
            print("\nNo valid video paths provided. Using images instead.")
            use_video_backgrounds = False
    
    # Step 3: Animation options
    print("\nSTEP 3: Choose text animation style")
    print("1. Mixed (recommended)")
    print("2. Fade (smooth fade in/out)")
    print("3. Pop (bouncy attention-grabbing)")
    print("4. Type (typewriter effect)")
    print("5. Highlight (professional emphasis)")
    
    animation_choice = input("Enter your choice (1-5): ").strip()
    animation_map = {
        "1": "mixed",
        "2": "fade",
        "3": "pop",
        "4": "type",
        "5": "highlight"
    }
    text_animation = animation_map.get(animation_choice, "mixed")
    
    # Step 4: Background effect
    print("\nSTEP 4: Choose background effect")
    print("1. Zoom (smooth zoom effect)")
    print("2. TikTok (trendy shake and zoom)")
    print("3. Smooth (subtle professional effect)")
    print("4. None (static background)")
    
    effect_choice = input("Enter your choice (1-4): ").strip()
    effect_map = {
        "1": "zoom",
        "2": "tiktok",
        "3": "smooth",
        "4": "none"
    }
    bg_effect = effect_map.get(effect_choice, "zoom")
    
    # Step 5: Text color
    print("\nSTEP 5: Choose text color")
    print("1. White (default, works with most backgrounds)")
    print("2. Yellow (popular on social media)")
    print("3. Green")
    print("4. Red (attention-grabbing)")
    print("5. Blue")
    print("6. Pink (trending on TikTok/Instagram)")
    
    color_choice = input("Enter your choice (1-6): ").strip()
    color_map = {
        "1": (255, 255, 255),
        "2": (255, 255, 0),
        "3": (0, 255, 0),
        "4": (255, 0, 0),
        "5": (0, 0, 255),
        "6": (255, 0, 255)
    }
    subtitle_color = color_map.get(color_choice, (255, 255, 255))
    
    # Step 6: Output settings
    print("\nSTEP 6: Output settings")
    print("Enter output filename (default: video_output.mp4):")
    output_filename = input("> ").strip()
    if not output_filename:
        output_filename = "video_output.mp4"
    elif not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    print("\nEnter segment duration in seconds (default: 5.0):")
    segment_duration_input = input("> ").strip()
    try:
        segment_duration = float(segment_duration_input) if segment_duration_input else 5.0
    except ValueError:
        segment_duration = 5.0
        print("Invalid value, using default: 5.0 seconds")
    
    # Extract keywords from text
    common_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", "is", "are", "was", "were"}
    words = [word.lower() for word in script_text.split() if len(word) > 3]
    
    # Filter out common words
    filtered_words = [word for word in words if word not in common_words]
    
    # Get unique words
    keywords = list(set(filtered_words))
    
    # Take up to 5 keywords
    if len(keywords) > 5:
        keywords = keywords[:5]
    
    print(f"\nExtracted keywords: {', '.join(keywords)}")
    print("Want to use different keywords? (Enter new keywords or press Enter to keep these)")
    keywords_input = input("> ").strip()
    if keywords_input:
        keywords = [k.strip() for k in keywords_input.split(',')]
    
    # Create generator
    print("\nInitializing video generator...")
    generator = TextToVideoGenerator(
        font_size=60,
        subtitle_color=subtitle_color
    )
    
    # Display processing message
    print("\nGenerating video...")
    print("This may take several minutes depending on the script length.")
    print("-" * 50)
    
    # Generate video
    output_path = generator.create_video(
        text=script_text,
        keywords=keywords,
        segment_duration=segment_duration,
        output_filename=output_filename,
        bg_effect=bg_effect,
        text_animation=text_animation,
        use_video_backgrounds=use_video_backgrounds,
        video_paths=video_paths
    )
    
    print("\nVideo generated successfully!")
    print(f"Saved as: {output_path}")
    
    return output_path


def run_simple_mode():
    """Run the generator in very simple mode with minimal input required."""
    print("\n" + "="*70)
    print(" QUICK TEXT TO VIDEO GENERATOR ".center(70, "="))
    print("="*70)
    
    # Get text input
    print("\nEnter your text for the video (type 'END' on a new line when finished):")
    
    lines = []
    try:
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
    except EOFError:  # Handle Ctrl+D
        pass
    
    script_text = "\n".join(lines)
    
    if not script_text.strip():
        script_text = "This is an example video.\nCreated with the Text to Video Generator.\nIt converts your text into animated videos."
        print(f"Using example text: {script_text}")
    
    # Ask for filename
    print("\nEnter output filename (or press Enter for default 'quick_video.mp4'):")
    output_filename = input().strip()
    if not output_filename:
        output_filename = "quick_video.mp4"
    elif not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Create generator with default settings
    generator = TextToVideoGenerator(
        font_size=60,
        subtitle_color=(255, 255, 0)  # Yellow text by default
    )
    
    # Extract keywords
    common_words = {"the", "and", "a", "an", "in", "on", "at", "to", "for", "with", "by", "is", "are", "was", "were"}
    words = [word.lower() for word in script_text.split() if len(word) > 3]
    
    # Filter out common words
    filtered_words = [word for word in words if word not in common_words]
    
    # Get unique words
    keywords = list(set(filtered_words))
    
    # Take up to 5 keywords
    if len(keywords) > 5:
        keywords = keywords[:5]
    
    print(f"Using keywords: {', '.join(keywords)}")
    
    # Generate video with default settings
    print("\nGenerating video with default settings...")
    print("(Using mixed animations, zoom effects, and 5 second segments)")
    
    output_path = generator.create_video(
        text=script_text,
        keywords=keywords,
        segment_duration=5.0,
        output_filename=output_filename,
        bg_effect="zoom",
        text_animation="mixed"
    )
    
    print("\nVideo generated successfully!")
    print(f"Saved as: {output_path}")
    
    return output_path


def main():
    """Main function with multiple modes of operation."""
    parser = argparse.ArgumentParser(description='User-friendly text-to-video generator')
    parser.add_argument('--text', type=str, help='Input text or path to text file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output filename')
    parser.add_argument('--simple', action='store_true', help='Run in simple mode with minimal prompts')
    parser.add_argument('--advanced', action='store_true', help='Run in advanced interactive mode')
    
    args = parser.parse_args()
    
    # Simple mode takes precedence if explicitly requested
    if args.simple:
        run_simple_mode()
        return
    
    # Advanced mode is the default for interactive use
    if args.advanced or len(sys.argv) == 1:
        run_interactive_mode()
        return
    
    # Command line mode
    if args.text:
        if os.path.isfile(args.text):
            with open(args.text, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.text
    else:
        # If no text provided, prompt user
        text = get_user_text_input()
    
    # Create generator with default settings
    generator = TextToVideoGenerator()
    
    # Generate video
    print("Generating video...")
    output_path = generator.create_video(
        text=text,
        output_filename=args.output
    )
    
    print(f"Video generated successfully: {output_path}")


if __name__ == "__main__":
    main()