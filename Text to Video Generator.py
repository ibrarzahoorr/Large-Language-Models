#!/usr/bin/env python
"""
Animated Text-to-Video Generator with Trending Effects
-----------------------------------------------------
This script creates videos with animated text effects and
trending animations using only OpenCV.
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

class AnimatedTextToVideo:
    """Generate videos with animated text effects and trending animations."""
    
    def __init__(self, 
                 output_size=(1280, 720),
                 font_path=None,
                 font_size=60,
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
    
    def find_font(self):
        """Find a suitable font on the system."""
        if self.found_font:
            return self.found_font
            
        # Try to use specified font or fall back to default
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, self.font_size)
                self.found_font = font
                return font
            else:
                # Try to use a system font
                try:
                    # Default fonts that might be available on different systems
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
        except Exception:
            # If loading TTF fails, use default font
            font = ImageFont.load_default()
            self.found_font = font
            
        return self.found_font
    
    def apply_image_effects(self, img, effect_type="zoom", frame_idx=0, total_frames=30):
        """
        Apply trending effects to background image.
        
        Args:
            img: OpenCV image
            effect_type: Type of effect to apply
            frame_idx: Current frame index
            total_frames: Total frames for this segment
            
        Returns:
            Processed image
        """
        # Convert to PIL for easier image processing
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        progress = frame_idx / total_frames
        
        if effect_type == "zoom":
            # Zoom in effect
            zoom_factor = 1.0 + (0.2 * progress)  # Start at 1.0, end at 1.2
            new_size = (int(pil_img.width * zoom_factor), int(pil_img.height * zoom_factor))
            zoomed = pil_img.resize(new_size, Image.LANCZOS)
            
            # Center crop
            left = (zoomed.width - pil_img.width) // 2
            top = (zoomed.height - pil_img.height) // 2
            right = left + pil_img.width
            bottom = top + pil_img.height
            cropped = zoomed.crop((left, top, right, bottom))
            
            processed = cropped
            
        elif effect_type == "pan":
            # Pan effect (horizontal movement)
            max_offset = int(pil_img.width * 0.1)  # 10% of width
            offset = int(max_offset * math.sin(progress * math.pi))
            
            # Create a new image with same size
            new_img = Image.new('RGB', pil_img.size)
            new_img.paste(pil_img, (offset, 0))
            
            # Fill the gap on the other side
            if offset > 0:
                new_img.paste(pil_img.crop((pil_img.width - offset, 0, pil_img.width, pil_img.height)), (0, 0))
            elif offset < 0:
                new_img.paste(pil_img.crop((0, 0, -offset, pil_img.height)), (pil_img.width + offset, 0))
                
            processed = new_img
            
        elif effect_type == "fade":
            # Fade in/out effect
            alpha = 1.0
            if frame_idx < total_frames * 0.2:  # First 20% - fade in
                alpha = frame_idx / (total_frames * 0.2)
            elif frame_idx > total_frames * 0.8:  # Last 20% - fade out
                alpha = (total_frames - frame_idx) / (total_frames * 0.2)
                
            enhancer = ImageEnhance.Brightness(pil_img)
            processed = enhancer.enhance(0.3 + 0.7 * alpha)
            
        elif effect_type == "blur":
            # Dynamic blur effect
            if frame_idx < total_frames / 2:
                # Increasing blur
                blur_radius = (frame_idx / (total_frames / 2)) * 5
            else:
                # Decreasing blur
                blur_radius = ((total_frames - frame_idx) / (total_frames / 2)) * 5
                
            processed = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        elif effect_type == "tiktok":
            # TikTok-style zoom & shake effect
            zoom_factor = 1.0 + (0.05 * math.sin(progress * math.pi * 4))
            shake_x = int(5 * math.sin(progress * math.pi * 8))
            shake_y = int(5 * math.cos(progress * math.pi * 6))
            
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
            
        elif effect_type == "glitch":
            # Glitch effect
            processed = pil_img.copy()
            
            # Only apply glitch occasionally
            if random.random() < 0.2:
                # RGB shift
                r, g, b = processed.split()
                r_shift = random.randint(-10, 10)
                b_shift = random.randint(-10, 10)
                
                # Create a new image with shifted channels
                processed = Image.merge('RGB', (
                    r.transform(r.size, Image.AFFINE, (1, 0, r_shift, 0, 1, 0)),
                    g,
                    b.transform(b.size, Image.AFFINE, (1, 0, b_shift, 0, 1, 0))
                ))
                
                # Add random noise/static
                if random.random() < 0.3:
                    noise = Image.effect_noise(processed.size, 20)
                    noise = noise.convert('RGB')
                    processed = Image.blend(processed, noise, 0.1)
                    
        else:  # default or "none"
            processed = pil_img
            
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)
    
    def create_text_animation(self, text, frame_idx, total_frames, animation_type="fade_in"):
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
        rotation = 0
        y_offset = 0
        
        # Apply animation effects
        if animation_type == "fade_in":
            # Simple fade in
            if progress < 0.3:  # First 30% - fade in
                alpha = int(255 * (progress / 0.3))
            
        elif animation_type == "slide_in":
            # Slide in from bottom
            if progress < 0.3:  # First 30% - slide in
                y_offset = int((1.0 - progress / 0.3) * self.output_size[1] * 0.5)
            
        elif animation_type == "zoom_in":
            # Zoom in
            if progress < 0.3:  # First 30% - zoom in
                scale = 0.5 + 0.5 * (progress / 0.3)
                alpha = int(255 * (progress / 0.3))
            
        elif animation_type == "typewriter":
            # Typewriter effect
            char_count = len(wrapped_text)
            visible_chars = int(char_count * min(1.0, progress * 1.5))
            wrapped_text = wrapped_text[:visible_chars]
            
        elif animation_type == "bounce":
            # Bouncing text
            y_offset = int(15 * math.sin(progress * math.pi * 3))
            
        elif animation_type == "pop":
            # Pop effect
            if progress < 0.2:
                scale = 0.8 + progress * 1.5  # Overshoot
            elif progress < 0.3:
                scale = 1.1 - (progress - 0.2) * 1  # Settle back
            
        elif animation_type == "spin":
            # Spinning entrance
            if progress < 0.3:
                rotation = 360 * (1 - progress / 0.3)
                alpha = int(255 * (progress / 0.3))
                
        elif animation_type == "wave":
            # Wavy text effect
            # We would need to handle each character individually for a true wave effect
            # This is a simplified version
            y_offset = int(10 * math.sin(progress * math.pi * 6))
            
        elif animation_type == "glitch_text":
            # Text glitch effect
            if random.random() < 0.2:
                # Occasionally shift the text position
                x += random.randint(-5, 5)
                y += random.randint(-5, 5)
                
            # Occasionally change the text color
            if random.random() < 0.1:
                r_shift = random.randint(-50, 50)
                g_shift = random.randint(-50, 50)
                b_shift = random.randint(-50, 50)
                
                r = max(0, min(255, self.subtitle_color[0] + r_shift))
                g = max(0, min(255, self.subtitle_color[1] + g_shift))
                b = max(0, min(255, self.subtitle_color[2] + b_shift))
                
                text_color = (r, g, b)
        
        # Apply fade out for all animations at the end
        if progress > 0.7:  # Last 30% - fade out
            alpha = int(255 * ((1.0 - progress) / 0.3))
            alpha = max(0, min(255, alpha))
        
        # Apply transformations
        if rotation != 0:
            # For rotation, we need to create a separate image, rotate it, then paste
            char_img = Image.new('RGBA', self.output_size, (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            
            # Draw text
            char_draw.text((x, y), wrapped_text, font=font, fill=(*text_color, alpha))
            
            # Rotate
            rotated = char_img.rotate(rotation, center=(x + text_width/2, y + text_height/2), resample=Image.BICUBIC, expand=False)
            text_img = rotated
        else:
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
            
            # Apply y-offset transformation
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
                     bg_effects=None, text_animations=None):
        """
        Create a video from text with animated subtitles and background effects.
        
        Args:
            text: Input text to convert to video
            keywords: Keywords for image search (defaults to extracting from text)
            segment_duration: Duration of each text segment in seconds
            output_filename: Name of the output video file
            bg_effects: List of background effects to use (one per segment)
            text_animations: List of text animation types to use (one per segment)
            
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
        
        # Available background effects and text animations
        available_bg_effects = [
            "zoom", "pan", "fade", "blur", "tiktok", "glitch", "none"
        ]
        
        available_text_animations = [
            "fade_in", "slide_in", "zoom_in", "typewriter", 
            "bounce", "pop", "spin", "wave", "glitch_text"
        ]
        
        # Assign effects and animations if not provided
        if not bg_effects or len(bg_effects) < len(segments):
            bg_effects = random.choices(available_bg_effects, k=len(segments))
            
        if not text_animations or len(text_animations) < len(segments):
            text_animations = random.choices(available_text_animations, k=len(segments))
        
        # Fetch background images
        image_urls = self.fetch_images(keywords, num_images=len(segments))
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' if MP4 doesn't work
        fps = 30
        output_path = os.path.abspath(output_filename)
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, self.output_size)
        
        # For each segment, create video frames with animations
        for i, (segment, image_url, bg_effect, text_animation) in enumerate(
                zip(segments, image_urls, bg_effects, text_animations)):
            print(f"Processing segment {i+1}/{len(segments)}")
            print(f"  Text: {segment[:30]}...")
            print(f"  Background effect: {bg_effect}")
            print(f"  Text animation: {text_animation}")
            
            # Download background image
            background = self.download_image(image_url)
            
            # Create frames for this segment
            num_frames = int(segment_duration * fps)
            for frame_idx in range(num_frames):
                # Apply background effect
                processed_bg = self.apply_image_effects(
                    background, 
                    effect_type=bg_effect,
                    frame_idx=frame_idx,
                    total_frames=num_frames
                )
                
                # Create text animation
                text_overlay = self.create_text_animation(
                    segment,
                    frame_idx=frame_idx,
                    total_frames=num_frames,
                    animation_type=text_animation
                )
                
                # Blend background and text
                final_frame = self.blend_images(processed_bg, text_overlay)
                
                # Add frame to video
                video_writer.write(final_frame)
                
                # Progress indicator (update every 10%)
                if frame_idx % (num_frames // 10) == 0 or frame_idx == num_frames - 1:
                    progress_pct = int((frame_idx + 1) / num_frames * 100)
                    print(f"  Progress: {progress_pct}%")
        
        # Release video writer
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
        return output_path

def display_menu():
    """Display interactive menu for the animated text-to-video generator."""
    print("\n" + "="*60)
    print(" ANIMATED TEXT TO VIDEO GENERATOR - TRENDING EFFECTS ".center(60, "="))
    print("="*60 + "\n")
    
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
    print("\nEnter output filename (default: animated_video.mp4):")
    output_filename = input().strip()
    if not output_filename:
        output_filename = "animated_video.mp4"
    elif not output_filename.endswith('.mp4'):
        output_filename += '.mp4'
    
    # Animation options
    print("\nChoose text animation style:")
    print("1. Random (recommended)")
    print("2. Fade in/out")
    print("3. Slide in")
    print("4. Zoom in")
    print("5. Typewriter")
    print("6. Bounce")
    print("7. Pop")
    print("8. Spin")
    print("9. Wave")
    print("10. Glitch (TikTok style)")
    
    animation_choice = input("Enter your choice (1-10, default: 1): ").strip()
    animation_map = {
        "2": ["fade_in"] * 100,
        "3": ["slide_in"] * 100,
        "4": ["zoom_in"] * 100,
        "5": ["typewriter"] * 100,
        "6": ["bounce"] * 100,
        "7": ["pop"] * 100,
        "8": ["spin"] * 100,
        "9": ["wave"] * 100,
        "10": ["glitch_text"] * 100,
    }
    text_animations = animation_map.get(animation_choice, None)  # None means random
    
    # Background effect options
    print("\nChoose background effect style:")
    print("1. Random (recommended)")
    print("2. Zoom")
    print("3. Pan")
    print("4. Fade")
    print("5. Blur")
    print("6. TikTok style")
    print("7. Glitch")
    print("8. None (static)")
    
    effect_choice = input("Enter your choice (1-8, default: 1): ").strip()
    effect_map = {
        "2": ["zoom"] * 100,
        "3": ["pan"] * 100,
        "4": ["fade"] * 100,
        "5": ["blur"] * 100,
        "6": ["tiktok"] * 100,
        "7": ["glitch"] * 100,
        "8": ["none"] * 100,
    }
    bg_effects = effect_map.get(effect_choice, None)  # None means random
    
    # Subtitle color
    print("\nChoose subtitle color:")
    print("1. White (default)")
    print("2. Yellow")
    print("3. Green")
    print("4. Red")
    print("5. Blue")
    print("6. Pink")
    
    color_choice = input("Enter your choice (1-6, default: 1): ").strip()
    color_map = {
        "1": (255, 255, 255),
        "2": (255, 255, 0),
        "3": (0, 255, 0),
        "4": (255, 0, 0),
        "5": (0, 0, 255),
        "6": (255, 0, 255),
    }
    subtitle_color = color_map.get(color_choice, (255, 255, 255))
    
    # Segment duration
    print("\nEnter segment duration in seconds (default: 5.0):")
    segment_duration_input = input().strip()
    try:
        segment_duration = float(segment_duration_input) if segment_duration_input else 5.0
    except ValueError:
        segment_duration = 5.0
        print("Invalid value, using default: 5.0 seconds")
    
    # Create generator with selected options
    generator = AnimatedTextToVideo(
        font_size=60,
        subtitle_color=subtitle_color
    )
    
    # Display processing message
    print("\nGenerating animated video...")
    print("This may take several minutes depending on the script length.")
    print("Downloading images and creating animations...\n")
    
    # Generate video
    output_path = generator.create_video(
        text=script_text,
        keywords=keywords,
        segment_duration=segment_duration,
        output_filename=output_filename,
        bg_effects=bg_effects,
        text_animations=text_animations
    )
    
    print(f"\nVideo generated successfully!")
    print(f"Saved as: {output_path}")
    
    return output_path

def main():
    """Main function to handle command line arguments or run interactive menu."""
    parser = argparse.ArgumentParser(description='Create animated text-to-video with trending effects')
    parser.add_argument('--text', type=str, help='Input text or path to text file')
    parser.add_argument('--keywords', type=str, help='Keywords for image search (comma separated)')
    parser.add_argument('--output', type=str, default='animated_video.mp4', help='Output filename')
    parser.add_argument('--segment-duration', type=float, default=5.0, 
                        help='Duration of each text segment in seconds')
    parser.add_argument('--animation', type=str, default='random', 
                        help='Text animation type (fade_in, slide_in, zoom_in, typewriter, bounce, pop, spin, wave, glitch_text, random)')
    parser.add_argument('--effect', type=str, default='random',
                        help='Background effect type (zoom, pan, fade, blur, tiktok, glitch, none, random)')
    parser.add_argument('--color', type=str, default='white',
                        help='Subtitle color (white, yellow, green, red, blue, pink)')
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
    
    # Parse color
    color_map = {
        'white': (255, 255, 255),
        'yellow': (255, 255, 0),
        'green': (0, 255, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'pink': (255, 0, 255)
    }
    subtitle_color = color_map.get(args.color.lower(), (255, 255, 255))
    
    # Parse animation
    text_animations = None
    if args.animation != 'random':
        text_animations = [args.animation]
    
    # Parse effect
    bg_effects = None
    if args.effect != 'random':
        bg_effects = [args.effect]
    
    # Create generator
    generator = AnimatedTextToVideo(
        font_size=60,
        subtitle_color=subtitle_color
    )
    
    # Generate video
    print(f"Generating animated video from text...")
    output_path = generator.create_video(
        text=text,
        keywords=keywords,
        segment_duration=args.segment_duration,
        output_filename=args.output,
        bg_effects=bg_effects,
        text_animations=text_animations
    )
    
    print(f"Video generated successfully: {output_path}")

if __name__ == "__main__":
    main()