# Global emoji cache to avoid repeated loading
_emoji_cache = {}

def create_color_to_emoji_mapping(quantized_colors):
    """Create a consistent mapping from colors to specific emojis with caching"""
    import os
    import glob
    import random
    
    base_path = "./noto-emoji/png/128"
    color_emoji_map = {}
    
    # Check if folder exists
    if not os.path.exists(base_path):
        print("Noto-emoji folder not found, using fallback patterns")
        return {}
    
    # Get all available emoji files (only once)
    emoji_files = glob.glob(os.path.join(base_path, "*.png"))
    if not emoji_files:
        print("No emoji files found")
        return {}
    
    emoji_filenames = [os.path.basename(f) for f in emoji_files]
    print(f"Found {len(emoji_filenames)} emoji files")
    
    # Predefined emoji mappings for consistent color representation
    color_emoji_preferences = {
        'red': ['emoji_u2764_fe0f.png', 'emoji_u1f525.png', 'emoji_u1f345.png', 'emoji_u1f534.png'],
        'green': ['emoji_u1f33f.png', 'emoji_u1f340.png', 'emoji_u1f333.png', 'emoji_u1f7e2.png'],
        'blue': ['emoji_u1f535.png', 'emoji_u1f30a.png', 'emoji_u1f4a7.png', 'emoji_u1f499.png'],
        'yellow': ['emoji_u1f31f.png', 'emoji_u2b50.png', 'emoji_u1f31e.png', 'emoji_u1f7e1.png'],
        'purple': ['emoji_u1f7e3.png', 'emoji_u1f347.png', 'emoji_u1f52e.png', 'emoji_u1f49c.png'],
        'black': ['emoji_u26ab.png', 'emoji_u1f5a4.png', 'emoji_u2b1b.png'],
        'white': ['emoji_u26aa.png', 'emoji_u1f90d.png', 'emoji_u2b1c.png'],
        'brown': ['emoji_u1f7e4.png', 'emoji_u1f90e.png']
    }

import numpy as np
import cv2
from PIL import Image, ImageDraw
import gradio as gr
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

def extract_metrics_for_report(image_input, n_colors, edge_sensitivity, tile_style, processing_times, grid_cells, palette_colors, mse_val, ssim_val):
    """
    Extract comprehensive metrics data for academic report analysis
    Returns structured data suitable for statistical analysis and visualization
    """
    
    # Analyze tile distribution
    tile_sizes = [cell['tile_size'] for cell in grid_cells]
    tile_size_counts = {}
    for size in tile_sizes:
        tile_size_counts[size] = tile_size_counts.get(size, 0) + 1
    
    # Calculate edge density statistics
    edge_densities = [cell['edge_density'] for cell in grid_cells]
    
    # Color usage analysis
    unique_colors_used = len(set(tuple(cell['color']) for cell in grid_cells))
    color_efficiency = unique_colors_used / n_colors
    
    # Compression analysis
    original_pixels = image_input.size[0] * image_input.size[1]
    processing_pixels = min(original_pixels, 600 * 600)
    compression_ratio = original_pixels / processing_pixels
    
    # Calculate edge threshold used
    edge_threshold = 0.2 - (edge_sensitivity - 1) * 0.03
    target_tiles = 150 + edge_sensitivity * 50
    
    # Compile comprehensive metrics dictionary
    metrics_data = {
        # Image properties
        'original_width': image_input.size[0],
        'original_height': image_input.size[1],
        'original_pixels': original_pixels,
        'compression_ratio': compression_ratio,
        
        # Algorithm parameters
        'n_colors': n_colors,
        'edge_sensitivity': edge_sensitivity,
        'edge_threshold': edge_threshold,
        'target_tiles': target_tiles,
        'tile_style': tile_style,
        
        # Performance timing
        'total_processing_time': processing_times['total'],
        'preprocessing_time': processing_times['preprocessing'],
        'grid_generation_time': processing_times['grid_generation'],
        'reconstruction_time': processing_times['reconstruction'],
        'preprocessing_percentage': (processing_times['preprocessing'] / processing_times['total']) * 100,
        'grid_generation_percentage': (processing_times['grid_generation'] / processing_times['total']) * 100,
        'reconstruction_percentage': (processing_times['reconstruction'] / processing_times['total']) * 100,
        
        # Tile analysis
        'total_tiles_generated': len(grid_cells),
        'tile_size_distribution': tile_size_counts,
        'avg_edge_density': np.mean(edge_densities),
        'max_edge_density': np.max(edge_densities),
        'min_edge_density': np.min(edge_densities),
        'std_edge_density': np.std(edge_densities),
        
        # Color analysis
        'colors_quantized': n_colors,
        'colors_actually_used': unique_colors_used,
        'color_efficiency': color_efficiency,
        
        # Similarity metrics
        'mse': mse_val,
        'ssim': ssim_val,
        'rmse': np.sqrt(mse_val),
        
        # Quality assessment
        'quality_grade': get_quality_grade(ssim_val),
        'processing_speed_pixels_per_second': original_pixels / processing_times['total']
    }
    
    return metrics_data

def get_quality_grade(ssim_val):
    """Convert SSIM to quality grade for analysis"""
    if ssim_val > 0.8:
        return "Excellent"
    elif ssim_val > 0.6:
        return "Good"
    elif ssim_val > 0.4:
        return "Fair"
    else:
        return "Poor"

def save_metrics_to_file(metrics_data, filename_prefix="metrics"):
    """Save metrics data to JSON file for analysis"""
    timestamp = int(time.time())
    filename = f"{filename_prefix}_{timestamp}.json"
    
    # Convert numpy types to Python native types for JSON serialization
    serializable_data = {}
    for key, value in metrics_data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_data[key] = value.item()
        else:
            serializable_data[key] = value
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        print(f"📊 Metrics saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return None

def print_metrics_summary(metrics_data):
    """Print a summary of metrics for console output"""
    print(f"\n=== PERFORMANCE METRICS SUMMARY ===")
    print(f"Image: {metrics_data['original_width']}x{metrics_data['original_height']} pixels")
    print(f"Processing time: {metrics_data['total_processing_time']:.2f}s")
    print(f"Processing speed: {metrics_data['processing_speed_pixels_per_second']:.0f} pixels/sec")
    print(f"SSIM: {metrics_data['ssim']:.3f} ({metrics_data['quality_grade']})")
    print(f"MSE: {metrics_data['mse']:.1f}")
    print(f"Tiles generated: {metrics_data['total_tiles_generated']}")
    print(f"Color efficiency: {metrics_data['color_efficiency']:.1%}")
    print(f"=== END SUMMARY ===\n")

def preprocess_image_with_quantization(image_pil, n_colors=16, max_size=400):
    """
    Step 1: Image preprocessing with color quantization - OPTIMIZED
    - Reduced to 400x400 max resolution for faster processing  
    - Use MiniBatchKMeans for 3-10x speed improvement over regular K-means
    - Apply color quantization to simplify color variations
    """
    print(f"🎨 Starting image preprocessing...")
    
    # Step 1: Resize to max 400px for even faster processing
    original_size = image_pil.size
    print(f"📏 Original size: {original_size}")
    
    # Always resize to max 400px for faster performance
    processing_image = image_pil
    scale_factor = 1.0
    
    if max(original_size) > max_size:
        scale_factor = max_size / max(original_size)
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        processing_image = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        print(f"📐 Resized to: {new_size} (limited to {max_size}px)")
    else:
        print(f"✅ Size within {max_size}px, no resize needed")
    
    print(f"🎯 Color quantization target: {n_colors} colors")
    
    # Convert to numpy array
    image_np = np.array(processing_image)
    original_shape = image_np.shape
    
    # Step 2: SUPER-FAST MiniBatchKMeans clustering with aggressive optimization
    pixels = image_np.reshape((-1, 3))
    total_pixels = len(pixels)
    print(f"📊 Total pixels: {total_pixels:,}")
    
    # Import MiniBatchKMeans for much faster clustering
    from sklearn.cluster import MiniBatchKMeans
    
    # More aggressive optimization for 400px max images
    if total_pixels > 15000:  # Lowered threshold further
        # Use even smaller batch size and sample more aggressively
        batch_size = min(800, total_pixels // 15)  # Smaller batch size
        sample_size = min(5000, total_pixels // 4)   # Much smaller sample size
        
        # Sample pixels for even faster processing
        sample_indices = np.random.choice(total_pixels, sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        print(f"⚡ Using MiniBatchKMeans: {sample_size:,} pixel sampling, batch={batch_size}")
        
        # MiniBatchKMeans with even more aggressive parameters
        kmeans = MiniBatchKMeans(
            n_clusters=n_colors, 
            batch_size=batch_size,
            max_iter=20,           # Further reduced iterations
            n_init=2,              # Even fewer initializations
            random_state=42,
            reassignment_ratio=0.005  # Less reassignment for more speed
        )
        kmeans.fit(sample_pixels)
        
    elif total_pixels > 3000:  # Medium images
        batch_size = min(300, total_pixels // 8)
        print(f"⚡ Using MiniBatchKMeans: full pixel processing, batch={batch_size}")
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            batch_size=batch_size, 
            max_iter=30,           # Reduced iterations
            n_init=2,              # Fewer initializations  
            random_state=42
        )
        kmeans.fit(pixels)
        
    else:  # Small images - use regular K-means (still fast for small data)
        print(f"🔍 Small image using standard K-means...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_colors, 
            random_state=42, 
            n_init=2,              # Reduced
            max_iter=50            # Reduced
        )
        kmeans.fit(pixels)
    
    # Get the quantized colors
    quantized_colors = kmeans.cluster_centers_.astype(np.uint8)
    print(f"🎨 Extracted dominant colors:")
    for i, color in enumerate(quantized_colors):
        print(f"   Color {i+1}: RGB{tuple(color)}")
    
    # Apply quantization to all pixels (this is fast with MiniBatch)
    print(f"🔄 Applying color quantization to all pixels...")
    labels = kmeans.predict(pixels)
    quantized_pixels = quantized_colors[labels]
    quantized_image_np = quantized_pixels.reshape(original_shape)
    
    # If we resized for processing, resize the quantized result back to original size
    if scale_factor < 1.0:
        quantized_temp = Image.fromarray(quantized_image_np)
        quantized_final = quantized_temp.resize(original_size, Image.Resampling.NEAREST)  # Use NEAREST to preserve quantized colors
        quantized_image_np = np.array(quantized_final)
        print(f"🔄 Restored to original size: {original_size}")
    
    quantized_image_pil = Image.fromarray(quantized_image_np)
    
    print(f"✅ MiniBatchKMeans color quantization complete!")
    
    return quantized_image_pil, quantized_image_np, quantized_colors

def create_adaptive_mosaic_with_edge_detection(quantized_image_np, quantized_colors, target_tile_count=200, edge_threshold=0.1):
    """
    Create mosaic with non-fixed-size grid using edge detection
    - Start with 32x32 and subdivide based on edge density
    - Complex areas get smaller tiles (8x8, 16x16)
    - Simple areas keep larger tiles (32x32)
    """
    h, w, _ = quantized_image_np.shape
    print(f"🧩 Starting adaptive mosaic creation, target tiles: {target_tile_count}")
    print(f"🔍 Edge detection threshold: {edge_threshold}")
    
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(quantized_image_np, cv2.COLOR_RGB2GRAY)
    
    def analyze_complexity_with_edges(x, y, size):
        """Analyze region complexity using edge detection"""
        x_end = min(x + size, w)
        y_end = min(y + size, h)
        
        # Extract cell region
        cell_gray = gray_image[y:y_end, x:x_end]
        cell_color = quantized_image_np[y:y_end, x:x_end]
        
        if cell_gray.size == 0:
            return False, (128, 128, 128), 0
        
        # Apply Canny edge detection
        if cell_gray.shape[0] > 3 and cell_gray.shape[1] > 3:
            edges = cv2.Canny(cell_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            edge_density = 0
        
        # Calculate representative color (center pixel method)
        center_y, center_x = cell_color.shape[0] // 2, cell_color.shape[1] // 2
        representative_color = tuple(cell_color[center_y, center_x])
        
        # Decide if subdivision is needed
        needs_subdivision = edge_density > edge_threshold
        
        return needs_subdivision, representative_color, edge_density
    
    def adaptive_subdivide(x, y, size, depth=0, max_depth=3):
        """
        Recursive subdivision based on edge detection
        - depth 0: 32x32 tiles
        - depth 1: 16x16 tiles  
        - depth 2: 8x8 tiles
        - depth 3: 4x4 tiles (minimum)
        """
        # Check if we can subdivide further
        if depth >= max_depth or size < 8:
            # Create final tile
            needs_sub, color, edge_density = analyze_complexity_with_edges(x, y, size)
            actual_w = min(size, w - x)
            actual_h = min(size, h - y)
            
            if actual_w > 0 and actual_h > 0:
                return [{
                    'x': x, 'y': y,
                    'width': actual_w, 'height': actual_h,
                    'color': color,
                    'edge_density': edge_density,
                    'tile_size': f"{actual_w}x{actual_h}"
                }]
            return []
        
        # Analyze current cell
        needs_subdivision, color, edge_density = analyze_complexity_with_edges(x, y, size)
        
        if not needs_subdivision:
            # Keep as large tile
            actual_w = min(size, w - x)
            actual_h = min(size, h - y)
            
            if actual_w > 0 and actual_h > 0:
                return [{
                    'x': x, 'y': y,
                    'width': actual_w, 'height': actual_h,
                    'color': color,
                    'edge_density': edge_density,
                    'tile_size': f"{actual_w}x{actual_h}"
                }]
            return []
        else:
            # Subdivide into 4 smaller tiles
            half_size = size // 2
            tiles = []
            
            # Top-left
            tiles.extend(adaptive_subdivide(x, y, half_size, depth + 1, max_depth))
            # Top-right  
            tiles.extend(adaptive_subdivide(x + half_size, y, half_size, depth + 1, max_depth))
            # Bottom-left
            tiles.extend(adaptive_subdivide(x, y + half_size, half_size, depth + 1, max_depth))
            # Bottom-right
            tiles.extend(adaptive_subdivide(x + half_size, y + half_size, half_size, depth + 1, max_depth))
            
            return tiles
    
    # Start with 32x32 base size and adaptively subdivide
    print(f"🔍 Starting 32x32 base grid edge detection analysis...")
    
    base_size = 32  # Start with 32x32 as specified
    all_tiles = []
    
    # Process image in 32x32 blocks
    for y in range(0, h, base_size):
        for x in range(0, w, base_size):
            tiles = adaptive_subdivide(x, y, base_size)
            all_tiles.extend(tiles)
    
    # Analyze tile size distribution
    size_counts = {}
    edge_densities = [tile['edge_density'] for tile in all_tiles]
    
    for tile in all_tiles:
        size_key = tile['tile_size']
        size_counts[size_key] = size_counts.get(size_key, 0) + 1
    
    print(f"✅ Generated {len(all_tiles)} adaptive tiles")
    print(f"📊 Tile size distribution:")
    for size, count in sorted(size_counts.items(), key=lambda x: int(x[0].split('x')[0]), reverse=True):
        percentage = (count / len(all_tiles)) * 100
        print(f"   {size}: {count} tiles ({percentage:.1f}%)")
    
    print(f"🔍 Edge density stats: avg {np.mean(edge_densities):.3f}, max {np.max(edge_densities):.3f}")
    
    return all_tiles

# Helper functions for emoji support
def create_color_to_emoji_mapping(quantized_colors):
    """Create a consistent mapping from colors to specific emojis"""
    import os
    import glob
    import random

    base_path = "./noto-emoji/png/128"
    color_emoji_map = {}
    
    # Check if folder exists
    if not os.path.exists(base_path):
        print("Noto-emoji folder not found, using fallback patterns")
        return {}
    
    # Get all available emoji files
    emoji_files = glob.glob(os.path.join(base_path, "*.png"))
    if not emoji_files:
        print("No emoji files found")
        return {}
    
    emoji_filenames = [os.path.basename(f) for f in emoji_files]
    print(f"Found {len(emoji_filenames)} emoji files")
    
    # Predefined emoji mappings for consistent color representation
    color_emoji_preferences = {
        'red': ['emoji_u2764_fe0f.png', 'emoji_u1f525.png', 'emoji_u1f345.png', 'emoji_u1f534.png'],
        'green': ['emoji_u1f33f.png', 'emoji_u1f340.png', 'emoji_u1f333.png', 'emoji_u1f7e2.png'],
        'blue': ['emoji_u1f535.png', 'emoji_u1f30a.png', 'emoji_u1f4a7.png', 'emoji_u1f499.png'],
        'yellow': ['emoji_u1f31f.png', 'emoji_u2b50.png', 'emoji_u1f31e.png', 'emoji_u1f7e1.png'],
        'purple': ['emoji_u1f7e3.png', 'emoji_u1f347.png', 'emoji_u1f52e.png', 'emoji_u1f49c.png'],
        'black': ['emoji_u26ab.png', 'emoji_u1f5a4.png', 'emoji_u2b1b.png'],
        'white': ['emoji_u26aa.png', 'emoji_u1f90d.png', 'emoji_u2b1c.png'],
        'brown': ['emoji_u1f7e4.png', 'emoji_u1f90e.png']
    }
    
    used_emojis = set()
    
    # For each quantized color, assign a specific emoji
    for i, color in enumerate(quantized_colors):
        r, g, b = color
        color_key = tuple(color)  # Use exact RGB as key
        
        # Determine color category
        if r > g and r > b and r > 150:
            category = 'red'
        elif g > r and g > b and g > 150:
            category = 'green'
        elif b > r and b > g and b > 150:
            category = 'blue'
        elif r > 200 and g > 200 and b < 100:
            category = 'yellow'
        elif r > 150 and b > 150 and g < 100:
            category = 'purple'
        elif r < 100 and g < 100 and b < 100:
            category = 'black'
        elif r > 200 and g > 200 and b > 200:
            category = 'white'
        else:
            category = 'brown'
        
        # Find an available emoji for this category
        assigned_emoji = None
        preferences = color_emoji_preferences.get(category, [])
        
        # Try preferred emojis first
        for pref_emoji in preferences:
            if pref_emoji in emoji_filenames and pref_emoji not in used_emojis:
                assigned_emoji = pref_emoji
                used_emojis.add(pref_emoji)
                break
        
        # If no preferred emoji available, pick any unused emoji from the category
        if not assigned_emoji:
            available_from_category = [e for e in preferences if e in emoji_filenames]
            if available_from_category:
                assigned_emoji = available_from_category[0]  # Take first available
            else:
                # Last resort: pick any available emoji
                available_emojis = [e for e in emoji_filenames if e not in used_emojis]
                if available_emojis:
                    assigned_emoji = random.choice(available_emojis)
                    used_emojis.add(assigned_emoji)
                else:
                    assigned_emoji = random.choice(emoji_filenames)  # Reuse if necessary
        
        color_emoji_map[color_key] = assigned_emoji
        print(f"Color {color} -> {assigned_emoji}")
    
    return color_emoji_map

def load_local_emoji(color, size, color_emoji_map):
    """Load specific emoji for a given color using consistent mapping"""
    import os
    
    color_key = tuple(color)
    
    # Get the assigned emoji for this color
    if color_key not in color_emoji_map:
        return None
    
    emoji_filename = color_emoji_map[color_key]
    base_path = "./noto-emoji/png/128"
    emoji_path = os.path.join(base_path, emoji_filename)
    
    try:
        if os.path.exists(emoji_path):
            emoji_img = Image.open(emoji_path)
            
            # Handle transparent background properly
            if emoji_img.mode in ('RGBA', 'LA'):
                # Create WHITE background for transparent emoji
                background = Image.new('RGB', emoji_img.size, (255, 255, 255))
                # Use the alpha channel as mask to paste emoji on white background
                background.paste(emoji_img, (0, 0), emoji_img)
                emoji_img = background
            elif emoji_img.mode == 'P':
                # Handle palette mode
                emoji_img = emoji_img.convert('RGBA')
                background = Image.new('RGB', emoji_img.size, (255, 255, 255))
                background.paste(emoji_img, (0, 0), emoji_img)
                emoji_img = background
            elif emoji_img.mode != 'RGB':
                emoji_img = emoji_img.convert('RGB')
            
            # Resize to target size
            emoji_img = emoji_img.resize(size, Image.Resampling.LANCZOS)
            return emoji_img
        else:
            return None
    except Exception as e:
        print(f"Error loading emoji {emoji_filename}: {e}")
        return None

def create_artistic_tile(color, size, style="3D Effect", color_emoji_map=None):
    """Create artistic tiles using quantized colors"""
    if size[0] <= 0 or size[1] <= 0:
        return Image.new('RGB', (max(1, size[0]), max(1, size[1])), color)
    
    w, h = size
    tile = Image.new('RGB', (w, h), color)
    
    if style == "3D Effect":
        # 3D effect with quantized colors
        pixels = tile.load()
        
        for i in range(w):
            for j in range(h):
                # Create subtle 3D effect
                edge_dist_x = min(i, w - 1 - i) / (w / 2) if w > 1 else 1
                edge_dist_y = min(j, h - 1 - j) / (h / 2) if h > 1 else 1
                edge_factor = min(edge_dist_x, edge_dist_y)
                edge_factor = max(0.85, min(1.0, edge_factor))
                
                new_color = tuple(int(c * edge_factor) for c in color)
                pixels[i, j] = new_color
        
        # Add highlights
        draw = ImageDraw.Draw(tile)
        if w > 2 and h > 2:
            highlight = tuple(min(255, int(c * 1.15)) for c in color)
            draw.line([(1, 1), (w-2, 1)], fill=highlight)
            draw.line([(1, 1), (1, h-2)], fill=highlight)
    
    elif style == "Emoji Tiles":
        from PIL import ImageFont
        emoji_char = map_color_to_emoji(color)
        tile = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(tile)
        try:
            font_size = int(min(w, h) * 0.8)
            font = ImageFont.truetype("AppleColorEmoji.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("Segoe UI Emoji.ttf", font_size)
            except:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), emoji_char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos = ((w - tw) // 2, (h - th) // 2)
    
        draw.text(pos, emoji_char, font=font, fill=(0, 0, 0))
            
    elif style == "Flat Color":
        # Pure flat quantized color
        pass
    
    return tile

def map_color_to_emoji(color):
    """Map RGB color to appropriate emoji"""
    r, g, b = color
    
    # Simple color mapping to emoji
    if r > g and r > b and r > 150:  # Red dominant
        emojis = ["🔴", "❤️", "🟥", "🍎", "🌹"]
    elif g > r and g > b and g > 150:  # Green dominant  
        emojis = ["🟢", "💚", "🟩", "🍀", "🥒"]
    elif b > r and b > g and b > 150:  # Blue dominant
        emojis = ["🔵", "💙", "🟦", "🫐", "🌀"]
    elif r > 200 and g > 200 and b < 100:  # Yellow
        emojis = ["🟡", "💛", "🟨", "⭐", "🌟"]
    elif r > 150 and b > 150 and g < 100:  # Purple/Magenta
        emojis = ["🟣", "💜", "🟪", "🍇", "🔮"]
    elif r > 200 and g > 150 and b < 100:  # Orange
        emojis = ["🟠", "🧡", "🟧", "🍊", "🎃"]
    elif r < 100 and g < 100 and b < 100:  # Dark/Black
        emojis = ["⚫", "🖤", "⬛", "🕳️", "🌑"]
    elif r > 200 and g > 200 and b > 200:  # Light/White
        emojis = ["⚪", "🤍", "⬜", "☁️", "🥛"]
    else:  # Gray or mixed
        emojis = ["⚫", "⚪", "🔘", "⭕", "🔵"]
    
    # Return a random emoji from the appropriate category
    import random
    return random.choice(emojis)

def reconstruct_quantized_mosaic(grid_cells, target_size, tile_style="3D Effect", color_emoji_map=None):
    """Reconstruct mosaic using NumPy operations for massive speed improvement"""
    w, h = target_size
    print(f"🖼️ Reconstructing quantized mosaic, size: {w}x{h}")
    
    # Create numpy array for fast batch operations
    mosaic_array = np.full((h, w, 3), [255, 255, 255], dtype=np.uint8)  # White background
    
    if tile_style == "Emoji Tiles" and color_emoji_map:
        # For emoji tiles, we still need PIL operations but with caching
        mosaic = Image.new('RGB', (w, h), (255, 255, 255))
        
        # Pre-load and cache all needed emoji images
        emoji_cache = {}
        unique_colors = set(tuple(cell['color']) for cell in grid_cells)
        
        print(f"📦 Pre-loading {len(unique_colors)} color emojis...")
        for color_tuple in unique_colors:
            if color_tuple in color_emoji_map:
                emoji_filename = color_emoji_map[color_tuple]
                emoji_cache[color_tuple] = load_emoji_to_cache(emoji_filename)
        
        # Fast paste with pre-loaded emojis
        for tile_info in grid_cells:
            x, y = tile_info['x'], tile_info['y']
            width, height = tile_info['width'], tile_info['height']
            color_tuple = tuple(tile_info['color'])
            
            if x >= w or y >= h:
                continue
            
            actual_width = min(width, w - x)
            actual_height = min(height, h - y)
            
            if actual_width <= 0 or actual_height <= 0:
                continue
            
            if color_tuple in emoji_cache and emoji_cache[color_tuple]:
                # Resize cached emoji and paste
                emoji_img = emoji_cache[color_tuple].resize((actual_width, actual_height), Image.Resampling.LANCZOS)
                mosaic.paste(emoji_img, (x, y))
            else:
                # Fallback to numpy operations for missing emojis
                mosaic_array[y:y+actual_height, x:x+actual_width] = tile_info['color']
        
        # Convert final result if needed
        if np.any(mosaic_array != 255):  # If we used numpy fallback
            numpy_result = Image.fromarray(mosaic_array)
            # Composite with emoji result
            mosaic = Image.alpha_composite(mosaic.convert('RGBA'), numpy_result.convert('RGBA')).convert('RGB')
        
        print("✅ Emoji mosaic reconstruction complete")
        return mosaic
    
    else:
        # For non-emoji styles, use pure NumPy operations for maximum speed
        print(f"⚡ Using NumPy batch operations for {len(grid_cells)} tiles...")
        
        if tile_style == "Flat Color":
            # Fastest: direct color filling
            for tile_info in grid_cells:
                x, y = tile_info['x'], tile_info['y']
                width, height = tile_info['width'], tile_info['height']
                color = tile_info['color']
                
                x_end = min(x + width, w)
                y_end = min(y + height, h)
                
                if x < w and y < h and x_end > x and y_end > y:
                    mosaic_array[y:y_end, x:x_end] = color
        
        elif tile_style == "3D Effect":
            # Optimized 3D effect using vectorized operations
            for tile_info in grid_cells:
                x, y = tile_info['x'], tile_info['y']
                width, height = tile_info['width'], tile_info['height']
                color = np.array(tile_info['color'])
                
                x_end = min(x + width, w)
                y_end = min(y + height, h)
                
                if x < w and y < h and x_end > x and y_end > y:
                    tile_w, tile_h = x_end - x, y_end - y
                    
                    if tile_w > 0 and tile_h > 0:
                        # Create 3D effect using numpy broadcasting
                        i_coords = np.arange(tile_w)
                        j_coords = np.arange(tile_h)
                        
                        # Vectorized edge distance calculation
                        edge_dist_x = np.minimum(i_coords, tile_w - 1 - i_coords) / (tile_w / 2) if tile_w > 1 else np.ones(tile_w)
                        edge_dist_y = np.minimum(j_coords, tile_h - 1 - j_coords) / (tile_h / 2) if tile_h > 1 else np.ones(tile_h)
                        
                        # Create 2D edge factor matrix
                        edge_factor_2d = np.minimum(edge_dist_y[:, np.newaxis], edge_dist_x[np.newaxis, :])
                        edge_factor_2d = np.clip(edge_factor_2d, 0.85, 1.0)
                        
                        # Apply 3D effect
                        tile_colors = color[np.newaxis, np.newaxis, :] * edge_factor_2d[:, :, np.newaxis]
                        tile_colors = np.clip(tile_colors, 0, 255).astype(np.uint8)
                        
                        # Add highlights (vectorized)
                        if tile_w > 2 and tile_h > 2:
                            highlight = np.clip(color * 1.15, 0, 255).astype(np.uint8)
                            tile_colors[0, :] = highlight  # Top edge
                            tile_colors[:, 0] = highlight  # Left edge
                        
                        # Assign to mosaic
                        mosaic_array[y:y_end, x:x_end] = tile_colors
        
        # Convert numpy array to PIL Image once at the end
        mosaic = Image.fromarray(mosaic_array)
        print("✅ NumPy batch mosaic reconstruction complete")
        return mosaic

def load_emoji_to_cache(emoji_filename):
    """Load emoji to cache at original size"""
    import os
    global _emoji_cache
    
    if emoji_filename in _emoji_cache:
        return _emoji_cache[emoji_filename]

    base_path = "./noto-emoji/png/128"
    emoji_path = os.path.join(base_path, emoji_filename)
    
    try:
        if os.path.exists(emoji_path):
            emoji_img = Image.open(emoji_path)
            
            # Handle transparent background properly
            if emoji_img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', emoji_img.size, (255, 255, 255))
                background.paste(emoji_img, (0, 0), emoji_img)
                emoji_img = background
            elif emoji_img.mode == 'P':
                emoji_img = emoji_img.convert('RGBA')
                background = Image.new('RGB', emoji_img.size, (255, 255, 255))
                background.paste(emoji_img, (0, 0), emoji_img)
                emoji_img = background
            elif emoji_img.mode != 'RGB':
                emoji_img = emoji_img.convert('RGB')
            
            _emoji_cache[emoji_filename] = emoji_img
            return emoji_img
        else:
            return None
    except Exception as e:
        print(f"Error caching emoji {emoji_filename}: {e}")
        return None

def compute_metrics(original_pil, mosaic_pil):
    """Compute similarity metrics"""
    original_np = np.array(original_pil)
    mosaic_np = np.array(mosaic_pil)
    
    # Ensure same size
    if original_np.shape != mosaic_np.shape:
        mosaic_pil_resized = mosaic_pil.resize(original_pil.size, Image.Resampling.LANCZOS)
        mosaic_np = np.array(mosaic_pil_resized)
    
    # Convert to grayscale
    original_gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    mosaic_gray = cv2.cvtColor(mosaic_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate metrics
    mse_val = np.mean((original_gray - mosaic_gray) ** 2)
    ssim_val = ssim(original_gray, mosaic_gray, data_range=255)
    
    return mse_val, ssim_val

def quantized_mosaic_pipeline(image_input, n_colors, edge_sensitivity, tile_style):
    """Main pipeline with adaptive non-fixed-size grid using edge detection"""
    try:
        if image_input is None:
            return None, None, "⚠ Please upload an image"
        
        start_time = time.time()
        print(f"🚀 Starting adaptive mosaic generation...")
        print(f"Parameters: colors={n_colors}, edge sensitivity={edge_sensitivity}, style={tile_style}")
        
        # Step 1: Preprocess with quantization (unified 600px processing)
        original_size = image_input.size
        print(f"📏 Original size: {original_size}")
        
        max_processing_size = 600
        
        step1_start = time.time()
        quantized_pil, quantized_np, palette_colors = preprocess_image_with_quantization(
            image_input.convert("RGB"), n_colors, max_processing_size
        )
        step1_time = time.time() - step1_start
        print(f"⏱️ Color quantization time: {step1_time:.2f}s")
        
        # Step 2: Create adaptive mosaic using edge detection
        step2_start = time.time()
        
        # Convert edge_sensitivity to threshold (higher sensitivity = lower threshold)
        edge_threshold = 0.2 - (edge_sensitivity - 1) * 0.03  # Range: 0.05 to 0.17
        target_tiles = 150 + edge_sensitivity * 50  # More sensitive = more tiles
        
        grid_cells = create_adaptive_mosaic_with_edge_detection(
            quantized_np, palette_colors, target_tiles, edge_threshold
        )
        step2_time = time.time() - step2_start
        print(f"⏱️ Adaptive grid generation time: {step2_time:.2f}s")
        
        if not grid_cells:
            return None, None, "⚠ Failed to generate tiles"
        
        # Step 3: Reconstruct mosaic with original size using fast NumPy operations
        step3_start = time.time()
        
        # Create consistent color-to-emoji mapping for Emoji tiles
        color_emoji_map = None
        if tile_style == "Emoji Tiles":
            print("🎭 Creating consistent color-to-emoji mapping...")
            color_emoji_map = create_color_to_emoji_mapping(palette_colors)
            print(f"✅ Created {len(color_emoji_map)} color-emoji mappings")
        
        mosaic_pil = reconstruct_quantized_mosaic(grid_cells, original_size, tile_style, color_emoji_map)
        step3_time = time.time() - step3_start
        print(f"⏱️ Fast mosaic reconstruction time: {step3_time:.2f}s")
        
        # Step 4: Compute metrics
        mse_val, ssim_val = compute_metrics(quantized_pil, mosaic_pil)
        
        # Create color palette visualization
        palette_img = create_color_palette_visualization(palette_colors)
        
        # Analyze tile statistics
        unique_colors_used = len(set(cell['color'] for cell in grid_cells))
        size_distribution = {}
        total_area = sum(cell['width'] * cell['height'] for cell in grid_cells)
        avg_tile_size = total_area / len(grid_cells)
        
        # Count tile sizes
        for cell in grid_cells:
            size_key = cell['tile_size']
            size_distribution[size_key] = size_distribution.get(size_key, 0) + 1
        
        # Find most common sizes
        sorted_sizes = sorted(size_distribution.items(), key=lambda x: x[1], reverse=True)
        top_sizes = sorted_sizes[:3]
        
        total_time = time.time() - start_time
        
        # Calculate compression info
        original_pixels = original_size[0] * original_size[1]
        processing_pixels = min(original_pixels, 600 * 600)
        compression_ratio = original_pixels / processing_pixels if processing_pixels < original_pixels else 1.0
        
        # Prepare processing times for metrics extraction
        processing_times = {
            'total': total_time,
            'preprocessing': step1_time,
            'grid_generation': step2_time,
            'reconstruction': step3_time
        }
        
        # Extract comprehensive metrics for report analysis
        metrics_data = extract_metrics_for_report(
            image_input, n_colors, edge_sensitivity, tile_style,
            processing_times, grid_cells, palette_colors, mse_val, ssim_val
        )
        
        # Save metrics to file for analysis
        save_metrics_to_file(metrics_data)
        
        # Print summary to console
        print_metrics_summary(metrics_data)
        
        metrics_text = f"""
## 🧩 Adaptive Mosaic Report (Non-Fixed Grid)

### ⚡ Performance Statistics
- **Total Processing Time**: {total_time:.2f} seconds
- **Color Quantization**: {step1_time:.2f}s | **Adaptive Grid**: {step2_time:.2f}s | **Image Reconstruction**: {step3_time:.2f}s

### 📊 Tile Size Distribution ({len(grid_cells)} total)
- **{top_sizes[0][0]}**: {top_sizes[0][1]} tiles ({top_sizes[0][1]/len(grid_cells)*100:.1f}%)
- **{top_sizes[1][0]}**: {top_sizes[1][1]} tiles ({top_sizes[1][1]/len(grid_cells)*100:.1f}%)
- **{top_sizes[2][0]}**: {top_sizes[2][1]} tiles ({top_sizes[2][1]/len(grid_cells)*100:.1f}%)
- **Average Tile Size**: {avg_tile_size:.1f} pixels²

### 🎨 Color Quantization Info  
- **Final Output Size**: {original_size[0]} × {original_size[1]} pixels
- **Processing Optimization**: {f"Compression ratio {compression_ratio:.1f}x" if compression_ratio > 1 else "No compression needed"}
- **Quantized Colors**: {n_colors} colors
- **Actually Used Colors**: {unique_colors_used} colors

### 📈 Quality Assessment
- **Mean Squared Error (MSE)**: {mse_val:.2f}
- **Structural Similarity (SSIM)**: {ssim_val:.4f}
- **Quality Grade**: {("🟢 Excellent" if ssim_val > 0.8 else "🟡 Good" if ssim_val > 0.6 else "🟠 Fair" if ssim_val > 0.4 else "🔴 Needs Optimization")}

### 💡 Optimization Tips
{get_adaptive_optimization_tips(ssim_val, len(grid_cells), edge_sensitivity, size_distribution)}


        """
        
        print(f"✅ Adaptive mosaic generation complete! Total time: {total_time:.2f}s")
        return mosaic_pil, palette_img, metrics_text
        
    except Exception as e:
        import traceback
        error_msg = f"⚠ Processing error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return None, None, error_msg

def create_color_palette_visualization(palette_colors):
    """Create a visualization of the extracted color palette"""
    n_colors = len(palette_colors)
    
    # Create palette image
    palette_width = min(400, n_colors * 25)
    palette_height = 60
    palette_img = Image.new('RGB', (palette_width, palette_height), (255, 255, 255))
    
    color_width = palette_width // n_colors
    
    for i, color in enumerate(palette_colors):
        x_start = i * color_width
        x_end = min((i + 1) * color_width, palette_width)
        
        # Fill color rectangle
        for x in range(x_start, x_end):
            for y in range(palette_height):
                palette_img.putpixel((x, y), tuple(color))
    
    # Add border
    draw = ImageDraw.Draw(palette_img)
    draw.rectangle([0, 0, palette_width-1, palette_height-1], outline=(0, 0, 0), width=2)
    
    # Add color separators
    for i in range(1, n_colors):
        x = i * color_width
        draw.line([(x, 0), (x, palette_height)], fill=(255, 255, 255), width=1)
    
    return palette_img

def get_adaptive_optimization_tips(ssim_val, tile_count, edge_sensitivity, size_distribution):
    """Provide optimization suggestions for adaptive mosaic"""
    large_tiles = size_distribution.get('32x32', 0) + size_distribution.get('16x16', 0)
    small_tiles = size_distribution.get('8x8', 0) + size_distribution.get('4x4', 0)
    
    if ssim_val > 0.8:
        return "✨ Excellent quality! The adaptive grid effect is working well."
    elif large_tiles > tile_count * 0.8:
        return "📝 The image is relatively simple. You can increase edge sensitivity for more detailed tiles."
    elif small_tiles > tile_count * 0.6:
        return "🧩 Very rich in details! If you prefer simpler results, you can reduce edge sensitivity."
    elif ssim_val < 0.5:
        return "⚡ Try adjusting edge sensitivity or choosing a different tile style."
    else:
        return "👍 Good adaptive effect! Tile sizes are well distributed across different regions."

def create_interface():
    
    
    with gr.Blocks(title="🎨 Interactive Image Mosaic Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Interactive Image Mosaic Generator
        
        **Transform your photos into stunning emoji mosaic art with intelligent adaptive tiling!**
        
        🎯 **Key Features:**
        - **🚀 Smart Color Quantization** - MiniBatchKMeans algorithm, 3-10x faster than K-means
        - **🔍 Edge-Driven Adaptive Tiling** - Uses Canny edge detection for intelligent tile sizing
        - **🧩 Dynamic Grid System** - Complex areas get small tiles (8×8, 16×16), simple areas keep large tiles (32×32)
        - **🎭 Emoji Tile Style** - Replace traditional color blocks with colorful emojis for more visual interest
        - **⚡ Lightning Fast** - Typically completes in 2-4 seconds
        - **📐 Preserves Original Size** - Mosaic maintains exact same dimensions as input image
        
        Upload an image and watch as it transforms into a beautiful mosaic with intelligent tile placement based on image complexity!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 Upload Image",
                    type="pil",
                    height=300
                )
                
                n_colors = gr.Slider(
                    minimum=4,
                    maximum=32,
                    value=12,
                    step=2,
                    label="🎨 Number of Colors",
                    info="How many colors to use in quantization"
                )
                
                edge_sensitivity = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="🔍 Edge Sensitivity",
                    info="Higher sensitivity = smaller tiles in detailed areas"
                )
                
                tile_style = gr.Dropdown(
                    choices=["Emoji Tiles", "3D Effect", "Flat Color"],
                    value="Emoji Tiles",
                    label="🎭 Tile Style",
                    info="Choose the visual style for mosaic tiles"
                )
                
                generate_btn = gr.Button("🚀 Generate Mosaic", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                mosaic_output = gr.Image(
                    label="🖼️ Generated Mosaic Art",
                    height=400
                )
                
                palette_output = gr.Image(
                    label="🎨 Extracted Color Palette",
                    height=80
                )
                
                metrics_output = gr.Markdown(
                    "### 🎯 Ready to create amazing mosaic art!\n\nUpload an image and adjust the parameters above, then click **Generate Mosaic** to start the transformation process."
                )
        
        gr.Markdown("""
        ### 💡 How It Works
        
        1. **Color Quantization** - Advanced clustering reduces your image to the specified number of dominant colors
        2. **Edge Detection** - Canny algorithm analyzes image complexity to determine optimal tile placement  
        3. **Adaptive Tiling** - Intelligent grid system uses small tiles for complex areas, large tiles for simple regions
        4. **Artistic Reconstruction** - Rebuilds the image using your chosen tile style with perfect size preservation
        
        ### 🎨 Style Guide
        
        - **Emoji Tiles**: Colorful emoji characters replace traditional mosaic tiles for a playful, modern look
        - **3D Effect**: Classic raised tile appearance with subtle shadows and highlights
        - **Flat Color**: Clean, minimal geometric tiles in pure quantized colors
        
        ### ⚡ Performance Notes
        
        The app automatically optimizes processing based on your image size while maintaining quality. Larger images are intelligently processed for speed without sacrificing the final output resolution.
        """)
        
        # Event handlers
        generate_btn.click(
            fn=quantized_mosaic_pipeline,
            inputs=[image_input, n_colors, edge_sensitivity, tile_style],
            outputs=[mosaic_output, palette_output, metrics_output]
        )
        
        # Auto-update on parameter change
        for component in [n_colors, edge_sensitivity, tile_style]:
            component.change(
                fn=quantized_mosaic_pipeline,
                inputs=[image_input, n_colors, edge_sensitivity, tile_style],
                outputs=[mosaic_output, palette_output, metrics_output],
                show_progress=True
            )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Interactive Image Mosaic Generator...")
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)