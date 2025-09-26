# Interactive Image Mosaic Generator

Transform any image into stunning mosaic art using intelligent color quantization and adaptive tile sizing!

## Features

- **Smart Color Quantization**: Extracts dominant colors from your image using advanced MiniBatchKMeans clustering
- **Adaptive Grid System**: Automatically uses small tiles for detailed areas and large tiles for uniform regions
- **Multiple Styles**: Choose from 3D Effect, Emoji Tiles, or Flat Color rendering
- **Real-Time Processing**: Processes high-resolution images in under 7 seconds
- **Original Size Preservation**: Output maintains exact input dimensions

## How to Use

1. **Upload Your Image**: Drag and drop or click to upload any image
   or download a sample image：
   ![greece](https://github.com/user-attachments/assets/c4adf8cd-9bd6-4f8a-8a12-e518f31d53b0)


3. **Adjust Parameters**:
   - **Color Count**: 4-32 colors (recommended: 8-16)
   - **Edge Sensitivity**: 1-5 (lower = larger tiles, higher = more detail)
   - **Tile Style**: Choose your preferred visual effect
4. **Generate**: Click the button or adjust parameters for auto-generation

## Recommended Settings

| Image Type | Colors | Edge Sensitivity | Style |
|------------|--------|------------------|-------|
| Portraits | 12-16 | 2-3 | Emoji Tiles |
| Landscapes | 8-12 | 1-2 | 3D Effect |
| Architecture | 10-14 | 1 | Flat Color |
| Simple Graphics | 6-10 | 2-3 | Any |

## Technical Details

- **Algorithm**: Edge detection-driven adaptive subdivision
- **Base Grid**: 32×32 tiles with recursive subdivision to 4×4
- **Performance**: Up to 3M+ pixels/second processing speed
- **Quality Metrics**: SSIM and MSE similarity measurements included

## Tile Styles

- **Emoji Tiles**: Uses real emoji images for creative mosaic effects
- **3D Effect**: Adds depth with lighting and shadows
- **Flat Color**: Clean, minimalist color blocks

## Tips for Best Results

- For complex images (buildings, detailed scenes): Use edge sensitivity = 1
- For simple images (portraits, landscapes): Use edge sensitivity = 2-3
- More colors = more detail but slower processing
- Emoji style works best with 8-16 colors

## Performance

Typical processing times for high-resolution images:
- Small images (<1MP): 1-2 seconds
- Medium images (1-5MP): 2-4 seconds  
- Large images (5-10MP): 3-7 seconds

---

*Built with Python, OpenCV, scikit-learn, and Gradio*
