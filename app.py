from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def greedy_pixel_assignment(source, target):
    """Fast greedy pixel assignment"""
    h, w, c = source.shape
    n_pixels = h * w
    
    # Flatten
    source_pixels = source.reshape(n_pixels, 3).astype(np.float32)
    target_pixels = target.reshape(n_pixels, 3).astype(np.float32)
    
    # Track assignments
    used = np.zeros(n_pixels, dtype=bool)
    mapping = np.zeros(n_pixels, dtype=np.int32)
    
    # Sort target pixels by importance
    target_importance = np.sum(target_pixels, axis=1)
    sorted_indices = np.argsort(target_importance)
    
    for idx, target_idx in enumerate(sorted_indices):
        target_color = target_pixels[target_idx]
        
        # Find closest available source pixel
        available_mask = ~used
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            distances = np.sum((source_pixels - target_color) ** 2, axis=1)
            best_source_idx = np.argmin(distances)
        else:
            available_pixels = source_pixels[available_indices]
            distances = np.sum((available_pixels - target_color) ** 2, axis=1)
            best_idx_in_available = np.argmin(distances)
            best_source_idx = available_indices[best_idx_in_available]
            used[best_source_idx] = True
        
        mapping[target_idx] = best_source_idx
    
    return mapping

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('html_for_web_version.html')

@app.route('/static/seal.jpg')
def serve_seal():
    """Serve the seal image"""
    return send_file('seal.jpg', mimetype='image/jpeg')

@app.route('/api/transform', methods=['POST'])
def transform():
    """Calculate pixel mapping and return data for animation"""
    try:
        print("=== Transform Request Received ===")
        
        # Check if files are present
        if 'source' not in request.files:
            print("ERROR: No source image in request")
            return jsonify({'error': 'No source image provided', 'success': False}), 400
        
        source_file = request.files['source']
        print(f"Source file: {source_file.filename}")
        
        if source_file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No source file selected', 'success': False}), 400
        
        if not allowed_file(source_file.filename):
            print(f"ERROR: Invalid file type: {source_file.filename}")
            return jsonify({'error': 'Invalid file type. Use JPG, PNG, or BMP', 'success': False}), 400
        
        print("Loading source image...")
        # Load source image
        source_image = Image.open(source_file.stream).convert('RGB')
        print(f"Source image loaded: {source_image.size}")
        
        # Load target seal image
        seal_path = 'seal.jpg'
        if not os.path.exists(seal_path):
            print(f"ERROR: seal.jpg not found at {os.path.abspath(seal_path)}")
            return jsonify({'error': 'seal.jpg not found on server. Please ensure seal.jpg is in the app directory.', 'success': False}), 500
        
        print("Loading seal.jpg...")
        target_image = Image.open(seal_path).convert('RGB')
        print(f"Seal image loaded: {target_image.size}")
        
        # Resize to 128x128
        resolution = 128
        print(f"Resizing images to {resolution}x{resolution}...")
        source = np.array(source_image.resize((resolution, resolution), Image.Resampling.LANCZOS))
        target = np.array(target_image.resize((resolution, resolution), Image.Resampling.LANCZOS))
        
        # Calculate pixel mapping
        print("Calculating pixel mapping...")
        mapping = greedy_pixel_assignment(source, target)
        print("Mapping complete!")
        
        # Prepare response data
        h, w, c = source.shape
        
        # Source positions (original layout)
        source_positions = [[y, x] for y in range(h) for x in range(w)]
        
        # Target positions (where pixels should go)
        target_positions = []
        for target_idx in range(h * w):
            ty, tx = divmod(target_idx, w)
            target_positions.append([ty, tx])
        
        # Source pixels (colors)
        source_pixels = source.reshape(-1, 3).tolist()
        
        # Convert source image to base64
        source_img_b64 = image_to_base64(Image.fromarray(source))
        
        print("Returning response data...")
        return jsonify({
            'success': True,
            'width': w,
            'height': h,
            'source_positions': source_positions,
            'target_positions': target_positions,
            'mapping': mapping.tolist(),
            'source_pixels': source_pixels,
            'source_image': source_img_b64
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("=== ERROR ===")
        print(error_trace)
        return jsonify({'error': str(e), 'success': False, 'trace': error_trace}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Print startup info
    print("=" * 60)
    print("🦭 Sealify Web Server Starting...")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Templates folder: {os.path.join(os.getcwd(), 'templates')}")
    print(f"Seal image: {os.path.join(os.getcwd(), 'seal.jpg')}")
    
    # Check if seal.jpg exists
    if os.path.exists('seal.jpg'):
        print("✓ seal.jpg found!")
    else:
        print("✗ WARNING: seal.jpg NOT FOUND!")
        print("  Please add seal.jpg to the current directory")
    
    # Check if templates folder exists
    if os.path.exists('templates/html_for_web_version.html'):
        print("✓ templates/index.html found!")
    else:
        print("✗ WARNING: templates/index.html NOT FOUND!")
        print("  Please create templates/ folder and add index.html")
    
    print("=" * 60)
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server running on: http://localhost:{port}")
    print("   Press CTRL+C to stop")
    print("=" * 60)
    print()
    
    app.run(host='0.0.0.0', port=port, debug=True)