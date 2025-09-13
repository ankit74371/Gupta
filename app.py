import os
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend access

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
model = None
device = None
transform = None
class_names = ['roads', 'streetlights', 'garbage', 'not_civic']

def create_model_from_architecture(architecture_name, num_classes):
    """Create model based on detected architecture"""
    architecture_name = architecture_name.lower()
    
    if 'mobilenet' in architecture_name:
        # MobileNet architecture
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif 'resnet18' in architecture_name:
        # ResNet18 architecture  
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif 'resnet50' in architecture_name:
        # ResNet50 architecture
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif 'vgg' in architecture_name:
        # VGG architecture
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    else:
        # Default to ResNet18
        logger.warning(f"Unknown architecture '{architecture_name}', defaulting to ResNet18")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

def detect_model_architecture(state_dict):
    """Detect model architecture from state_dict keys"""
    keys = list(state_dict.keys())
    
    # Check for MobileNet
    if any('mobilenet' in key for key in keys):
        return 'mobilenet'
    
    # Check for ResNet architectures
    if any('layer4' in key for key in keys):
        if any('layer4.2' in key for key in keys):
            return 'resnet50'  # ResNet50 has 3 blocks in layer4
        else:
            return 'resnet18'  # ResNet18 has 2 blocks in layer4
    
    # Check for VGG
    if any('features.30' in key for key in keys):
        return 'vgg'
    
    # Check for simple CNN patterns
    if any('conv' in key.lower() for key in keys) and len(keys) < 20:
        return 'simple_cnn'
    
    # Default fallback
    return 'resnet18'

class SimpleCNN(nn.Module):
    """Simple CNN architecture for custom models"""
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Global average pooling and classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def load_model():
    """Load the PyTorch model with comprehensive error handling"""
    global model, device, class_names
    
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if model file exists
        model_path = 'civic_issue_classifier.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it's in the same directory as app.py")
        
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        logger.info("Checkpoint loaded successfully")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Extract information from checkpoint
            if 'class_names' in checkpoint:
                class_names = checkpoint['class_names']
                logger.info(f"Loaded class names: {class_names}")
            
            num_classes = checkpoint.get('num_classes', len(class_names))
            logger.info(f"Number of classes: {num_classes}")
            
            # Get the actual model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Checkpoint might be the state dict itself
                state_dict = checkpoint
            
            # Handle DataParallel models (strip 'module.' prefix)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Detect model architecture from state dict
            architecture = detect_model_architecture(new_state_dict)
            logger.info(f"Detected architecture: {architecture}")
            
            # Create model based on detected architecture
            if architecture == 'simple_cnn':
                model = SimpleCNN(num_classes=num_classes)
            else:
                model = create_model_from_architecture(architecture, num_classes)
            
            # Load state dict with error handling
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                    logger.warning(f"Strict loading failed, trying with strict=False: {str(e)}")
                    # Try loading with strict=False
                    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                    
                    # If still too many missing keys, try different architectures
                    if len(missing_keys) > len(list(model.state_dict().keys())) * 0.5:
                        logger.info("Too many missing keys, trying different architectures...")
                        
                        # Try MobileNet if not already tried
                        if architecture != 'mobilenet':
                            try:
                                logger.info("Trying MobileNet architecture...")
                                model = models.mobilenet_v2(weights=None)
                                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                                model.load_state_dict(new_state_dict, strict=False)
                            except:
                                # Try SimpleCNN as last resort
                                logger.info("Trying SimpleCNN architecture...")
                                model = SimpleCNN(num_classes=num_classes)
                                model.load_state_dict(new_state_dict, strict=False)
                else:
                    raise e
            
        else:
            # If checkpoint is just the model state dict
            logger.info("Checkpoint appears to be a direct state dict")
            
            # Handle DataParallel models (strip 'module.' prefix)
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Detect architecture
            architecture = detect_model_architecture(new_state_dict)
            logger.info(f"Detected architecture: {architecture}")
            
            # Create and load model
            if architecture == 'simple_cnn':
                model = SimpleCNN(num_classes=len(class_names))
            else:
                model = create_model_from_architecture(architecture, len(class_names))
            
            model.load_state_dict(new_state_dict, strict=False)
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded and set to evaluation mode successfully")
        
        # Log training history if available
        if isinstance(checkpoint, dict) and 'training_history' in checkpoint:
            history = checkpoint['training_history']
            if isinstance(history, dict) and 'final_accuracy' in history:
                logger.info(f"Model final training accuracy: {history['final_accuracy']:.4f}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise e
    except KeyError as e:
        logger.error(f"Key error in checkpoint: {str(e)}")
        raise KeyError(f"Required key not found in checkpoint: {str(e)}")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(f"Model architecture mismatch: {str(e)}")
            raise RuntimeError(f"Model size mismatch. Please check if the model architecture matches the checkpoint: {str(e)}")
        else:
            logger.error(f"Runtime error loading model: {str(e)}")
            raise e
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

def setup_preprocessing():
    """Setup image preprocessing pipeline"""
    global transform
    
    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    logger.info("Image preprocessing pipeline setup complete")

def preprocess_image(image_file):
    """Preprocess uploaded image for model inference"""
    try:
        # Open and convert image
        image = Image.open(image_file)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image from {image.mode} to RGB")
        
        # Apply transformations
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        logger.info(f"Image preprocessed successfully. Shape: {image_tensor.shape}")
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

def predict_image(image_tensor):
    """Make prediction on preprocessed image tensor"""
    try:
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)
            
            # Get probabilities for all classes
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            predicted_idx = predicted.item()
            
            # Get class name and confidence
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()
            
            # Get all class probabilities
            all_probs = {}
            for i, class_name in enumerate(class_names):
                all_probs[class_name] = round(probabilities[0][i].item(), 4)
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            
            return predicted_class, confidence, all_probs
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise e

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ============= ROUTES =============

@app.route('/', methods=['GET'])
@app.route('/status', methods=['GET'])
def health_check():
    """Health check and status endpoint"""
    try:
        status_data = {
            "status": "API is running",
            "model_loaded": model is not None,
            "device": str(device) if device else "Not initialized",
            "classes": class_names,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "endpoints": {
                "status": {"method": "GET", "url": "/status"},
                "predict": {"method": "POST", "url": "/predict", "content_type": "multipart/form-data"}
            }
        }
        
        logger.info("Status check requested")
        return jsonify(status_data), 200
        
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return jsonify({"error": "Status check failed", "timestamp": datetime.now().isoformat()}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    request_timestamp = datetime.now().isoformat()
    logger.info(f"Received prediction request at {request_timestamp}")
    
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({
                "error": "Model not loaded. Please restart the service.",
                "timestamp": request_timestamp
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({
                "error": "No file provided. Please upload an image file.",
                "expected_field": "file",
                "content_type": "multipart/form-data",
                "timestamp": request_timestamp
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '' or file.filename is None:
            logger.warning("No file selected")
            return jsonify({
                "error": "No file selected. Please choose an image file.",
                "timestamp": request_timestamp
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({
                "error": "Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, TIFF, or WEBP files only.",
                "allowed_types": ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
                "received_file": file.filename,
                "timestamp": request_timestamp
            }), 400
        
        # Check file size (additional check beyond Flask's MAX_CONTENT_LENGTH)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({
                "error": f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB.",
                "file_size_mb": round(file_size / (1024*1024), 2),
                "max_size_mb": app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
                "timestamp": request_timestamp
            }), 413
        
        # Preprocess image
        try:
            image_tensor = preprocess_image(file)
        except OSError as e:
            logger.error(f"PIL Error - Invalid or corrupted image: {str(e)}")
            return jsonify({
                "error": "Invalid or corrupted image file. Please upload a valid image.",
                "filename": file.filename,
                "timestamp": request_timestamp
            }), 400
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return jsonify({
                "error": f"Error processing image: {str(e)}",
                "filename": file.filename,
                "timestamp": request_timestamp
            }), 400
        
        # Make prediction
        try:
            predicted_class, confidence, all_probabilities = predict_image(image_tensor)
        except RuntimeError as e:
            if "device" in str(e).lower():
                logger.error(f"Device mismatch error: {str(e)}")
                return jsonify({
                    "error": "Device mismatch error. Please restart the service.",
                    "timestamp": request_timestamp
                }), 500
            else:
                logger.error(f"Runtime error during prediction: {str(e)}")
                return jsonify({
                    "error": f"Prediction error: {str(e)}",
                    "timestamp": request_timestamp
                }), 500
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                "error": f"Error making prediction: {str(e)}",
                "timestamp": request_timestamp
            }), 500
        
        # Return successful prediction
        result = {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 4),
                "all_probabilities": all_probabilities
            },
            "file_info": {
                "filename": file.filename,
                "size_mb": round(file_size / (1024*1024), 2)
            },
            "processing_info": {
                "device": str(device),
                "model_classes": class_names,
                "timestamp": request_timestamp
            }
        }
        
        logger.info(f"Successful prediction: {predicted_class} ({confidence:.4f}) for {file.filename}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error. Please try again.",
            "timestamp": request_timestamp
        }), 500

# ============= ERROR HANDLERS =============

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    logger.warning("File too large error")
    return jsonify({
        "error": "File too large. Please upload a smaller image.",
        "max_size_mb": app.config['MAX_CONTENT_LENGTH'] // (1024*1024),
        "timestamp": datetime.now().isoformat()
    }), 413

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors"""
    logger.warning(f"Bad request: {str(e)}")
    return jsonify({
        "error": "Bad request. Please check your input.",
        "timestamp": datetime.now().isoformat()
    }), 400

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    logger.warning(f"Endpoint not found: {request.url}")
    return jsonify({
        "error": "Endpoint not found.",
        "available_endpoints": {
            "status": {"method": "GET", "url": "/status"},
            "predict": {"method": "POST", "url": "/predict"}
        },
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors"""
    logger.warning(f"Method not allowed for {request.url}: {request.method}")
    return jsonify({
        "error": f"Method {request.method} not allowed for this endpoint.",
        "allowed_methods": {
            "/status": ["GET"],
            "/predict": ["POST"]
        },
        "timestamp": datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

# ============= INITIALIZATION =============

def initialize_app():
    """Initialize the application"""
    try:
        logger.info("Initializing Civic Issue Classifier API...")
        
        # Load model
        load_model()
        
        # Setup preprocessing
        setup_preprocessing()
        
        logger.info("Application initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        return False

# ============= MAIN =============

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        logger.info("Starting Flask development server on port 5800...")
        # For development
        app.run(host='0.0.0.0', port=5800, debug=False)
    else:
        logger.error("Failed to start application due to initialization errors")
        exit(1)

# For production deployment with gunicorn
def create_app():
    """Factory function for gunicorn deployment"""
    if initialize_app():
        return app
    else:
        raise RuntimeError("Failed to initialize application")

# Gunicorn entry point
if __name__ != '__main__':
    # When run with gunicorn, initialize the app
    initialize_app()

