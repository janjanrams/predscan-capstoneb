"""
Dashboard routes for PredscanAI
Protected routes that require authentication
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from flask_login import login_required, current_user
from functools import wraps
from werkzeug.utils import secure_filename
import os
import random
import time
import pickle
import pandas as pd
from datetime import datetime
from models import Analysis
from models import db

# Import analysis functions from utils.py
from utils import extract_features_multiprocessed_article, predscan_analyze

# Pre-load models at startup for better performance
def preload_models():
    """Pre-load models at startup to improve first-request performance"""
    print("Pre-loading analysis models for optimal performance...")
    try:
        # Pre-load dashboard models
        model = get_cached_model()
        scaler = get_cached_scaler()
        
        # Pre-load utils models
        from utils import _get_cached_model, _get_cached_scaler
        utils_model = _get_cached_model()
        utils_scaler = _get_cached_scaler()
        
        if model and scaler and utils_model and utils_scaler:
            print("✓ All models pre-loaded successfully - ready for fast analysis")
        else:
            print("⚠️ Warning: Some models failed to pre-load")
    except Exception as e:
        print(f"⚠️ Error during model pre-loading: {e}")

# Initialize models when the module loads
try:
    preload_models()
except Exception as e:
    print(f"Module initialization warning: {e}")

dashboard_bp = Blueprint('dashboard', __name__)

# PERFORMANCE OPTIMIZATION: Global model cache to avoid reloading on every request
_MODEL_CACHE = {}
_SCALER_CACHE = {}

def get_cached_model():
    """Get the trained model with caching for performance"""
    if 'model' not in _MODEL_CACHE:
        try:
            print("Loading gradient boosting model (first time)...")
            _MODEL_CACHE['model'] = pickle.load(open('final_gradient_boosting.pkl', 'rb'))
            print("Model loaded and cached successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return _MODEL_CACHE['model']

def get_cached_scaler():
    """Get the preprocessing scaler with caching for performance"""
    if 'scaler' not in _SCALER_CACHE:
        try:
            print("Loading preprocessing scaler (first time)...")
            import joblib
            _SCALER_CACHE['scaler'] = joblib.load('preprocessing_scaler.pkl')
            print("Scaler loaded and cached successfully")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return None
    return _SCALER_CACHE['scaler']

def unverified_user_required(f):
    """Decorator to ensure only unverified users access regular dashboard"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if current_user.is_verified:
            flash('You have admin access. Redirecting to admin dashboard.', 'info')
            return redirect(url_for('admin.index'))
        return f(*args, **kwargs)
    return decorated_function

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_upload_folder():
    """Create upload folder if it doesn't exist"""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

@dashboard_bp.route('/')
@unverified_user_required  
def index():
    """Main dashboard page - shows recent analyses"""
    # Query user's saved analyses, ordered by timestamp (most recent first), limit to 10
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.analysis_timestamp.desc()).limit(10).all()
    return render_template('dashboard/index.html', user=current_user, analyses=analyses)

@dashboard_bp.route('/analysis')
@unverified_user_required
def analysis():
    """Redirect old analysis route to new analyzer page"""
    return redirect(url_for('dashboard.analyzer'))


@dashboard_bp.route('/analyzer', methods=['GET', 'POST'])
@unverified_user_required
def analyzer():
    """Document analyzer page with Dashboard-consistent design"""
    
    if request.method == 'POST':
        # Check if this is a request to load an existing analysis
        analysis_id = request.form.get('analysis_id')
        if analysis_id:
            return handle_analyzer_view_analysis(analysis_id)
        # Otherwise, handle normal file upload
        return handle_analyzer_upload()
    return render_template('analyzer/index.html', user=current_user)

def handle_analyzer_view_analysis(analysis_id):
    """Handle POST request to view an existing analysis in the analyzer - merges functionality from view_analysis route"""
    try:
        # Validate analysis_id is an integer
        try:
            analysis_id = int(analysis_id)
        except (ValueError, TypeError):
            flash('Invalid analysis ID.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Find the analysis by ID
        analysis = Analysis.query.get(analysis_id)
        
        # Check if analysis exists
        if not analysis:
            flash('Analysis not found.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Validate that the analysis belongs to the current user
        if analysis.user_id != current_user.id:
            flash('Unauthorized: Analysis does not belong to current user.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Prepare analysis data for rendering (same logic as view_analysis route)
        analysis_results = {
            'hdd': analysis.hdd,
            'lexical_density': analysis.lexical_density,
            'review_speed': analysis.review_speed,
            'result': analysis.result,
            'remark': analysis.remark,
            'description': analysis.description,
            'filename': analysis.filename
        }
        
        # Try to find the actual file with timestamp prefix for preview
        upload_folder = os.path.join('static', 'uploads')
        possible_files = []
        if os.path.exists(upload_folder):
            for file in os.listdir(upload_folder):
                if file.endswith(f'_{analysis.filename}'):
                    possible_files.append(file)
        
        # Use the most recent file if multiple matches (should be rare)
        if possible_files:
            preview_filename = max(possible_files)  # Latest timestamp
            analysis_results['preview_filename'] = preview_filename
            # Try static URL first, fallback to dashboard route
            try:
                analysis_results['preview_url'] = url_for('static', filename=f'uploads/{preview_filename}')
            except Exception:
                analysis_results['preview_url'] = url_for('dashboard.uploaded_file', filename=preview_filename)
        
        # Create analysis_result object for the template (same structure as handle_analyzer_upload)
        # Calculate probabilities based on result (0 = legitimate, 1 = predatory)
        if analysis_results['result'] == 1:
            # Predatory document
            probability_predatory = 85.0  # Default high confidence for predatory
            probability_legitimate = 15.0
        else:
            # Legitimate document
            probability_legitimate = 85.0  # Default high confidence for legitimate
            probability_predatory = 15.0
            
        analysis_result = {
            'success': True,
            'prediction': analysis_results['result'],
            'probability_legitimate': float(analysis.probability_legitimate) if analysis.probability_legitimate is not None else float(probability_legitimate),
            'probability_predatory': float(analysis.probability_predatory) if analysis.probability_predatory is not None else float(probability_predatory),
            'confidence': analysis.confidence or 'High',  # Use stored confidence or default
            'features_used': {
                # Include ALL the features that the template expects - using the exact names expected by the template
                'grammar_suggestions': float(analysis.grammar_suggestions) if analysis.grammar_suggestions is not None else 0.0,
                'hdd': float(analysis.hdd) if analysis.hdd is not None else 0.0,
                'reference_count': float(analysis.reference_count) if analysis.reference_count is not None else 0.0,
                'word_count': float(analysis.word_count) if analysis.word_count is not None else 0.0,
                'review_speed': float(analysis.review_speed) if analysis.review_speed is not None else 90.0,
                'mtld': float(analysis.mtld) if analysis.mtld is not None else 0.0,
                'lexical_density': float(analysis.lexical_density) if analysis.lexical_density is not None else 0.0
            },
            'model_info': {
                'model_name': 'Gradient Boosting Classifier',
                'features_count': 5,
                'preprocessing': 'RobustScaler'
            },
            'remark': analysis_results['remark'],
            'description': analysis_results['description'],
            'details': {
                'word_count': analysis.word_count or 'N/A',
                'grammar_errors': analysis.grammar_suggestions or 'N/A',
                'hdd': analysis_results['hdd'],
                'lexical_density': analysis_results['lexical_density'],
                'review_speed': analysis_results['review_speed'],
                'mtld': analysis.mtld or 'N/A',
                'readability': 'N/A',  # Not stored in saved analyses
                'reference_count': analysis.reference_count or 0
            }
        }
        
        flash('Analysis loaded successfully!', 'info')
        
        # Render the analyzer template with the saved analysis data
        return render_template('analyzer/index.html', 
                             user=current_user, 
                             analysis_result=analysis_result,
                             # Keep legacy variables for compatibility
                             **analysis_results)
        
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}', 'error')
        return redirect(url_for('dashboard.analyzer'))

def handle_analyzer_upload():
    """Handle PDF upload and predatory journal analysis using real AI models"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            flash('Please select a file to upload.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('Please select a file to upload.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Validate file
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PDF files only.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            flash('File too large. Maximum file size is 10MB.', 'error')
            return redirect(url_for('dashboard.analyzer'))
        
        # Save file to static/uploads folder for iframe preview
        upload_folder = os.path.join('static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        
        # REAL ANALYSIS: Extract features using optimized utils.py functions
        print(f"Starting optimized analysis of {filename}...")
        
        try:
            # Step 1: Extract features from PDF (now with enhanced caching and progress tracking)
            print(f"  Phase 1/3: Extracting features from PDF...")
            features = extract_features_multiprocessed_article(file_path)
            
            if features is None:
                print(f"  ERROR: Feature extraction failed for {filename}")
                # Create failed analysis result object for template
                failed_analysis_result = {
                    'success': False,
                    'error_message': 'Failed to extract features from the PDF. Please ensure the document contains readable text and is not corrupted.',
                    'filename': filename
                }
                
                return render_template('analyzer/index.html', 
                                     user=current_user,
                                     analysis_result=failed_analysis_result,
                                     filename=filename,
                                     preview_filename=unique_filename,
                                     preview_url=url_for('static', filename=f'uploads/{unique_filename}'))
            
            # Step 2: Run prediction analysis with cached models
            print(f"  Phase 2/3: Running ML prediction with cached models...")
            prediction_results = predscan_analyze(features)
            
            if not prediction_results.get('success', False):
                print(f"  ERROR: Prediction failed for {filename}")
                error_msg = prediction_results.get('error_message', 'Unknown analysis error')
                
                # Create failed analysis result object for template
                failed_analysis_result = {
                    'success': False,
                    'error_message': error_msg,
                    'filename': filename
                }
                
                return render_template('analyzer/index.html', 
                                     user=current_user,
                                     analysis_result=failed_analysis_result,
                                     filename=filename,
                                     preview_filename=unique_filename,
                                     preview_url=url_for('static', filename=f'uploads/{unique_filename}'))
            
            # Step 3: Format results for template
            print(f"  Phase 3/3: Formatting results for display...")
            analysis_results = format_analysis_results(features, prediction_results)
            print(f"  SUCCESS: Analysis completed for {filename}")
            
        except Exception as analysis_error:
            print(f"Analysis error: {analysis_error}")
            
            # Create failed analysis result object for template
            failed_analysis_result = {
                'success': False,
                'error_message': f'Analysis failed due to processing error: {str(analysis_error)}',
                'filename': filename
            }
            
            return render_template('analyzer/index.html', 
                                 user=current_user,
                                 analysis_result=failed_analysis_result,
                                 filename=filename,
                                 preview_filename=unique_filename,
                                 preview_url=url_for('static', filename=f'uploads/{unique_filename}'))
        
        # Add file information for template
        analysis_results['filename'] = filename  # For database storage
        analysis_results['preview_filename'] = unique_filename  # For PDF preview
        
        # Try static URL first, fallback to dashboard route
        try:
            analysis_results['preview_url'] = url_for('static', filename=f'uploads/{unique_filename}')
        except Exception:
            analysis_results['preview_url'] = url_for('dashboard.uploaded_file', filename=unique_filename)
        
        # Create analysis_result object for the updated template
        analysis_result = {
            'success': True,
            'prediction': analysis_results['result'],
            'probability_legitimate': analysis_results.get('probability_legitimate', 100 - analysis_results.get('probability_predatory', 50)),
            'probability_predatory': analysis_results.get('probability_predatory', 50),
            'confidence': analysis_results.get('confidence_level', 'Medium'),
            'features_used': prediction_results.get('features_used', {}),  # Use features_used directly from prediction_results
            'model_info': prediction_results.get('model_info', {
                'model_name': 'Gradient Boosting Classifier',
                'features_count': 5,  # Updated to reflect the 5 selected features
                'preprocessing': 'RobustScaler'
            }),
            'remark': analysis_results['remark'],
            'description': analysis_results['description'],
            'details': {
                'word_count': analysis_results.get('word_count', 'N/A'),
                'grammar_errors': analysis_results.get('grammar_errors', 'N/A'),
                'hdd': analysis_results['hdd'],
                'lexical_density': analysis_results['lexical_density'],
                'review_speed': analysis_results['review_speed'],
                'mtld': analysis_results.get('mtld', 'N/A'),
                'readability': analysis_results.get('readability', 'N/A'),
                'reference_count': analysis_results.get('reference_count', 0)
            }
        }
        
        flash('Document analysis completed successfully!', 'success')
        return render_template('analyzer/index.html', 
                             user=current_user, 
                             analysis_result=analysis_result,
                             # Keep legacy variables for compatibility
                             **analysis_results)
        
    except Exception as e:
        print(f"Upload handler error: {e}")
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('dashboard.analyzer'))

def format_analysis_results(features, prediction_results):
    """Format analysis results from utils.py functions for template display"""
    
    # Extract prediction information
    result = prediction_results['prediction']  # 0 or 1
    probability_legitimate = prediction_results['probability_legitimate']
    probability_predatory = prediction_results['probability_predatory']
    confidence = prediction_results['confidence']
    
    # Get grammar errors and word count for calculations
    grammar_errors = features.get('grammar_errors', 0)
    word_count = features.get('word_count', 1)  # Avoid division by zero
    
    # Get other metrics from features
    hdd = features.get('hdd', 0.5)
    lexical_density = features.get('lexical_density', 0.5)
    review_speed = features.get('review_speed', 90)  # Default to 90 days if None
    
    # Generate appropriate remark and description based on prediction
    if result == 1:
        remark = f"Warning: This document shows characteristics commonly associated with predatory journals (Confidence: {confidence}, {probability_predatory:.1f}% predatory)."
        description = f"The AI analysis indicates potential issues that may suggest predatory publishing practices. Key indicators include linguistic patterns, review process characteristics, and structural elements. The model confidence is {confidence.lower()} with a {probability_predatory:.1f}% probability of being predatory. Please verify the journal's reputation through additional sources."
    else:
        remark = f"This document appears to be from a legitimate academic source (Confidence: {confidence}, {probability_legitimate:.1f}% legitimate)."
        description = f"The AI analysis shows good quality indicators typical of reputable academic journals. The linguistic patterns, structural elements, and other features meet expected standards for legitimate publications. The model confidence is {confidence.lower()} with a {probability_legitimate:.1f}% probability of being legitimate."
    
    return {
        'hdd': hdd,
        'lexical_density': lexical_density,
        'review_speed': review_speed if review_speed is not None else 90,
        'result': result,
        'remark': remark,
        'description': description,
        
        # Additional information for debugging/transparency
        'word_count': word_count,
        'grammar_errors': grammar_errors,
        'mtld': features.get('mtld', 'N/A'),
        'readability': features.get('readability', 'N/A'),
        'reference_count': features.get('reference_count', 0),
        'received_date': features.get('received_date', None),
        'accepted_date': features.get('accepted_date', None),
        
        # Prediction details
        'probability_legitimate': probability_legitimate,
        'probability_predatory': probability_predatory,
        'confidence_level': confidence,
        'model_features_used': prediction_results.get('features_used', {})
    }

@dashboard_bp.route('/save-analysis', methods=['POST'])
@unverified_user_required
def save_analysis():
    """Save analysis results to the database"""
    try:
        # Check if request contains JSON data
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['result', 'hdd', 
                          'lexical_density', 'review_speed', 'remark', 'description', 'filename',
                          'grammar_suggestions', 'reference_count', 'word_count', 'mtld',
                          'probability_legitimate', 'probability_predatory', 'confidence', 'analysis_timestamp']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract data from request
        result = data['result']
        hdd = data['hdd']
        lexical_density = data['lexical_density']
        review_speed = data['review_speed']
        remark = data['remark']
        description = data['description']
        filename = data['filename']
        
        # Extract the new fields
        grammar_suggestions = data['grammar_suggestions']
        reference_count = data['reference_count']
        word_count = data['word_count']
        mtld = data['mtld']
        probability_legitimate = data['probability_legitimate']
        probability_predatory = data['probability_predatory']
        confidence = data['confidence']
        analysis_timestamp_str = data['analysis_timestamp']
        
        # Validate data types
        try:
            result = int(result)
            hdd = float(hdd)
            lexical_density = float(lexical_density)
            review_speed = float(review_speed)
            
            # Validate the new numeric fields
            grammar_suggestions = float(grammar_suggestions)
            reference_count = float(reference_count)
            word_count = float(word_count)
            mtld = float(mtld)
            probability_legitimate = float(probability_legitimate)
            probability_predatory = float(probability_predatory)
            # confidence is a string, no conversion needed
            
            # Parse the timestamp from local time format (no timezone conversion needed)
            analysis_timestamp = datetime.fromisoformat(analysis_timestamp_str)
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'message': f'Invalid data types for fields: {str(e)}'
            }), 400
        
        # Save analysis using the Analysis model
        analysis, error = Analysis.create_analysis(
            user_id=current_user.id,
            filename=filename,
            result=result,
            hdd=hdd,
            lexical_density=lexical_density,
            review_speed=review_speed,
            remark=remark,
            description=description,
            grammar_suggestions=grammar_suggestions,
            reference_count=reference_count,
            word_count=word_count,
            mtld=mtld,
            probability_legitimate=probability_legitimate,
            probability_predatory=probability_predatory,
            confidence=confidence,
            analysis_timestamp=analysis_timestamp
        )
        
        if error:
            return jsonify({
                'success': False,
                'message': f'Database error: {error}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Analysis saved successfully',
            'analysis_id': analysis.id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred while saving analysis: {str(e)}'
        }), 500

# Route to serve uploaded files (matches thesisA pattern of /filename)
@dashboard_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files for iframe preview"""
    upload_folder = os.path.join('static', 'uploads')
    
    # Validate filename to prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        flash('Invalid filename.', 'error')
        return redirect(url_for('dashboard.analyzer'))
    
    try:
        response = send_from_directory(upload_folder, filename)
        # Set proper headers for PDF display in iframe
        if filename.lower().endswith('.pdf'):
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = 'inline'
        return response
    except FileNotFoundError:
        flash('File not found.', 'error')
        return redirect(url_for('dashboard.analyzer'))

@dashboard_bp.route('/delete-analysis/<int:analysis_id>', methods=['POST', 'DELETE'])
@unverified_user_required
def delete_analysis(analysis_id):
    """Delete an analysis - returns JSON response for AJAX handling"""
    try:
        # Use the Analysis model's delete_analysis method
        success, message = Analysis.delete_analysis(analysis_id, current_user.id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            # Determine appropriate status code based on message content
            if 'not found' in message.lower():
                status_code = 404
            elif 'unauthorized' in message.lower() or 'does not belong' in message.lower():
                status_code = 403
            else:
                status_code = 500
            
            return jsonify({
                'success': False,
                'message': message
            }), status_code
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

@dashboard_bp.route('/view-analysis/<int:analysis_id>')
@unverified_user_required
def view_analysis(analysis_id):
    """View a saved analysis - redirects to analyzer with analysis data loaded"""
    try:
        # Find the analysis by ID
        analysis = Analysis.query.get(analysis_id)
        
        # Check if analysis exists
        if not analysis:
            flash('Analysis not found.', 'error')
            return redirect(url_for('dashboard.index'))
        
        # Validate that the analysis belongs to the current user
        if analysis.user_id != current_user.id:
            flash('Unauthorized: Analysis does not belong to current user.', 'error')
            return redirect(url_for('dashboard.index'))
        
        # Prepare analysis data for rendering
        analysis_results = {
            'hdd': analysis.hdd,
            'lexical_density': analysis.lexical_density,
            'review_speed': analysis.review_speed,
            'result': analysis.result,
            'remark': analysis.remark,
            'description': analysis.description,
            'filename': analysis.filename
        }
        
        # Try to find the actual file with timestamp prefix for preview
        upload_folder = os.path.join('static', 'uploads')
        possible_files = []
        if os.path.exists(upload_folder):
            for file in os.listdir(upload_folder):
                if file.endswith(f'_{analysis.filename}'):
                    possible_files.append(file)
        
        # Use the most recent file if multiple matches (should be rare)
        if possible_files:
            preview_filename = max(possible_files)  # Latest timestamp
            analysis_results['preview_filename'] = preview_filename
            # Try static URL first, fallback to dashboard route
            try:
                analysis_results['preview_url'] = url_for('static', filename=f'uploads/{preview_filename}')
            except Exception:
                analysis_results['preview_url'] = url_for('dashboard.uploaded_file', filename=preview_filename)
        
        # Render the analyzer template with the saved analysis data
        return render_template('analyzer/index.html', 
                             user=current_user, 
                             **analysis_results)
        
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}', 'error')
        return redirect(url_for('dashboard.index'))