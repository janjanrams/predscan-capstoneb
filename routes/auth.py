"""
Authentication routes for PredscanAI
Handles login, signup, password reset, and email verification
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_mail import Message
from werkzeug.security import generate_password_hash
from models import User, LoginAttempt, db
from app import mail
import re
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    return re.match(pattern, email) is not None

def is_strong_password(password):
    """Check if password meets strength requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Password is strong"

def send_email(to, subject, template, **kwargs):
    """Send email using Flask-Mail"""
    try:
        msg = Message(
            subject=f"PredscanAI - {subject}",
            recipients=[to],
            html=template,
            sender=current_app.config['MAIL_DEFAULT_SENDER']
        )
        mail.send(msg)
        return True
    except Exception as e:
        current_app.logger.error(f"Failed to send email: {e}")
        return False

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        # Redirect based on verification status
        if current_user.is_verified:
            return redirect(url_for('admin.index'))
        else:
            return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        # Get client IP and user agent
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        user_agent = request.headers.get('User-Agent', '')
        
        # Input validation
        if not email or not password:
            error = "Please fill in all fields"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/login.html'), 400
        
        if not is_valid_email(email):
            error = "Please enter a valid email address"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/login.html'), 400
        
        # Check rate limiting
        if LoginAttempt.check_rate_limit(email, client_ip):
            error = "Too many failed login attempts. Please try again later."
            if request.is_json:
                return jsonify({"success": False, "message": error}), 429
            flash(error, 'error')
            return render_template('auth/login.html'), 429
        
        # Find user and verify password
        user = User.get_by_email(email)
        
        if user and user.check_password(password) and user.is_active:
            # Log successful attempt
            LoginAttempt.log_attempt(email, client_ip, True, user_agent)
            
            # Update last login
            user.update_last_login()
            db.session.commit()
            
            # Log in user
            login_user(user, remember=remember_me)
            
            # Determine redirect URL based on verification status
            if user.is_verified:
                # Verified user - redirect to admin dashboard
                redirect_url = url_for('admin.index')
            else:
                # Unverified user - redirect to normal user dashboard  
                redirect_url = url_for('dashboard.index')
            
            # Handle JSON response
            if request.is_json:
                return jsonify({
                    "success": True, 
                    "message": "Login successful!",
                    "redirect_url": redirect_url
                })
            
            # Handle form submission
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = redirect_url
            return redirect(next_page)
        
        else:
            # Log failed attempt
            LoginAttempt.log_attempt(email, client_ip, False, user_agent)
            
            error = "Invalid email or password"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 401
            flash(error, 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if current_user.is_authenticated:
        # Redirect based on verification status
        if current_user.is_verified:
            return redirect(url_for('admin.index'))
        else:
            return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Input validation
        errors = []
        
        if not full_name:
            errors.append("Full name is required")
        elif len(full_name) < 2:
            errors.append("Full name must be at least 2 characters long")
        
        if not email:
            errors.append("Email is required")
        elif not is_valid_email(email):
            errors.append("Please enter a valid email address")
        
        if not password:
            errors.append("Password is required")
        else:
            is_strong, strength_message = is_strong_password(password)
            if not is_strong:
                errors.append(strength_message)
        
        if password != confirm_password:
            errors.append("Passwords do not match")
        
        # Check if user already exists
        if User.get_by_email(email):
            errors.append("An account with this email already exists")
        
        if errors:
            error_message = "; ".join(errors)
            if request.is_json:
                return jsonify({"success": False, "message": error_message}), 400
            for error in errors:
                flash(error, 'error')
            return render_template('auth/signup.html'), 400
        
        # Create user
        user, create_error = User.create_user(full_name, email, password)
        
        if create_error:
            if request.is_json:
                return jsonify({"success": False, "message": create_error}), 500
            flash(create_error, 'error')
            return render_template('auth/signup.html'), 500
        
        # Send verification email
        verification_url = url_for('auth.verify_email', token=user.verification_token, _external=True)
        email_template = f"""
        <div style="max-width: 600px; margin: 0 auto; padding: 20px; font-family: Inter, sans-serif;">
            <h2 style="color: #053460;">Welcome to PredscanAI!</h2>
            <p>Thank you for signing up. Please verify your email address by clicking the link below:</p>
            <a href="{verification_url}" 
               style="display: inline-block; padding: 12px 24px; background-color: #053460; color: white; text-decoration: none; border-radius: 8px; margin: 20px 0;">
                Verify Email Address
            </a>
            <p>If you didn't create this account, you can safely ignore this email.</p>
            <p>Best regards,<br>The PredscanAI Team</p>
        </div>
        """
        
        email_sent = send_email(email, "Verify Your Email", email_template)
        
        if request.is_json:
            return jsonify({
                "success": True,
                "message": "Account created successfully! Please check your email to verify your account.",
                "email_sent": email_sent
            })
        
        flash("Account created successfully! Please check your email to verify your account.", 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/signup.html')

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Password reset request"""
    if current_user.is_authenticated:
        # Redirect based on verification status
        if current_user.is_verified:
            return redirect(url_for('admin.index'))
        else:
            return redirect(url_for('dashboard.index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        
        if not email:
            error = "Email address is required"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/forgot_password.html')
        
        if not is_valid_email(email):
            error = "Please enter a valid email address"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/forgot_password.html')
        
        # Find user (but don't reveal if they exist for security)
        user = User.get_by_email(email)
        
        if user and user.is_active:
            # Generate reset token
            reset_token = user.generate_reset_token()
            db.session.commit()
            
            # Send reset email
            reset_url = url_for('auth.reset_password', token=reset_token, _external=True)
            email_template = f"""
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; font-family: Inter, sans-serif;">
                <h2 style="color: #053460;">Password Reset Request</h2>
                <p>You requested a password reset for your PredscanAI account.</p>
                <p>Click the link below to reset your password (valid for 1 hour):</p>
                <a href="{reset_url}" 
                   style="display: inline-block; padding: 12px 24px; background-color: #f15f28; color: white; text-decoration: none; border-radius: 8px; margin: 20px 0;">
                    Reset Password
                </a>
                <p>If you didn't request this reset, you can safely ignore this email.</p>
                <p>Best regards,<br>The PredscanAI Team</p>
            </div>
            """
            
            send_email(email, "Password Reset Request", email_template)
        
        # Always show success message for security
        success_message = f"If an account with {email} exists, password reset instructions have been sent."
        
        if request.is_json:
            return jsonify({"success": True, "message": success_message})
        
        flash(success_message, 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/forgot_password.html')

@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Password reset with token"""
    if current_user.is_authenticated:
        # Redirect based on verification status
        if current_user.is_verified:
            return redirect(url_for('admin.index'))
        else:
            return redirect(url_for('dashboard.index'))
    
    # Find user with valid reset token
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or not user.verify_reset_token(token):
        flash('Invalid or expired password reset link', 'error')
        return redirect(url_for('auth.forgot_password'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not password:
            error = "Password is required"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/reset_password.html', token=token)
        
        is_strong, strength_message = is_strong_password(password)
        if not is_strong:
            if request.is_json:
                return jsonify({"success": False, "message": strength_message}), 400
            flash(strength_message, 'error')
            return render_template('auth/reset_password.html', token=token)
        
        if password != confirm_password:
            error = "Passwords do not match"
            if request.is_json:
                return jsonify({"success": False, "message": error}), 400
            flash(error, 'error')
            return render_template('auth/reset_password.html', token=token)
        
        # Update password
        user.set_password(password)
        user.clear_reset_token()
        db.session.commit()
        
        success_message = "Password reset successful! You can now log in with your new password."
        
        if request.is_json:
            return jsonify({
                "success": True,
                "message": success_message,
                "redirect_url": url_for('auth.login')
            })
        
        flash(success_message, 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_password.html', token=token)

@auth_bp.route('/verify-email/<token>')
def verify_email(token):
    """Email verification"""
    if current_user.is_authenticated:
        flash('Your email is already verified', 'info')
        # Redirect based on verification status
        if current_user.is_verified:
            return redirect(url_for('admin.index'))
        else:
            return redirect(url_for('dashboard.index'))
    
    # Find user with verification token
    user = User.query.filter_by(verification_token=token).first()
    
    if not user:
        flash('Invalid verification link', 'error')
        return redirect(url_for('auth.login'))
    
    if user.is_verified:
        flash('Your email is already verified', 'info')
        return redirect(url_for('auth.login'))
    
    # Verify email
    if user.verify_email(token):
        db.session.commit()
        flash('Email verified successfully! You can now log in.', 'success')
    else:
        flash('Email verification failed', 'error')
    
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('auth.login'))