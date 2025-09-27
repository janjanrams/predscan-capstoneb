from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timedelta
import secrets
from werkzeug.security import check_password_hash, generate_password_hash
from flask_sqlalchemy import SQLAlchemy

# Create db instance - will be initialized in app.py
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and profile management"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    
    # Verification token for email verification
    verification_token = db.Column(db.String(100), unique=True)
    verification_sent_at = db.Column(db.DateTime)
    
    # Password reset functionality
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expires = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<User {self.email}>'
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password, password)
    
    def generate_verification_token(self):
        """Generate email verification token"""
        self.verification_token = secrets.token_urlsafe(32)
        self.verification_sent_at = datetime.utcnow()
        return self.verification_token
    
    def verify_email(self, token):
        """Verify email with token"""
        if self.verification_token == token:
            self.is_verified = True
            self.verification_token = None
            self.verification_sent_at = None
            return True
        return False
    
    def generate_reset_token(self):
        """Generate password reset token"""
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        return self.reset_token
    
    def verify_reset_token(self, token):
        """Verify password reset token"""
        if (self.reset_token == token and 
            self.reset_token_expires and 
            datetime.utcnow() < self.reset_token_expires):
            return True
        return False
    
    def clear_reset_token(self):
        """Clear password reset token after use"""
        self.reset_token = None
        self.reset_token_expires = None
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
    
    @staticmethod
    def get_by_email(email):
        """Get user by email address"""
        return User.query.filter_by(email=email.lower()).first()
    
    @staticmethod
    def create_user(full_name, email, password):
        """Create new user with validation"""
        # Check if user already exists
        if User.get_by_email(email):
            return None, "User with this email already exists"
        
        # Create new user
        user = User(
            full_name=full_name.strip(),
            email=email.lower().strip()
        )
        user.set_password(password)
        user.generate_verification_token()
        
        try:
            db.session.add(user)
            db.session.commit()
            return user, None
        except Exception as e:
            db.session.rollback()
            return None, f"Error creating user: {str(e)}"

class LoginAttempt(db.Model):
    """Track login attempts for security"""
    
    __tablename__ = 'login_attempts'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=False)
    success = db.Column(db.Boolean, default=False, nullable=False)
    attempt_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    user_agent = db.Column(db.Text)
    
    @staticmethod
    def log_attempt(email, ip_address, success, user_agent=None):
        """Log login attempt"""
        attempt = LoginAttempt(
            email=email.lower(),
            ip_address=ip_address,
            success=success,
            user_agent=user_agent
        )
        db.session.add(attempt)
        db.session.commit()
    
    @staticmethod
    def check_rate_limit(email, ip_address, minutes=15, max_attempts=5):
        """Check if email or IP is rate limited"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        # Check failed attempts from this email
        email_attempts = LoginAttempt.query.filter(
            LoginAttempt.email == email.lower(),
            LoginAttempt.success == False,
            LoginAttempt.attempt_time > cutoff_time
        ).count()
        
        # Check failed attempts from this IP
        ip_attempts = LoginAttempt.query.filter(
            LoginAttempt.ip_address == ip_address,
            LoginAttempt.success == False,
            LoginAttempt.attempt_time > cutoff_time
        ).count()
        
        return email_attempts >= max_attempts or ip_attempts >= max_attempts


class Analysis(db.Model):
    """Analysis model for storing saved analysis results"""
    
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    analysis_timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    # Analysis results
    result = db.Column(db.Integer, nullable=False)  # 0 = legitimate, 1 = predatory
    hdd = db.Column(db.Float, nullable=False)
    lexical_density = db.Column(db.Float, nullable=False)
    review_speed = db.Column(db.Float, nullable=False)
    remark = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    # Additional analysis features that are currently missing
    grammar_suggestions = db.Column(db.Float, nullable=True)  # Grammar errors count
    reference_count = db.Column(db.Float, nullable=True)      # Reference count
    word_count = db.Column(db.Float, nullable=True)           # Word count
    mtld = db.Column(db.Float, nullable=True)                 # Measure of Textual Lexical Diversity
    
    # Prediction probabilities and confidence
    probability_legitimate = db.Column(db.Float, nullable=True)  # Probability of being legitimate
    probability_predatory = db.Column(db.Float, nullable=True)   # Probability of being predatory 
    confidence = db.Column(db.String(20), nullable=True)        # Confidence level (Low/Medium/High)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('analyses', lazy=True, cascade='all, delete-orphan'))
    
    def __repr__(self):
        return f'<Analysis {self.id}: {self.filename} by User {self.user_id}>'
    
    @staticmethod
    def create_analysis(user_id, filename, result, hdd, lexical_density, review_speed, 
                       remark, description, grammar_suggestions=None, reference_count=None, 
                       word_count=None, mtld=None, probability_legitimate=None, 
                       probability_predatory=None, confidence=None, analysis_timestamp=None):
        """Create new analysis record"""
        analysis = Analysis(
            user_id=user_id,
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
            confidence=confidence
        )
        
        # Set custom timestamp if provided
        if analysis_timestamp:
            analysis.analysis_timestamp = analysis_timestamp
        
        try:
            db.session.add(analysis)
            db.session.commit()
            return analysis, None
        except Exception as e:
            db.session.rollback()
            return None, f"Error saving analysis: {str(e)}"
    
    def get_result_status(self):
        """Get human-readable result status"""
        # Use probability-based determination if probabilities are available
        if self.probability_predatory is not None and self.probability_legitimate is not None:
            return "Suspected Predatory" if self.probability_predatory > self.probability_legitimate else "Non-Predatory"
        # Fallback to result field for backward compatibility
        return "Suspected Predatory" if self.result == 1 else "Non-Predatory"
    
    def get_risk_level(self):
        """Get risk level based on result"""
        # Use probability-based determination if probabilities are available
        if self.probability_predatory is not None and self.probability_legitimate is not None:
            return "high" if self.probability_predatory > self.probability_legitimate else "low"
        # Fallback to result field for backward compatibility
        return "high" if self.result == 1 else "low"
    
    def format_timestamp(self):
        """Format analysis timestamp for display"""
        return self.analysis_timestamp.strftime("%B %d, %Y at %I:%M %p")
    
    @staticmethod
    def delete_analysis(analysis_id, user_id):
        """Delete analysis record with user ownership verification"""
        try:
            # Find the analysis by ID
            analysis = Analysis.query.filter_by(id=analysis_id).first()
            
            if not analysis:
                return False, "Analysis not found"
            
            # Verify ownership - security check
            if analysis.user_id != user_id:
                return False, "Unauthorized: Analysis belongs to another user"
            
            # Delete the analysis
            db.session.delete(analysis)
            db.session.commit()
            return True, "Analysis deleted successfully"
            
        except Exception as e:
            db.session.rollback()
            return False, f"Error deleting analysis: {str(e)}"
    
    @property
    def overall_score(self):
        """Get the probability score for this analysis
        Returns the appropriate probability value based on the result
        """
        # Use stored probabilities if available
        if self.result == 1:  # Predatory
            if self.probability_predatory is not None:
                return self.probability_predatory / 100
        else:  # Legitimate
            if self.probability_legitimate is not None:
                return self.probability_legitimate / 100
        
        # Fallback to calculated score if probabilities not available
        feature_score = (self.hdd * 0.3 + 
                        self.lexical_density * 0.3 + 
                        (1 - min(self.review_speed/180, 1)) * 0.4)
        return feature_score