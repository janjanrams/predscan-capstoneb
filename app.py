from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from werkzeug.security import generate_password_hash
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()  # safe to call multiple times
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize extensions
login_manager = LoginManager()
mail = Mail()

def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Import db here to avoid circular imports
    from models import db
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///predscan.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
    
    # Email configuration for password reset
    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', '587'))
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    
    # Login manager configuration
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.admin import admin_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    # Root redirect with smart routing
    @app.route('/')
    def index():
        from flask_login import current_user
        if current_user.is_authenticated:
            if current_user.is_verified:
                return redirect(url_for('admin.index'))
            else:
                return redirect(url_for('dashboard.index'))
        return redirect(url_for('auth.login'))
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Create default admin user if none exists
        from models import User
        if not User.query.filter_by(email='admin@predscan.ai').first():
            admin_user = User(
                full_name='Admin User',
                email='admin@predscan.ai',
                password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                is_verified=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Created default admin user: admin@predscan.ai / admin123")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)