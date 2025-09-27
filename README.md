# PredscanAI Flask Application

A secure Flask web application for journal authenticity analysis with complete user authentication system.

## Features

### ✅ **Authentication System**
- **User Registration**: Secure account creation with email verification
- **Login/Logout**: Session-based authentication with remember me option
- **Password Reset**: Email-based password recovery system
- **Security**: Password hashing, rate limiting, and CSRF protection

### 🎨 **Modern UI/UX**
- **Responsive Design**: Mobile-first approach that works on all devices
- **Clean Aesthetics**: Minimalistic design with consistent color theme
- **Form Validation**: Real-time client-side and server-side validation
- **User Feedback**: Flash messages and loading states

### 🏗️ **Technical Architecture**
- **Flask Blueprints**: Modular route organization
- **SQLAlchemy ORM**: Database abstraction with SQLite
- **Flask-Login**: Session management
- **Flask-Mail**: Email functionality
- **Jinja2 Templates**: Dynamic HTML rendering

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```
SECRET_KEY=your-secure-secret-key
DATABASE_URL=sqlite:///predscan.db
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

### 3. Run Application
```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`

## Default Credentials

**Admin Account:**
- Email: `admin@predscan.ai`
- Password: `admin123`

## Project Structure

```
predscan/
├── app.py                 # Flask application factory
├── models.py             # Database models
├── run.py               # Application runner
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
│
├── routes/             # Flask blueprints
│   ├── __init__.py
│   ├── auth.py        # Authentication routes
│   └── dashboard.py   # Dashboard routes
│
├── templates/         # Jinja2 templates
│   ├── base.html     # Base template
│   ├── auth/         # Authentication templates
│   │   ├── login.html
│   │   ├── signup.html
│   │   ├── forgot_password.html
│   │   └── reset_password.html
│   └── dashboard/    # Dashboard templates
│       └── index.html
│
└── static/           # Static files
    ├── css/
    │   ├── auth.css     # Authentication styles
    │   └── dashboard.css # Dashboard styles
    └── js/             # JavaScript files
```

## Routes

### Authentication Routes (`/auth/`)
- `GET/POST /auth/login` - User login
- `GET/POST /auth/signup` - User registration
- `GET/POST /auth/forgot-password` - Password reset request
- `GET/POST /auth/reset-password/<token>` - Password reset with token
- `GET /auth/verify-email/<token>` - Email verification
- `GET /auth/logout` - User logout

### Dashboard Routes (`/dashboard/`)
- `GET /dashboard/` - Main dashboard (login required)

## Security Features

### Password Security
- **PBKDF2 Hashing**: Secure password storage
- **Strength Validation**: Minimum 8 characters with letters and numbers
- **Rate Limiting**: Prevents brute force attacks

### Session Management
- **Flask-Login Integration**: Secure session handling
- **Remember Me**: Optional persistent sessions
- **Automatic Logout**: Session timeout protection

### Input Validation
- **Email Format**: Regex validation for email addresses
- **CSRF Protection**: Built-in Flask security
- **SQL Injection**: SQLAlchemy ORM protection

## Email Configuration

For password reset functionality, configure email settings in `.env`:

### Gmail Setup
1. Enable 2-factor authentication
2. Generate app-specific password
3. Use app password in `MAIL_PASSWORD`

### Other Providers
Update `MAIL_SERVER` and `MAIL_PORT` in `.env` for your email provider.

## Database

The application uses SQLite by default. The database includes:

### Users Table
- User credentials and profile information
- Email verification status
- Password reset tokens
- Account creation and login tracking

### Login Attempts Table
- Security logging for failed login attempts
- IP address and user agent tracking
- Rate limiting enforcement

## Development

### Running in Development Mode
```bash
export FLASK_DEBUG=True
python run.py
```

### Database Migrations
To reset the database:
```bash
rm predscan.db
python run.py
```

## Deployment Considerations

### Production Settings
- Set `FLASK_DEBUG=False`
- Use strong `SECRET_KEY`
- Configure production database (PostgreSQL recommended)
- Set up proper email service
- Enable HTTPS
- Configure reverse proxy (nginx recommended)

### Environment Variables
```bash
SECRET_KEY=production-secret-key
DATABASE_URL=postgresql://user:password@localhost/predscan
FLASK_DEBUG=False
```

## Next Steps

The dashboard is currently a simple welcome page. Future enhancements could include:

- PDF analysis functionality integration
- User profile management
- Analytics dashboard
- Report generation
- API endpoints for analysis tools

## Support

For issues or questions:
- Check the Flask documentation
- Review error logs in the console
- Verify environment configuration
- Test with default admin credentials