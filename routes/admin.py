"""
Admin dashboard routes for PredscanAI
Protected routes that require admin verification (is_verified=True)
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, timedelta
from sqlalchemy import func, case
from models import User, Analysis, LoginAttempt, db

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """Decorator to require admin verification"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        if not current_user.is_verified:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('dashboard.index'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/')
@admin_required
def index():
    """Admin dashboard main page"""
    try:
        # Get basic user statistics with error handling
        total_users = User.query.count() or 0
        verified_users = User.query.filter_by(is_verified=True).count() or 0
        unverified_users = User.query.filter_by(is_verified=False).count() or 0
        
        # Get total analyses count
        total_analyses = Analysis.query.count() or 0
        
        # Calculate predatory and non-predatory counts based on probabilities
        predatory_count = 0
        non_predatory_count = 0
        
        if total_analyses > 0:
            # Count analyses where probability_predatory > probability_legitimate
            predatory_count = db.session.query(Analysis).filter(
                Analysis.probability_predatory > Analysis.probability_legitimate
            ).count() or 0
            
            # Count analyses where probability_legitimate > probability_predatory
            non_predatory_count = db.session.query(Analysis).filter(
                Analysis.probability_legitimate > Analysis.probability_predatory
            ).count() or 0
            
            # Handle cases where probabilities are equal or null by using the result column
            remaining_analyses = total_analyses - predatory_count - non_predatory_count
            if remaining_analyses > 0:
                # Use the result column as fallback for analyses without clear probability distinction
                fallback_predatory = Analysis.query.filter(
                    Analysis.result == 1,
                    (Analysis.probability_predatory.is_(None) | 
                     Analysis.probability_legitimate.is_(None) |
                     (Analysis.probability_predatory == Analysis.probability_legitimate))
                ).count() or 0
                
                fallback_legitimate = Analysis.query.filter(
                    Analysis.result == 0,
                    (Analysis.probability_predatory.is_(None) | 
                     Analysis.probability_legitimate.is_(None) |
                     (Analysis.probability_predatory == Analysis.probability_legitimate))
                ).count() or 0
                
                predatory_count += fallback_predatory
                non_predatory_count += fallback_legitimate
        
        # Get recent data
        recent_analyses = Analysis.query.order_by(Analysis.analysis_timestamp.desc()).limit(5).all() or []
        recent_users = User.query.order_by(User.date_created.desc()).limit(5).all() or []
        
        # Login attempts statistics
        today = datetime.utcnow().date()
        login_attempts_today = LoginAttempt.query.filter(
            LoginAttempt.attempt_time >= today
        ).count() or 0
        
        failed_logins_today = LoginAttempt.query.filter(
            LoginAttempt.attempt_time >= today,
            LoginAttempt.success == False
        ).count() or 0
        
        # Calculate additional statistics for template
        average_confidence = 0
        high_confidence_count = 0
        avg_processing_time = 0
        total_processing_time = 0
        
        if total_analyses > 0:
            # Calculate average confidence (placeholder - would need confidence data)
            high_confidence_analyses = Analysis.query.filter(
                Analysis.confidence == 'High'
            ).count() if hasattr(Analysis, 'confidence') else 0
            high_confidence_count = high_confidence_analyses
            
            if high_confidence_analyses > 0:
                average_confidence = round((high_confidence_analyses / total_analyses) * 100, 1)
        
        stats = {
            'total_users': total_users,
            'verified_users': verified_users,
            'unverified_users': unverified_users,
            'total_analyses': total_analyses,
            'predatory_count': predatory_count,
            'non_predatory_count': non_predatory_count,
            'recent_analyses': recent_analyses,
            'recent_users': recent_users,
            'login_attempts_today': login_attempts_today,
            'failed_logins_today': failed_logins_today,
            'average_confidence': average_confidence,
            'high_confidence_count': high_confidence_count,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': total_processing_time,
            'recent_activity_count': login_attempts_today  # Placeholder
        }
        
    except Exception as e:
        # Error handling - provide default values
        print(f"Error calculating admin statistics: {e}")
        stats = {
            'total_users': 0,
            'verified_users': 0,
            'unverified_users': 0,
            'total_analyses': 0,
            'predatory_count': 0,
            'non_predatory_count': 0,
            'recent_analyses': [],
            'recent_users': [],
            'login_attempts_today': 0,
            'failed_logins_today': 0,
            'average_confidence': 0,
            'high_confidence_count': 0,
            'avg_processing_time': 0,
            'total_processing_time': 0,
            'recent_activity_count': 0
        }
    
    return render_template('admin/index.html', user=current_user, stats=stats)

@admin_bp.route('/users')
@admin_required
def users():
    """User management page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users = User.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/users.html', user=current_user, users=users)

@admin_bp.route('/analyses')
@admin_required
def analyses():
    """All analyses overview page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    analyses = Analysis.query.order_by(Analysis.analysis_timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/analyses.html', user=current_user, analyses=analyses)

@admin_bp.route('/logs')
@admin_required
def logs():
    """System logs and login attempts"""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    login_attempts = LoginAttempt.query.order_by(LoginAttempt.attempt_time.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/logs.html', user=current_user, login_attempts=login_attempts)

@admin_bp.route('/system')
@admin_required
def system():
    """System information and settings"""
    return render_template('admin/system.html', user=current_user)

@admin_bp.route('/verify-user/<int:user_id>', methods=['POST'])
@admin_required
def verify_user(user_id):
    """Verify/unverify a user"""
    try:
        user_to_modify = User.query.get(user_id)
        
        if not user_to_modify:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        
        # Toggle verification status
        user_to_modify.is_verified = not user_to_modify.is_verified
        db.session.commit()
        
        status = 'verified' if user_to_modify.is_verified else 'unverified'
        return jsonify({
            'success': True,
            'message': f'User {user_to_modify.email} has been {status}',
            'is_verified': user_to_modify.is_verified
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error updating user: {str(e)}'
        }), 500

@admin_bp.route('/toggle-user-status/<int:user_id>', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Activate/deactivate a user"""
    try:
        user_to_modify = User.query.get(user_id)
        
        if not user_to_modify:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        
        # Prevent admin from deactivating themselves
        if user_to_modify.id == current_user.id:
            return jsonify({
                'success': False,
                'message': 'You cannot deactivate your own account'
            }), 400
        
        # Toggle active status
        user_to_modify.is_active = not user_to_modify.is_active
        db.session.commit()
        
        status = 'activated' if user_to_modify.is_active else 'deactivated'
        return jsonify({
            'success': True,
            'message': f'User {user_to_modify.email} has been {status}',
            'is_active': user_to_modify.is_active
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error updating user status: {str(e)}'
        }), 500

@admin_bp.route('/delete-user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete a user and their analyses"""
    try:
        user_to_delete = User.query.get(user_id)
        
        if not user_to_delete:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        
        # Prevent admin from deleting themselves
        if user_to_delete.id == current_user.id:
            return jsonify({
                'success': False,
                'message': 'You cannot delete your own account'
            }), 400
        
        email = user_to_delete.email
        db.session.delete(user_to_delete)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'User {email} and all associated data have been deleted'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error deleting user: {str(e)}'
        }), 500

@admin_bp.route('/delete-analysis/<int:analysis_id>', methods=['DELETE'])
@admin_required
def delete_analysis(analysis_id):
    """Delete any analysis (admin can delete any user's analysis)"""
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({
                'success': False,
                'message': 'Analysis not found'
            }), 404
        
        filename = analysis.filename
        db.session.delete(analysis)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Analysis "{filename}" has been deleted'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error deleting analysis: {str(e)}'
        }), 500

@admin_bp.route('/export/users')
@admin_required
def export_users():
    """Export users data as CSV"""
    try:
        from flask import make_response
        import csv
        import io
        
        # Get all users data
        users = User.query.order_by(User.date_created.desc()).all()
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'ID', 'Full Name', 'Email', 'Date Created', 'Is Verified', 
            'Is Active', 'Total Analyses', 'Last Login'
        ])
        
        # Write user data
        for user in users:
            writer.writerow([
                user.id,
                user.full_name,
                user.email,
                user.date_created.strftime('%Y-%m-%d %H:%M:%S') if user.date_created else '',
                'Yes' if user.is_verified else 'No',
                'Yes' if user.is_active else 'No',
                len(user.analyses) if user.analyses else 0,
                user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never'
            ])
        
        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=users_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
        
    except Exception as e:
        flash(f'Error exporting users: {str(e)}', 'error')
        return redirect(url_for('admin.users'))

@admin_bp.route('/export/analyses')
@admin_required
def export_analyses():
    """Export analyses data as CSV"""
    try:
        from flask import make_response
        import csv
        import io
        
        # Get all analyses data
        analyses = Analysis.query.order_by(Analysis.analysis_timestamp.desc()).all()
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'ID', 'User Email', 'Filename', 'Analysis Date', 'Result', 
            'Probability Legitimate', 'Probability Predatory', 'Confidence',
            'HDD', 'Lexical Density', 'Review Speed', 'MTLD', 'Word Count'
        ])
        
        # Write analysis data
        for analysis in analyses:
            writer.writerow([
                analysis.id,
                analysis.user.email if analysis.user else 'Unknown',
                analysis.filename,
                analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S') if analysis.analysis_timestamp else '',
                'Predatory' if analysis.result == 1 else 'Legitimate',
                analysis.probability_legitimate or '',
                analysis.probability_predatory or '',
                analysis.confidence or '',
                analysis.hdd or '',
                analysis.lexical_density or '',
                analysis.review_speed or '',
                analysis.mtld or '',
                analysis.word_count or ''
            ])
        
        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=analyses_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
        
    except Exception as e:
        flash(f'Error exporting analyses: {str(e)}', 'error')
        return redirect(url_for('admin.analyses'))