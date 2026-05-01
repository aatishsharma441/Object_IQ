"""
Authentication Module
User authentication with Flask-Login for the dashboard.
Includes profile image support and user data management.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import base64
import os
# import imghdr

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# Allowed image extensions for profile pictures
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_PROFILE_SIZE = 5 * 1024 * 1024  # 5MB max


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class User(UserMixin):
    """
    User model for authentication with profile support.
    """
    
    def __init__(self, user_data: Dict[str, Any]):
        """
        Initialize user from database document.
        
        Args:
            user_data: User document from MongoDB
        """
        self.id = str(user_data.get('_id', ''))
        self.username = user_data.get('username', '')
        self.email = user_data.get('email', '')
        self.password_hash = user_data.get('password_hash', '')
        self.role = user_data.get('role', 'user')
        self.created_at = user_data.get('created_at', datetime.utcnow())
        self.last_login = user_data.get('last_login')
        self._is_active = user_data.get('is_active', True)
        
        # Profile fields
        self.profile_image = user_data.get('profile_image')  # Base64 encoded image
        self.profile_image_type = user_data.get('profile_image_type')  # image type
        self.first_name = user_data.get('first_name', '')
        self.last_name = user_data.get('last_name', '')
        self.phone = user_data.get('phone', '')
        self.organization = user_data.get('organization', '')
        self.bio = user_data.get('bio', '')
        self.preferences = user_data.get('preferences', {})
        self.notification_settings = user_data.get('notification_settings', {
            'email_alerts': True,
            'threat_alerts': True,
            'crowd_alerts': False,
            'daily_summary': True
        })
    
    @property
    def is_active(self):
        """Override UserMixin is_active property."""
        return self._is_active
    
    
    @property
    def display_name(self) -> str:
        """Get user's display name."""
        if self.first_name or self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        return self.username
    
    
    @property
    def profile_image_url(self) -> Optional[str]:
        """Get profile image as data URL."""
        if self.profile_image and self.profile_image_type:
            return f"data:{self.profile_image_type};base64,{self.profile_image}"
        return None
    
    
    def verify_password(self, password: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password: Plain text password
            
        Returns:
            True if password matches
        """
        return check_password_hash(self.password_hash, password)
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'is_active': self._is_active,
            'profile_image': self.profile_image,
            'profile_image_type': self.profile_image_type,
            'profile_image_url': self.profile_image_url,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'phone': self.phone,
            'organization': self.organization,
            'bio': self.bio,
            'display_name': self.display_name,
            'preferences': self.preferences,
            'notification_settings': self.notification_settings
        }


class UserManager:
    """
    Manages user authentication with MongoDB.
    """
    
    def __init__(self, db):
        """
        Initialize user manager.
        
        Args:
            db: Database instance
        """
        self.db = db
        self._users_collection = None
    
    @property
    def users(self):
        """Get users collection."""
        if self._users_collection is None and self.db._db is not None:
            self._users_collection = self.db._db['users']
        return self._users_collection
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = 'user'
    ) -> Tuple[bool, str]:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            role: User role (admin, user)
            
        Returns:
            Tuple of (success, message)
        """
        from typing import Tuple
        
        # Check if username exists
        if self.users is not None:
            if self.users.find_one({'username': username}):
                return False, 'Username already exists'
            
            if self.users.find_one({'email': email}):
                return False, 'Email already registered'
            
            # Create user document
            user_doc = {
                'username': username,
                'email': email,
                'password_hash': generate_password_hash(password),
                'role': role,
                'created_at': datetime.utcnow(),
                'last_login': None,
                'is_active': True
            }
            
            try:
                result = self.users.insert_one(user_doc)
                return True, str(result.inserted_id)
            except Exception as e:
                return False, f'Failed to create user: {str(e)}'
        
        return False, 'Database not connected'
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None
        """
        from bson import ObjectId
        
        if self.users is None:
            return None
        
        try:
            user_data = self.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        except:
            pass
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User object or None
        """
        if self.users is None:
            return None
        
        user_data = self.users.find_one({'username': username})
        if user_data:
            return User(user_data)
        
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            email: Email address
            
        Returns:
            User object or None
        """
        if self.users is None:
            return None
        
        user_data = self.users.find_one({'email': email})
        if user_data:
            return User(user_data)
        
        return None
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user by username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User object if authenticated, None otherwise
        """
        user = self.get_user_by_username(username)
        if user and user.verify_password(password):
            # Update last login
            if self.users is not None:
                self.users.update_one(
                    {'username': username},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
            return user
        
        return None
    
    def update_password(self, user_id: str, new_password: str) -> bool:
        """
        Update user password.
        
        Args:
            user_id: User ID
            new_password: New plain text password
            
        Returns:
            True if successful
        """
        from bson import ObjectId
        
        if self.users is None:
            return False
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'password_hash': generate_password_hash(new_password)}}
            )
            return result.modified_count > 0
        except:
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        from bson import ObjectId
        
        if self.users is None:
            return False
        
        try:
            result = self.users.delete_one({'_id': ObjectId(user_id)})
            return result.deleted_count > 0
        except:
            return False
    
    def get_all_users(self) -> list:
        """Get all users (for admin)."""
        if self.users is None:
            return []
        
        users = []
        for doc in self.users.find():
            users.append(User(doc).to_dict())
        return users
    
    def create_default_admin(self) -> None:
        """Create default admin user if no users exist."""
        if self.users is None:
            return
        
        if self.users.count_documents({}) == 0:
            self.create_user(
                username='admin',
                email='admin@localhost',
                password='admin123',
                role='admin'
            )
            print("Created default admin user: admin / admin123")
    
    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update user profile.
        
        Args:
            user_id: User ID
            updates: Dictionary of fields to update
            
        Returns:
            Tuple of (success, message)
        """
        from bson import ObjectId
        
        if self.users is None:
            return False, 'Database not connected'
        
        # Allowed fields to update
        allowed_fields = [
            'email', 'first_name', 'last_name', 'phone', 
            'organization', 'bio', 'preferences', 'notification_settings'
        ]
        
        filtered_updates = {
            k: v for k, v in updates.items() 
            if k in allowed_fields
        }
        
        if not filtered_updates:
            return False, 'No valid fields to update'
        
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': filtered_updates}
            )
            if result.modified_count > 0:
                return True, 'Profile updated successfully'
            return True, 'No changes made'
        except Exception as e:
            return False, f'Failed to update profile: {str(e)}'
    
    def update_profile_image(self, user_id: str, image_data: bytes, content_type: str) -> Tuple[bool, str]:
        """
        Update user profile image.
        
        Args:
            user_id: User ID
            image_data: Image binary data
            content_type: MIME type of image
            
        Returns:
            Tuple of (success, message)
        """
        from bson import ObjectId
        
        if self.users is None:
            return False, 'Database not connected'
        
        # Check size
        if len(image_data) > MAX_PROFILE_SIZE:
            return False, f'Image too large. Max size is {MAX_PROFILE_SIZE // (1024*1024)}MB'
        
        # Encode to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {
                    'profile_image': image_base64,
                    'profile_image_type': content_type
                }}
            )
            if result.modified_count > 0:
                return True, 'Profile image updated'
            return True, 'No changes made'
        except Exception as e:
            return False, f'Failed to update profile image: {str(e)}'
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change user password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        from bson import ObjectId
        
        if self.users is None:
            return False, 'Database not connected'
        
        # Get user and verify current password
        user = self.get_user(user_id)
        if not user:
            return False, 'User not found'
        
        if not user.verify_password(current_password):
            return False, 'Current password is incorrect'
        
        # Validate new password
        if len(new_password) < 6:
            return False, 'New password must be at least 6 characters'
        
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'password_hash': generate_password_hash(new_password)}}
            )
            if result.modified_count > 0:
                return True, 'Password changed successfully'
            return False, 'Failed to update password'
        except Exception as e:
            return False, f'Failed to change password: {str(e)}'
    
    def update_notification_settings(self, user_id: str, settings: Dict[str, bool]) -> Tuple[bool, str]:
        """
        Update user notification settings.
        
        Args:
            user_id: User ID
            settings: Notification settings dictionary
            
        Returns:
            Tuple of (success, message)
        """
        from bson import ObjectId
        
        if self.users is None:
            return False, 'Database not connected'
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'notification_settings': settings}}
            )
            return True, 'Notification settings updated'
        except Exception as e:
            return False, f'Failed to update settings: {str(e)}'
    
    def delete_profile_image(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete user profile image.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (success, message)
        """
        from bson import ObjectId
        
        if self.users is None:
            return False, 'Database not connected'
        
        try:
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$unset': {'profile_image': '', 'profile_image_type': ''}}
            )
            return True, 'Profile image deleted'
        except Exception as e:
            return False, f'Failed to delete profile image: {str(e)}'
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        from bson import ObjectId
        
        if self.users is None:
            return {}
        
        try:
            user_data = self.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return {
                    'total_logins': user_data.get('total_logins', 0),
                    'last_login': user_data.get('last_login'),
                    'created_at': user_data.get('created_at'),
                    'alerts_received': user_data.get('alerts_received', 0),
                    'reports_generated': user_data.get('reports_generated', 0)
                }
        except:
            pass
        
        return {}
