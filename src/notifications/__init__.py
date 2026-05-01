"""
Notification Module
Contains email notification services and templates.
"""

from .email_service import EmailService
from .templates import EmailTemplate

__all__ = ["EmailService", "EmailTemplate"]
