"""
Unit tests for Notification Module
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from notifications.templates import EmailTemplate
from notifications.email_service import EmailConfig, EmailService


class TestEmailTemplate:
    """Tests for EmailTemplate class."""
    
    def test_alert_email(self):
        """Test alert email generation."""
        subject, html, text = EmailTemplate.alert_email(
            object_type="person",
            confidence=0.85,
            timestamp=datetime.now(),
            roi_zone="entrance",
            consecutive_frames=3
        )
        
        assert "person" in subject.lower()
        assert "85%" in html or "85.0%" in html
        assert "entrance" in html
        assert "person" in text.lower()
    
    def test_system_notification(self):
        """Test system notification generation."""
        subject, html, text = EmailTemplate.system_notification(
            title="Test Notification",
            message="This is a test",
            level="info"
        )
        
        assert "Test Notification" in subject
        assert "This is a test" in html
        assert "This is a test" in text
    
    def test_daily_summary(self):
        """Test daily summary generation."""
        subject, html, text = EmailTemplate.daily_summary(
            total_detections=100,
            object_counts={"person": 50, "car": 50},
            alerts_sent=10,
            date=datetime.now()
        )
        
        assert "100" in html
        assert "person" in html.lower()
        assert "10" in html


class TestEmailService:
    """Tests for EmailService class."""
    
    def test_email_config_creation(self):
        """Test email config creation."""
        config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_password="password",
            recipient_email="recipient@test.com"
        )
        
        assert config.smtp_server == "smtp.test.com"
        assert config.smtp_port == 587
    
    def test_service_initialization(self):
        """Test email service initialization."""
        config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_password="password",
            recipient_email="recipient@test.com"
        )
        
        service = EmailService(config)
        
        assert service.config == config
        assert service.max_retries == 3
    
    def test_validate_config(self):
        """Test config validation."""
        # Complete config
        config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_password="password",
            recipient_email="recipient@test.com"
        )
        service = EmailService(config)
        assert service._validate_config() is True
        
        # Incomplete config
        incomplete_config = EmailConfig(
            smtp_server="",
            smtp_port=587,
            sender_email="",
            sender_password="",
            recipient_email=""
        )
        incomplete_service = EmailService(incomplete_config)
        assert incomplete_service._validate_config() is False
    
    def test_update_config(self):
        """Test config update."""
        config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_password="password",
            recipient_email="recipient@test.com"
        )
        
        service = EmailService(config)
        service.update_config(smtp_port=465)
        
        assert service.config.smtp_port == 465
    
    @patch('smtplib.SMTP')
    def test_test_connection_success(self, mock_smtp):
        """Test connection test success."""
        config = EmailConfig(
            smtp_server="smtp.test.com",
            smtp_port=587,
            sender_email="test@test.com",
            sender_password="password",
            recipient_email="recipient@test.com"
        )
        
        service = EmailService(config)
        success, message = service.test_connection()
        
        # Should attempt connection
        mock_smtp.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
