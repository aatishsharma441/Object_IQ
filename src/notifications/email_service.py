"""
Email Service Module
Handles sending email notifications via SMTP.
"""

import smtplib
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import List, Optional, Tuple

from .templates import EmailTemplate


@dataclass
class EmailConfig:
    """
    Email configuration settings.
    
    Attributes:
        smtp_server: SMTP server address
        smtp_port: SMTP port number
        sender_email: Sender email address
        sender_password: Sender email password/app password
        recipient_email: Recipient email address
        use_tls: Whether to use TLS encryption
    """
    smtp_server: str
    smtp_port: int
    sender_email: str
    sender_password: str
    recipient_email: str
    use_tls: bool = True


class EmailService:
    """
    Email notification service using SMTP.
    Supports HTML emails with optional image attachments.
    """
    
    def __init__(self, config: EmailConfig):
        """
        Initialize email service.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self._smtp_connection = None
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def send_alert(
        self,
        object_type: str,
        confidence: float,
        timestamp=None,
        roi_zone: Optional[str] = None,
        consecutive_frames: int = 1,
        snapshot_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Send a detection alert email.
        
        Args:
            object_type: Type of detected object
            confidence: Detection confidence
            timestamp: Detection timestamp (defaults to now)
            roi_zone: ROI zone name
            consecutive_frames: Number of consecutive frames
            snapshot_path: Path to snapshot image
            
        Returns:
            Tuple of (success, message)
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate email content
        subject, html_body, text_body = EmailTemplate.alert_email(
            object_type=object_type,
            confidence=confidence,
            timestamp=timestamp,
            roi_zone=roi_zone,
            consecutive_frames=consecutive_frames
        )
        
        return self._send_email(
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            attachment_path=snapshot_path
        )
    
    def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info"
    ) -> Tuple[bool, str]:
        """
        Send a system notification email.
        
        Args:
            title: Notification title
            message: Notification message
            level: Alert level (info, warning, error)
            
        Returns:
            Tuple of (success, message)
        """
        subject, html_body, text_body = EmailTemplate.system_notification(
            title=title,
            message=message,
            level=level
        )
        
        return self._send_email(subject, html_body, text_body)
    
    def send_daily_summary(
        self,
        total_detections: int,
        object_counts: dict,
        alerts_sent: int,
        date=None
    ) -> Tuple[bool, str]:
        """
        Send a daily summary email.
        
        Args:
            total_detections: Total detections for the day
            object_counts: Dictionary of object counts
            alerts_sent: Number of alerts sent
            date: Date of summary (defaults to today)
            
        Returns:
            Tuple of (success, message)
        """
        from datetime import datetime
        
        if date is None:
            date = datetime.now()
        
        subject, html_body, text_body = EmailTemplate.daily_summary(
            total_detections=total_detections,
            object_counts=object_counts,
            alerts_sent=alerts_sent,
            date=date
        )
        
        return self._send_email(subject, html_body, text_body)
    
    def _send_email(
        self,
        subject: str,
        html_body: str,
        text_body: str,
        attachment_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Send an email with optional attachment.
        
        Args:
            subject: Email subject
            html_body: HTML content
            text_body: Plain text content
            attachment_path: Path to attachment file
            
        Returns:
            Tuple of (success, message)
        """
        # Validate configuration
        if not self._validate_config():
            return False, "Email configuration is incomplete"
        
        # Create message
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = self.config.sender_email
        msg['To'] = self.config.recipient_email
        
        # Create alternative part for text and HTML
        alt_part = MIMEMultipart('alternative')
        alt_part.attach(MIMEText(text_body, 'plain'))
        alt_part.attach(MIMEText(html_body, 'html'))
        msg.attach(alt_part)
        
        # Attach image if provided
        if attachment_path:
            try:
                image_path = Path(attachment_path)
                if image_path.exists():
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                    
                    image = MIMEImage(image_data)
                    image.add_header('Content-ID', '<snapshot>')
                    image.add_header('Content-Disposition', 'attachment', filename=image_path.name)
                    msg.attach(image)
            except Exception as e:
                # Log but don't fail if attachment fails
                pass
        
        # Send with retries
        for attempt in range(self.max_retries):
            try:
                # Connect to SMTP server
                if self.config.use_tls:
                    server = smtplib.SMTP(
                        self.config.smtp_server,
                        self.config.smtp_port,
                        timeout=30
                    )
                    server.starttls()
                else:
                    server = smtplib.SMTP(
                        self.config.smtp_server,
                        self.config.smtp_port,
                        timeout=30
                    )
                
                # Login and send
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    [self.config.recipient_email],
                    msg.as_string()
                )
                server.quit()
                
                return True, "Email sent successfully"
                
            except smtplib.SMTPAuthenticationError:
                return False, "SMTP authentication failed. Check your email and password."
            except smtplib.SMTPException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False, f"SMTP error: {str(e)}"
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False, f"Failed to send email: {str(e)}"
        
        return False, "Max retries exceeded"
    
    def _validate_config(self) -> bool:
        """Validate email configuration is complete."""
        return all([
            self.config.smtp_server,
            self.config.smtp_port,
            self.config.sender_email,
            self.config.sender_password,
            self.config.recipient_email
        ])
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test SMTP connection without sending email.
        
        Returns:
            Tuple of (success, message)
        """
        if not self._validate_config():
            return False, "Email configuration is incomplete"
        
        try:
            server = smtplib.SMTP(
                self.config.smtp_server,
                self.config.smtp_port,
                timeout=10
            )
            
            if self.config.use_tls:
                server.starttls()
            
            server.login(self.config.sender_email, self.config.sender_password)
            server.quit()
            
            return True, "SMTP connection successful"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Authentication failed. Check your credentials."
        except smtplib.SMTPConnectError:
            return False, "Could not connect to SMTP server."
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def update_config(self, **kwargs) -> None:
        """
        Update email configuration.
        
        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
