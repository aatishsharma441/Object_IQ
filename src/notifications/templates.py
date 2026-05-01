"""
Email Templates Module
Provides HTML email templates for notification alerts.
"""

from datetime import datetime
from typing import Any, Dict, Optional


class EmailTemplate:
    """
    Generates HTML email content for detection alerts.
    """
    
    @staticmethod
    def alert_email(
        object_type: str,
        confidence: float,
        timestamp: datetime,
        roi_zone: Optional[str] = None,
        consecutive_frames: int = 1,
        snapshot_path: Optional[str] = None
    ) -> tuple:
        """
        Generate alert email content.
        
        Args:
            object_type: Detected object type
            confidence: Detection confidence
            timestamp: Detection timestamp
            roi_zone: ROI zone name (optional)
            consecutive_frames: Number of consecutive frames
            snapshot_path: Path to snapshot image (optional)
            
        Returns:
            Tuple of (subject, html_body, text_body)
        """
        subject = f"[Alert] {object_type.capitalize()} Detected - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Build HTML body
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .info-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        .info-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 40%;
        }}
        .info-table td:last-child {{
            color: #333;
        }}
        .confidence {{
            display: inline-block;
            background-color: #27ae60;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .footer {{
            background-color: #f9f9f9;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #888;
        }}
        .snapshot {{
            text-align: center;
            margin: 20px 0;
        }}
        .snapshot img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Object Detection Alert</h1>
            <p>{object_type.capitalize()} has been detected</p>
        </div>
        <div class="content">
            <table class="info-table">
                <tr>
                    <td>Object Type</td>
                    <td><strong>{object_type.capitalize()}</strong></td>
                </tr>
                <tr>
                    <td>Confidence</td>
                    <td><span class="confidence">{confidence:.1%}</span></td>
                </tr>
                <tr>
                    <td>Detection Time</td>
                    <td>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td>Consecutive Frames</td>
                    <td>{consecutive_frames}</td>
                </tr>
                {f'<tr><td>Zone</td><td>{roi_zone}</td></tr>' if roi_zone else ''}
            </table>
        </div>
        <div class="footer">
            <p>This is an automated alert from the Object Detection System.</p>
            <p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Build text body
        text_body = f"""
OBJECT DETECTION ALERT
======================

Object Detected: {object_type.capitalize()}
Confidence: {confidence:.1%}
Detection Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Consecutive Frames: {consecutive_frames}
{f'Zone: {roi_zone}' if roi_zone else ''}

This is an automated alert from the Object Detection System.
Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return subject, html_body, text_body
    
    @staticmethod
    def system_notification(
        title: str,
        message: str,
        level: str = "info"
    ) -> tuple:
        """
        Generate system notification email.
        
        Args:
            title: Notification title
            message: Notification message
            level: Alert level (info, warning, error)
            
        Returns:
            Tuple of (subject, html_body, text_body)
        """
        colors = {
            'info': '#3498db',
            'warning': '#f39c12',
            'error': '#e74c3c'
        }
        color = colors.get(level, '#3498db')
        
        subject = f"[Object Detection] {title}"
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background-color: {color};
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
        }}
        .content {{
            padding: 30px;
            line-height: 1.6;
        }}
        .footer {{
            background-color: #f9f9f9;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>
        <div class="content">
            <p>{message}</p>
        </div>
        <div class="footer">
            <p>Object Detection System Notification</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_body = f"""
{title}
{'=' * len(title)}

{message}

Object Detection System
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return subject, html_body, text_body
    
    @staticmethod
    def daily_summary(
        total_detections: int,
        object_counts: Dict[str, int],
        alerts_sent: int,
        date: datetime
    ) -> tuple:
        """
        Generate daily summary email.
        
        Args:
            total_detections: Total detections for the day
            object_counts: Dictionary of object types and their counts
            alerts_sent: Number of alerts sent
            date: Date of summary
            
        Returns:
            Tuple of (subject, html_body, text_body)
        """
        subject = f"Daily Summary - {date.strftime('%Y-%m-%d')}"
        
        # Build object count rows
        object_rows = ""
        for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            object_rows += f"""
                <tr>
                    <td>{obj_type.capitalize()}</td>
                    <td>{count}</td>
                </tr>
"""
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .content {{
            padding: 30px;
        }}
        .stat-box {{
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }}
        .stat-box h3 {{
            margin: 0;
            color: #333;
        }}
        .stat-box p {{
            margin: 5px 0 0;
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        .object-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .object-table th {{
            background-color: #f9f9f9;
            padding: 10px;
            text-align: left;
        }}
        .object-table td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        .footer {{
            background-color: #f9f9f9;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Daily Detection Summary</h1>
            <p>{date.strftime('%B %d, %Y')}</p>
        </div>
        <div class="content">
            <div class="stat-box">
                <h3>Total Detections</h3>
                <p>{total_detections}</p>
            </div>
            <div class="stat-box">
                <h3>Alerts Sent</h3>
                <p>{alerts_sent}</p>
            </div>
            
            <h3>Detections by Object Type</h3>
            <table class="object-table">
                <tr>
                    <th>Object Type</th>
                    <th>Count</th>
                </tr>
                {object_rows}
            </table>
        </div>
        <div class="footer">
            <p>Object Detection System - Daily Summary</p>
        </div>
    </div>
</body>
</html>
"""
        
        text_body = f"""
DAILY DETECTION SUMMARY
{date.strftime('%B %d, %Y')}
========================

Total Detections: {total_detections}
Alerts Sent: {alerts_sent}

Detections by Object Type:
"""
        for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            text_body += f"  - {obj_type}: {count}\n"
        
        return subject, html_body, text_body
