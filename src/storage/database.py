"""
Database Module - MongoDB Atlas Implementation
Handles MongoDB Atlas operations for detection logging.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase


class Database:
    """
    MongoDB Atlas database manager for detection logging.
    Provides cloud-based storage for detections and alerts with offline caching.
    """
    
    def __init__(
        self,
        connection_string: str,
        database_name: str = "object_detection",
        collections: Optional[Dict[str, str]] = None,
        options: Optional[Dict[str, Any]] = None,
        retention_days: int = 30,
        cache_enabled: bool = True,
        cache_max_size_mb: int = 100
    ):
        """
        Initialize MongoDB Atlas database connection.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Database name
            collections: Dictionary mapping collection types to names
            options: MongoDB connection options
            retention_days: Data retention period in days
            cache_enabled: Enable local caching for offline support
            cache_max_size_mb: Maximum cache size in MB
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collections_config = collections or {
            'detections': 'detections',
            'alerts': 'alerts'
        }
        self.options = options or {}
        self.retention_days = retention_days
        self.cache_enabled = cache_enabled
        self.cache_max_size_mb = cache_max_size_mb
        
        # MongoDB connections
        self._client: Optional[MongoClient] = None
        self._db: Optional[MongoDatabase] = None
        
        # Collections
        self._detections_collection: Optional[Collection] = None
        self._alerts_collection: Optional[Collection] = None
        
        # Local cache for offline support
        self._cache: List[Dict[str, Any]] = []
        
        # Initialize connection
        self._connect()
    
    def _connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Returns:
            True if connected successfully
        """
        try:
            # Build connection with options
            self._client = MongoClient(
                self.connection_string,
                **self.options
            )
            
            # Test connection
            self._client.admin.command('ping')
            
            # Get database
            self._db = self._client[self.database_name]
            
            # Get collections
            self._detections_collection = self._db[self.collections_config['detections']]
            self._alerts_collection = self._db[self.collections_config['alerts']]
            
            # Create indexes
            self._create_indexes()
            
            # Flush any cached data
            if self._cache:
                self._flush_cache()
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection failed: {e}")
            self._client = None
            self._db = None
            return False
    
    def _create_indexes(self) -> None:
        """Create database indexes for optimal query performance."""
        if self._detections_collection is None:
            return
        
        # Detections indexes
        self._detections_collection.create_index([("timestamp", DESCENDING)])
        self._detections_collection.create_index([("object_type", ASCENDING)])
        self._detections_collection.create_index([("timestamp", DESCENDING), ("object_type", ASCENDING)])
        
        # Alerts indexes
        if self._alerts_collection is not None:
            self._alerts_collection.create_index([("sent_at", DESCENDING)])
            self._alerts_collection.create_index([("object_type", ASCENDING)])
    
    def _flush_cache(self) -> None:
        """Flush cached data to MongoDB when connection is restored."""
        if not self._cache or self._detections_collection is None:
            return
        
        try:
            self._detections_collection.insert_many(self._cache)
            self._cache.clear()
        except Exception as e:
            print(f"Failed to flush cache: {e}")
    
    def _is_connected(self) -> bool:
        """Check if database is connected."""
        if self._client is None:
            return False
        try:
            self._client.admin.command('ping')
            return True
        except:
            return False
    
    def log_detection(
        self,
        object_type: str,
        confidence: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_number: Optional[int] = None,
        snapshot_path: Optional[str] = None,
        roi_zone: Optional[str] = None,
        notified: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log a detection to MongoDB.
        
        Args:
            object_type: Type of detected object
            confidence: Detection confidence
            bbox: Bounding box coordinates
            frame_number: Frame number
            snapshot_path: Path to snapshot image
            roi_zone: ROI zone name
            notified: Whether notification was sent
            metadata: Additional metadata
            
        Returns:
            Detection ID as string
        """
        document = {
            "object_type": object_type,
            "confidence": confidence,
            "bbox": list(bbox) if bbox else None,
            "frame_number": frame_number,
            "snapshot_path": snapshot_path,
            "roi_zone": roi_zone,
            "notified": notified,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        
        if self._is_connected() and self._detections_collection is not None:
            try:
                result = self._detections_collection.insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                print(f"Failed to insert detection: {e}")
        
        # Cache if not connected and caching is enabled
        if self.cache_enabled:
            self._cache.append(document)
            # Check cache size limit
            self._check_cache_size()
        
        return None
    
    def log_alert(
        self,
        detection_id: Optional[str],
        object_type: str,
        channel: str,
        recipient: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[str]:
        """
        Log an alert to MongoDB.
        
        Args:
            detection_id: Related detection ID
            object_type: Type of object
            channel: Notification channel (email, sms, etc.)
            recipient: Recipient address/number
            status: Alert status (sent, failed)
            error_message: Error message if failed
            
        Returns:
            Alert ID as string
        """
        document = {
            "detection_id": detection_id,
            "object_type": object_type,
            "channel": channel,
            "recipient": recipient,
            "status": status,
            "error_message": error_message,
            "sent_at": datetime.utcnow()
        }
        
        if self._is_connected() and self._alerts_collection is not None:
            try:
                result = self._alerts_collection.insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                print(f"Failed to insert alert: {e}")
        
        return None
    
    def mark_detection_notified(self, detection_id: str) -> bool:
        """
        Mark a detection as notified.
        
        Args:
            detection_id: Detection ID
            
        Returns:
            True if updated successfully
        """
        if not self._is_connected() or self._detections_collection is None:
            return False
        
        from bson import ObjectId
        
        try:
            result = self._detections_collection.update_one(
                {"_id": ObjectId(detection_id)},
                {"$set": {"notified": True}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Failed to update detection: {e}")
            return False
    
    def get_recent_detections(
        self,
        limit: int = 100,
        object_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent detections from MongoDB.
        
        Args:
            limit: Maximum number of results
            object_type: Filter by object type (optional)
            
        Returns:
            List of detection dictionaries
        """
        if not self._is_connected() or self._detections_collection is None:
            return []
        
        query = {}
        if object_type:
            query["object_type"] = object_type
        
        try:
            cursor = self._detections_collection.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            return [self._document_to_dict(doc) for doc in cursor]
        except Exception as e:
            print(f"Failed to get detections: {e}")
            return []
    
    def get_detection_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get detection statistics from MongoDB.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Statistics dictionary
        """
        if not self._is_connected() or self._detections_collection is None:
            return {
                'total_detections': 0,
                'object_counts': {},
                'average_confidence': 0,
                'alerts_today': 0
            }
        
        # Build date filter
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        
        query = {}
        if date_filter:
            query["timestamp"] = date_filter
        
        try:
            # Total count
            total_count = self._detections_collection.count_documents(query)
            
            # Object type counts using aggregation
            pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$object_type",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            
            object_counts = {}
            for doc in self._detections_collection.aggregate(pipeline):
                object_counts[doc["_id"]] = doc["count"]
            
            # Average confidence
            avg_pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": None,
                    "avg_confidence": {"$avg": "$confidence"}
                }}
            ]
            
            avg_result = list(self._detections_collection.aggregate(avg_pipeline))
            avg_confidence = avg_result[0]["avg_confidence"] if avg_result else 0
            
            # Alerts today
            today_start = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            
            alerts_today = 0
            if self._alerts_collection is not None:
                alerts_today = self._alerts_collection.count_documents({
                    "sent_at": {"$gte": today_start}
                })
            
            return {
                'total_detections': total_count,
                'object_counts': object_counts,
                'average_confidence': avg_confidence,
                'alerts_today': alerts_today
            }
            
        except Exception as e:
            print(f"Failed to get stats: {e}")
            return {
                'total_detections': 0,
                'object_counts': {},
                'average_confidence': 0,
                'alerts_today': 0
            }
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary for a specific date.
        
        Args:
            date: Date to summarize (defaults to today)
            
        Returns:
            Summary dictionary
        """
        if date is None:
            date = datetime.utcnow()
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        return self.get_detection_stats(start, end)
    
    def cleanup_old_records(self, retention_days: Optional[int] = None) -> int:
        """
        Remove records older than retention period.
        
        Args:
            retention_days: Number of days to keep (uses config if not provided)
            
        Returns:
            Number of deleted records
        """
        if not self._is_connected():
            return 0
        
        retention = retention_days or self.retention_days
        cutoff = datetime.utcnow() - timedelta(days=retention)
        
        deleted = 0
        
        try:
            # Delete old alerts
            if self._alerts_collection is not None:
                alerts_result = self._alerts_collection.delete_many({
                    "sent_at": {"$lt": cutoff}
                })
                deleted += alerts_result.deleted_count
                        
            # Delete old detections
            if self._detections_collection is not None:
                detections_result = self._detections_collection.delete_many({
                    "timestamp": {"$lt": cutoff}
                })
                deleted += detections_result.deleted_count
            
        except Exception as e:
            print(f"Failed to cleanup records: {e}")
        
        return deleted
    
    def _check_cache_size(self) -> None:
        """Check and manage cache size."""
        if not self._cache:
            return
        
        import sys
        
        # Estimate cache size
        cache_size = sum(sys.getsizeof(doc) for doc in self._cache) / (1024 * 1024)
        
        if cache_size > self.cache_max_size_mb:
            # Remove oldest entries
            while cache_size > self.cache_max_size_mb * 0.8 and self._cache:
                self._cache.pop(0)
                cache_size = sum(sys.getsizeof(doc) for doc in self._cache) / (1024 * 1024)
    
    def _document_to_dict(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MongoDB document to dictionary.
        
        Args:
            doc: MongoDB document
            
        Returns:
            Dictionary representation
        """
        result = dict(doc)
        
        # Convert ObjectId to string
        if "_id" in result:
            result["_id"] = str(result["_id"])
        
        # Convert datetime to ISO format
        if "timestamp" in result and isinstance(result["timestamp"], datetime):
            result["timestamp"] = result["timestamp"].isoformat()
        if "sent_at" in result and isinstance(result["sent_at"], datetime):
            result["sent_at"] = result["sent_at"].isoformat()
        
        return result
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test MongoDB Atlas connection.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            if self._client is None:
                return False, "MongoDB client not initialized"
            
            self._client.admin.command('ping')
            return True, "MongoDB Atlas connection successful"
            
        except ConnectionFailure:
            return False, "Failed to connect to MongoDB Atlas"
        except ServerSelectionTimeoutError:
            return False, "MongoDB Atlas server selection timeout"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to MongoDB Atlas.
        
        Returns:
            True if reconnected successfully
        """
        if self._client:
            try:
                self._client.close()
            except:
                pass
        
        return self._connect()
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            try:
                self._client.close()
            except:
                pass
        
        self._client = None
        self._db = None
        self._detections_collection = None
        self._alerts_collection = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor."""
        self.close()
    
    # ==================== Export Functions ====================
    
    def export_detections_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000
    ) -> str:
        """
        Export detections to CSV format.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            CSV string
        """
        import csv
        import io
        
        if not self._is_connected():
            return ""
        
        # Build query
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        
        query = {}
        if date_filter:
            query["timestamp"] = date_filter
        
        try:
            cursor = self._detections_collection.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                'ID', 'Timestamp', 'Object Type', 'Confidence',
                'BBox', 'Frame Number', 'Snapshot Path', 'ROI Zone',
                'Notified', 'Suspicious', 'Severity', 'Crowd Count'
            ])
            
            for doc in cursor:
                metadata = doc.get('metadata', {})
                writer.writerow([
                    str(doc.get('_id', '')),
                    doc.get('timestamp', '').isoformat() if isinstance(doc.get('timestamp'), datetime) else str(doc.get('timestamp', '')),
                    doc.get('object_type', ''),
                    f"{(doc.get('confidence', 0) * 100):.1f}%",
                    str(doc.get('bbox', '')),
                    doc.get('frame_number', ''),
                    doc.get('snapshot_path', ''),
                    doc.get('roi_zone', ''),
                    doc.get('notified', False),
                    metadata.get('suspicious', False),
                    metadata.get('severity', ''),
                    metadata.get('crowd_count', '')
                ])
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Failed to export detections: {e}")
            return ""
    
    def export_detections_json(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Export detections to JSON format.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            List of detection dictionaries
        """
        if not self._is_connected():
            return []
        
        # Build query
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        
        query = {}
        if date_filter:
            query["timestamp"] = date_filter
        
        try:
            cursor = self._detections_collection.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            results = []
            for doc in cursor:
                result = self._document_to_dict(doc)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Failed to export detections: {e}")
            return []
    
    def get_detection_images(
        self,
        suspicious_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get detection records with images (base64 encoded).
        
        Args:
            suspicious_only: Only get suspicious detections
            limit: Maximum number of records
            
        Returns:
            List of detection records with images
        """
        if not self._is_connected():
            return []
        
        query = {}
        if suspicious_only:
            query["metadata.suspicious"] = True
        
        # Only get records with images
        query["metadata.image_base64"] = {"$exists": True, "$ne": None}
        
        try:
            cursor = self._detections_collection.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            results = []
            for doc in cursor:
                result = self._document_to_dict(doc)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Failed to get detection images: {e}")
            return []
    
    def get_threat_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get threat detection summary.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Summary dictionary
        """
        if not self._is_connected():
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
        
        # Build query for suspicious detections
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        
        query = {"metadata.suspicious": True}
        if date_filter:
            query["timestamp"] = date_filter
        
        try:
            # Total count
            total = self._detections_collection.count_documents(query)
            
            # By severity
            severity_pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$metadata.severity",
                    "count": {"$sum": 1}
                }}
            ]
            
            by_severity = {}
            for doc in self._detections_collection.aggregate(severity_pipeline):
                severity = doc.get('_id', 'unknown') or 'unknown'
                by_severity[severity] = doc.get('count', 0)
            
            # By type
            type_pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$object_type",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            by_type = {}
            for doc in self._detections_collection.aggregate(type_pipeline):
                obj_type = doc.get('_id', 'unknown') or 'unknown'
                by_type[obj_type] = doc.get('count', 0)
            
            return {
                'total': total,
                'by_severity': by_severity,
                'by_type': by_type
            }
            
        except Exception as e:
            print(f"Failed to get threat summary: {e}")
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
    
    def get_hourly_activity(
        self,
        date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get hourly detection activity for a specific date.
        
        Args:
            date: Date to analyze (defaults to today)
            
        Returns:
            List of hourly activity data
        """
        if not self._is_connected():
            return []
        
        if date is None:
            date = datetime.utcnow()
        
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        try:
            pipeline = [
                {"$match": {
                    "timestamp": {"$gte": start, "$lt": end}
                }},
                {"$group": {
                    "_id": {"$hour": "$timestamp"},
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            results = []
            for doc in self._detections_collection.aggregate(pipeline):
                results.append({
                    'hour': doc.get('_id', 0),
                    'count': doc.get('count', 0)
                })
            
            return results
            
        except Exception as e:
            print(f"Failed to get hourly activity: {e}")
            return []
