"""
Flask Web Application for Vehicle Recognition Database
======================================================
Modern web interface with persistent database connection
"""

from flask import Flask, render_template, request, jsonify, send_file
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime, timedelta
import pandas as pd
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production

# Database configuration with connection pooling
DB_CONFIG = {
    'host': 'localhost',
    'database': 'vehicle_recognition',
    'user': 'root',
    'password': '12345',
    'pool_name': 'mypool',
    'pool_size': 5,
    'pool_reset_session': True,
    'autocommit': True,
    'connect_timeout': 30,
    'use_pure': True
}

# Create connection pool
try:
    connection_pool = pooling.MySQLConnectionPool(**DB_CONFIG)
    print("✓ Database connection pool created")
except Error as e:
    print(f"✗ Error creating connection pool: {e}")
    connection_pool = None


class DatabaseConnection:
    """Database connection manager using connection pooling"""
    
    @staticmethod
    def get_connection():
        """Get connection from pool"""
        try:
            if connection_pool:
                conn = connection_pool.get_connection()
                if conn.is_connected():
                    # Ping to ensure connection is alive
                    conn.ping(reconnect=True, attempts=3, delay=1)
                    return conn
            return None
        except Error as e:
            print(f"✗ Database connection error: {e}")
            return None
    
    @staticmethod
    def execute_query(query, params=None, fetch_one=False):
        """Execute query and return results with automatic connection management"""
        conn = None
        cursor = None
        try:
            conn = DatabaseConnection.get_connection()
            if not conn:
                return None
            
            cursor = conn.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch_one:
                results = cursor.fetchone()
            else:
                results = cursor.fetchall()
            
            return results
            
        except Error as e:
            print(f"✗ Query error: {e}")
            return None
            
        finally:
            # Always close cursor and return connection to pool
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()  # This returns connection to pool, doesn't close it
    
    @staticmethod
    def execute_update(query, params=None):
        """Execute update/delete query and return affected rows"""
        conn = None
        cursor = None
        try:
            conn = DatabaseConnection.get_connection()
            if not conn:
                return 0
            
            cursor = conn.cursor()
            cursor.execute(query, params if params else ())
            affected_rows = cursor.rowcount
            
            return affected_rows
            
        except Error as e:
            print(f"✗ Update error: {e}")
            return 0
            
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()


db = DatabaseConnection()


# ============================================
# API ROUTES
# ============================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/statistics/total')
def get_total_statistics():
    """Get overall statistics"""
    query = """
    SELECT 
        COUNT(*) as total_detections,
        COUNT(DISTINCT vehicle_number) as unique_vehicles,
        COUNT(DISTINCT state_code) as states_detected,
        ROUND(AVG(quality_score), 2) as avg_quality_score,
        MIN(timestamp) as first_detection,
        MAX(timestamp) as last_detection
    FROM detected_vehicles
    """
    
    result = db.execute_query(query, fetch_one=True)
    
    if result:
        # Convert datetime to string
        if result['first_detection']:
            result['first_detection'] = result['first_detection'].strftime('%Y-%m-%d %H:%M:%S')
        if result['last_detection']:
            result['last_detection'] = result['last_detection'].strftime('%Y-%m-%d %H:%M:%S')
        # Ensure avg_quality_score is a float
        if result['avg_quality_score']:
            result['avg_quality_score'] = float(result['avg_quality_score'])
        
        return jsonify({'success': True, 'data': result})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/recent-detections')
def get_recent_detections():
    """Get recent detections"""
    limit = request.args.get('limit', 10, type=int)
    
    query = """
    SELECT id, timestamp, vehicle_number, vehicle_type, 
           quality_score, state_name, detection_type,
           plate_confidence, vehicle_confidence
    FROM detected_vehicles 
    ORDER BY timestamp DESC 
    LIMIT %s
    """
    
    results = db.execute_query(query, (limit,))
    
    if results:
        # Convert datetime to string for JSON serialization
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            # Convert Decimal to float
            if 'quality_score' in row and row['quality_score']:
                row['quality_score'] = float(row['quality_score'])
            if 'plate_confidence' in row and row['plate_confidence']:
                row['plate_confidence'] = float(row['plate_confidence'])
            if 'vehicle_confidence' in row and row['vehicle_confidence']:
                row['vehicle_confidence'] = float(row['vehicle_confidence'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/detections-by-timeframe')
def get_detections_by_timeframe():
    """Get detections from last N hours"""
    hours = request.args.get('hours', 1, type=int)
    
    query = """
    SELECT id, timestamp, vehicle_number, vehicle_type, 
           quality_score, state_name, detection_type
    FROM detected_vehicles 
    WHERE timestamp > NOW() - INTERVAL %s HOUR 
    ORDER BY timestamp DESC
    """
    
    results = db.execute_query(query, (hours,))
    
    if results:
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if 'quality_score' in row and row['quality_score']:
                row['quality_score'] = float(row['quality_score'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/detections-by-date')
def get_detections_by_date():
    """Get detections for specific date"""
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    query = """
    SELECT id, timestamp, vehicle_number, vehicle_type, 
           quality_score, state_name, detection_type
    FROM detected_vehicles 
    WHERE DATE(timestamp) = %s 
    ORDER BY timestamp DESC
    """
    
    results = db.execute_query(query, (date_str,))
    
    if results:
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if 'quality_score' in row and row['quality_score']:
                row['quality_score'] = float(row['quality_score'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/statistics/vehicle-type')
def count_by_vehicle_type():
    """Count detections by vehicle type"""
    query = """
    SELECT vehicle_type, COUNT(*) as count 
    FROM detected_vehicles 
    GROUP BY vehicle_type 
    ORDER BY count DESC
    """
    
    results = db.execute_query(query)
    
    if results:
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/statistics/state')
def count_by_state():
    """Count detections by state"""
    query = """
    SELECT state_code, state_name, COUNT(*) as count 
    FROM detected_vehicles 
    WHERE state_code IS NOT NULL 
    GROUP BY state_code, state_name 
    ORDER BY count DESC
    """
    
    results = db.execute_query(query)
    
    if results:
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/statistics/detection-type')
def count_by_detection_type():
    """Count detections by detection type"""
    query = """
    SELECT detection_type, COUNT(*) as count 
    FROM detected_vehicles 
    GROUP BY detection_type 
    ORDER BY count DESC
    """
    
    results = db.execute_query(query)
    
    if results:
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/statistics/hourly')
def get_hourly_statistics():
    """Get hourly statistics for a date"""
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    query = """
    SELECT 
        HOUR(timestamp) as hour,
        COUNT(*) as total_detections,
        COUNT(DISTINCT vehicle_number) as unique_vehicles,
        ROUND(AVG(quality_score), 2) as avg_quality
    FROM detected_vehicles 
    WHERE DATE(timestamp) = %s 
    GROUP BY HOUR(timestamp) 
    ORDER BY hour
    """
    
    results = db.execute_query(query, (date_str,))
    
    if results:
        for row in results:
            if row['avg_quality']:
                row['avg_quality'] = float(row['avg_quality'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/search')
def search_vehicle():
    """Search for vehicle number"""
    plate_number = request.args.get('plate', '')
    
    if not plate_number:
        return jsonify({'success': False, 'message': 'Please provide a plate number'})
    
    query = """
    SELECT id, timestamp, vehicle_number, vehicle_type, 
           quality_score, state_name, detection_type
    FROM detected_vehicles 
    WHERE vehicle_number LIKE %s 
    ORDER BY timestamp DESC
    """
    
    results = db.execute_query(query, (f"%{plate_number}%",))
    
    if results:
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if 'quality_score' in row and row['quality_score']:
                row['quality_score'] = float(row['quality_score'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No matching vehicles found'})


@app.route('/api/high-quality')
def get_high_quality_detections():
    """Get high-quality detections"""
    threshold = request.args.get('threshold', 0.75, type=float)
    
    query = """
    SELECT id, timestamp, vehicle_number, vehicle_type, 
           quality_score, state_name, detection_type
    FROM detected_vehicles 
    WHERE quality_score >= %s 
    ORDER BY quality_score DESC, timestamp DESC
    LIMIT 100
    """
    
    results = db.execute_query(query, (threshold,))
    
    if results:
        for row in results:
            if 'timestamp' in row and row['timestamp']:
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if 'quality_score' in row and row['quality_score']:
                row['quality_score'] = float(row['quality_score'])
        return jsonify({'success': True, 'data': results})
    else:
        return jsonify({'success': False, 'message': 'No data found'})


@app.route('/api/export/csv')
def export_to_csv():
    """Export data to CSV"""
    try:
        conn = db.get_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        query = "SELECT * FROM detected_vehicles ORDER BY timestamp DESC"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Create exports directory if it doesn't exist
        os.makedirs('exports', exist_ok=True)
        
        filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('exports', filename)
        
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/delete-old')
def delete_old_records():
    """Delete old records"""
    days = request.args.get('days', 30, type=int)
    confirm = request.args.get('confirm', 'false')
    
    if confirm.lower() != 'true':
        return jsonify({'success': False, 'message': 'Confirmation required'})
    
    query = """
    DELETE FROM detected_vehicles 
    WHERE timestamp < NOW() - INTERVAL %s DAY
    """
    
    deleted = db.execute_update(query, (days,))
    
    return jsonify({
        'success': True, 
        'message': f'Deleted {deleted} records older than {days} days',
        'deleted_count': deleted
    })


if __name__ == '__main__':
    print("="*60)
    print("Vehicle Detection Database - Flask Web Application")
    print("="*60)
    print("Starting server with CONNECTION POOLING...")
    print("Access the dashboard at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)