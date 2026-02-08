"""
Database Query Helper for Vehicle Recognition System - PERSISTENT CONNECTION
============================================================================
Provides convenient functions to query and analyze logged vehicle data
Connection remains OPEN throughout the session
"""

import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate


class VehicleDataAnalyzer:
    """Helper class for querying and analyzing vehicle detection data with persistent connection"""
    
    def __init__(self, host='localhost', database='vehicle_recognition', 
                 user='root', password='12345'):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.connect()
    
    def connect(self):
        """Establish database connection with persistence"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                autocommit=True,  # Enable autocommit for persistent connection
                connection_timeout=10
            )
            if self.connection.is_connected():
                print(f"✓ Connected to database '{self.database}' (PERSISTENT)")
                self.reconnect_attempts = 0
                return True
        except Error as e:
            print(f"✗ Connection error: {e}")
            return False
    
    def ensure_connection(self):
        """Ensure database connection is alive, reconnect if needed"""
        try:
            if self.connection is None:
                return self.connect()
            
            # Ping to check if connection is alive
            self.connection.ping(reconnect=True, attempts=3, delay=1)
            return True
            
        except Error as e:
            print(f"! Connection lost, attempting to reconnect... ({e})")
            self.connection = None
            
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                return self.connect()
            else:
                print(f"! Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                return False
    
    def execute_query(self, query, params=None):
        """Execute a query and return results with connection check"""
        if not self.ensure_connection():
            print("✗ No database connection available")
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            print(f"✗ Query error: {e}")
            # Try to reconnect on next query
            self.connection = None
            return None
    
    def get_recent_detections(self, limit=10):
        """Get most recent detections"""
        query = """
        SELECT id, timestamp, vehicle_number, vehicle_type, 
               quality_score, state_name, detection_type
        FROM detected_vehicles 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        return self.execute_query(query, (limit,))
    
    def get_detections_by_timeframe(self, hours=1):
        """Get detections from last N hours"""
        query = """
        SELECT id, timestamp, vehicle_number, vehicle_type, 
               quality_score, state_name, detection_type
        FROM detected_vehicles 
        WHERE timestamp > NOW() - INTERVAL %s HOUR 
        ORDER BY timestamp DESC
        """
        return self.execute_query(query, (hours,))
    
    def get_detections_by_date(self, date_str):
        """Get detections for a specific date (YYYY-MM-DD)"""
        query = """
        SELECT id, timestamp, vehicle_number, vehicle_type, 
               quality_score, state_name, detection_type
        FROM detected_vehicles 
        WHERE DATE(timestamp) = %s 
        ORDER BY timestamp DESC
        """
        return self.execute_query(query, (date_str,))
    
    def count_by_vehicle_type(self):
        """Count detections by vehicle type"""
        query = """
        SELECT vehicle_type, COUNT(*) as count 
        FROM detected_vehicles 
        GROUP BY vehicle_type 
        ORDER BY count DESC
        """
        return self.execute_query(query)
    
    def count_by_state(self):
        """Count detections by state"""
        query = """
        SELECT state_code, state_name, COUNT(*) as count 
        FROM detected_vehicles 
        WHERE state_code IS NOT NULL 
        GROUP BY state_code, state_name 
        ORDER BY count DESC
        """
        return self.execute_query(query)
    
    def count_by_detection_type(self):
        """Count detections by detection type"""
        query = """
        SELECT detection_type, COUNT(*) as count 
        FROM detected_vehicles 
        GROUP BY detection_type 
        ORDER BY count DESC
        """
        return self.execute_query(query)
    
    def get_high_quality_detections(self, threshold=0.75):
        """Get high-quality detections only"""
        query = """
        SELECT id, timestamp, vehicle_number, vehicle_type, 
               quality_score, state_name, detection_type
        FROM detected_vehicles 
        WHERE quality_score >= %s 
        ORDER BY quality_score DESC, timestamp DESC
        """
        return self.execute_query(query, (threshold,))
    
    def search_vehicle_number(self, plate_number):
        """Search for specific vehicle number"""
        query = """
        SELECT id, timestamp, vehicle_number, vehicle_type, 
               quality_score, state_name, detection_type
        FROM detected_vehicles 
        WHERE vehicle_number LIKE %s 
        ORDER BY timestamp DESC
        """
        return self.execute_query(query, (f"%{plate_number}%",))
    
    def get_hourly_statistics(self, date_str=None):
        """Get hourly detection statistics for a date"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            HOUR(timestamp) as hour,
            COUNT(*) as total_detections,
            COUNT(DISTINCT vehicle_number) as unique_vehicles,
            AVG(quality_score) as avg_quality
        FROM detected_vehicles 
        WHERE DATE(timestamp) = %s 
        GROUP BY HOUR(timestamp) 
        ORDER BY hour
        """
        return self.execute_query(query, (date_str,))
    
    def get_total_statistics(self):
        """Get overall statistics"""
        query = """
        SELECT 
            COUNT(*) as total_detections,
            COUNT(DISTINCT vehicle_number) as unique_vehicles,
            COUNT(DISTINCT state_code) as states_detected,
            AVG(quality_score) as avg_quality_score,
            MIN(timestamp) as first_detection,
            MAX(timestamp) as last_detection
        FROM detected_vehicles
        """
        results = self.execute_query(query)
        return results[0] if results else None
    
    def delete_old_records(self, days=30):
        """Delete records older than N days"""
        if not self.ensure_connection():
            print("✗ No database connection available")
            return 0
        
        try:
            cursor = self.connection.cursor()
            query = """
            DELETE FROM detected_vehicles 
            WHERE timestamp < NOW() - INTERVAL %s DAY
            """
            cursor.execute(query, (days,))
            # No need for explicit commit with autocommit=True
            deleted = cursor.rowcount
            cursor.close()
            print(f"✓ Deleted {deleted} records older than {days} days")
            return deleted
        except Error as e:
            print(f"✗ Delete error: {e}")
            return 0
    
    def export_to_csv(self, filename, query=None):
        """Export query results to CSV"""
        if not self.ensure_connection():
            print("✗ No database connection available")
            return False
        
        if query is None:
            query = "SELECT * FROM detected_vehicles ORDER BY timestamp DESC"
        
        try:
            df = pd.read_sql(query, self.connection)
            df.to_csv(filename, index=False)
            print(f"✓ Exported {len(df)} records to {filename}")
            return True
        except Exception as e:
            print(f"✗ Export error: {e}")
            return False
    
    def print_results(self, results, title="Query Results"):
        """Pretty print query results"""
        if not results:
            print("No results found.")
            return
        
        print(f"\n{title}")
        print("=" * 80)
        
        if isinstance(results, list):
            print(tabulate(results, headers="keys", tablefmt="grid"))
        else:
            print(results)
        
        print("=" * 80)
    
    def __del__(self):
        """Destructor - only close connection when object is destroyed"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ Database connection closed (analyzer destroyed)")


def interactive_menu():
    """Interactive menu for database queries with persistent connection"""
    print("\n" + "="*60)
    print("Vehicle Detection Database - Interactive Query Tool")
    print("CONNECTION MODE: PERSISTENT (stays open)")
    print("="*60)
    
    # Get database credentials
    host = input("MySQL Host [localhost]: ").strip() or 'localhost'
    database = input("Database name [vehicle_recognition]: ").strip() or 'vehicle_recognition'
    user = input("MySQL user [root]: ").strip() or 'root'
    password = input("MySQL password [12345]: ").strip() or '12345'
    
    analyzer = VehicleDataAnalyzer(host, database, user, password)
    
    if not analyzer.connection:
        print("Failed to connect to database. Exiting.")
        return
    
    print("\n✓ Connection established and will remain OPEN during session")
    
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1.  View recent detections")
        print("2.  View detections from last N hours")
        print("3.  View detections for specific date")
        print("4.  Count by vehicle type")
        print("5.  Count by state")
        print("6.  Count by detection type")
        print("7.  View high-quality detections")
        print("8.  Search for vehicle number")
        print("9.  Hourly statistics for date")
        print("10. Overall statistics")
        print("11. Export all data to CSV")
        print("12. Delete old records")
        print("0.  Exit (will close connection)")
        print("="*60)
        print(f"Connection Status: {'CONNECTED' if analyzer.connection else 'DISCONNECTED'}")
        print("="*60)
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == '1':
            limit = int(input("Number of records [10]: ").strip() or 10)
            results = analyzer.get_recent_detections(limit)
            analyzer.print_results(results, f"Recent {limit} Detections")
        
        elif choice == '2':
            hours = int(input("Hours [1]: ").strip() or 1)
            results = analyzer.get_detections_by_timeframe(hours)
            analyzer.print_results(results, f"Detections from Last {hours} Hour(s)")
        
        elif choice == '3':
            date = input("Date (YYYY-MM-DD) [today]: ").strip()
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            results = analyzer.get_detections_by_date(date)
            analyzer.print_results(results, f"Detections for {date}")
        
        elif choice == '4':
            results = analyzer.count_by_vehicle_type()
            analyzer.print_results(results, "Detections by Vehicle Type")
        
        elif choice == '5':
            results = analyzer.count_by_state()
            analyzer.print_results(results, "Detections by State")
        
        elif choice == '6':
            results = analyzer.count_by_detection_type()
            analyzer.print_results(results, "Detections by Detection Type")
        
        elif choice == '7':
            threshold = float(input("Quality threshold [0.75]: ").strip() or 0.75)
            results = analyzer.get_high_quality_detections(threshold)
            analyzer.print_results(results, f"High Quality Detections (>= {threshold})")
        
        elif choice == '8':
            plate = input("Enter vehicle number (partial match): ").strip()
            results = analyzer.search_vehicle_number(plate)
            analyzer.print_results(results, f"Search Results for '{plate}'")
        
        elif choice == '9':
            date = input("Date (YYYY-MM-DD) [today]: ").strip()
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            results = analyzer.get_hourly_statistics(date)
            analyzer.print_results(results, f"Hourly Statistics for {date}")
        
        elif choice == '10':
            results = analyzer.get_total_statistics()
            analyzer.print_results(results, "Overall Statistics")
        
        elif choice == '11':
            filename = input("CSV filename [detections.csv]: ").strip() or 'detections.csv'
            analyzer.export_to_csv(filename)
        
        elif choice == '12':
            days = int(input("Delete records older than (days) [30]: ").strip() or 30)
            confirm = input(f"Are you sure you want to delete records older than {days} days? (yes/no): ")
            if confirm.lower() == 'yes':
                analyzer.delete_old_records(days)
        
        elif choice == '0':
            print("\nExiting and closing connection...")
            # Connection will be closed by destructor
            break
        
        else:
            print("Invalid choice. Please try again.")
    
    print("\n" + "="*60)
    print("Thank you for using Vehicle Detection Database Query Tool")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Detection Database Query Tool')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--host', type=str, default='localhost', help='MySQL host')
    parser.add_argument('--database', type=str, default='vehicle_recognition', help='Database name')
    parser.add_argument('--user', type=str, default='root', help='MySQL user')
    parser.add_argument('--password', type=str, default='12345', help='MySQL password')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_menu()
    else:
        # Example usage when run directly
        print("="*60)
        print("Running with PERSISTENT connection mode")
        print("="*60)
        
        analyzer = VehicleDataAnalyzer(args.host, args.database, args.user, args.password)
        
        if analyzer.connection:
            print("\nExample Queries:\n")
            
            # Recent detections
            results = analyzer.get_recent_detections(5)
            analyzer.print_results(results, "5 Most Recent Detections")
            
            # Statistics
            stats = analyzer.get_total_statistics()
            analyzer.print_results(stats, "Overall Statistics")
            
            # Count by vehicle type
            results = analyzer.count_by_vehicle_type()
            analyzer.print_results(results, "Detections by Vehicle Type")
            
            print("\n" + "="*60)
            print("✓ Queries completed")
            print("✓ Connection will remain OPEN until program exits")
            print("="*60)
            
            # DON'T CLOSE - let destructor handle it
            # analyzer.close()  # ❌ REMOVED
        
        print("\nTip: Run with --interactive flag for interactive menu")