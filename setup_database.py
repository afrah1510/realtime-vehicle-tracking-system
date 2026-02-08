"""
MySQL Database Setup Script - PERSISTENT CONNECTION
====================================================
This script creates the database and table for vehicle recognition logging.
The connection remains OPEN after setup for immediate use.
"""

import mysql.connector
from mysql.connector import Error


class DatabaseSetup:
    """Database setup with persistent connection"""
    
    def __init__(self, host='localhost', user='root', password='12345', database='vehicle_recognition'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def setup_database(self):
        """
        Setup MySQL database and table for vehicle recognition
        Returns: connection object (keeps it open)
        """
        print("="*60)
        print("MySQL Database Setup for Vehicle Recognition System")
        print("="*60)
        
        try:
            # Connect to MySQL server
            print(f"\nConnecting to MySQL server at {self.host}...")
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                autocommit=True  # Enable autocommit for persistent connection
            )
            
            if self.connection.is_connected():
                cursor = self.connection.cursor()
                print("✓ Connected successfully!")
                
                # Create database
                print(f"\nCreating database '{self.database}'...")
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                print(f"✓ Database '{self.database}' created/verified!")
                
                # Use database
                cursor.execute(f"USE {self.database}")
                
                # Create table
                print("\nCreating table 'detected_vehicles'...")
                create_table_query = """
                CREATE TABLE IF NOT EXISTS detected_vehicles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL DEFAULT NOW(),
                    vehicle_number VARCHAR(20) NOT NULL,
                    vehicle_type VARCHAR(50) NOT NULL,
                    plate_confidence FLOAT,
                    vehicle_confidence FLOAT,
                    quality_score FLOAT,
                    state_code VARCHAR(5),
                    state_name VARCHAR(100),
                    detection_type VARCHAR(20),
                    INDEX idx_vehicle_number (vehicle_number),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_detection_type (detection_type)
                )
                """
                cursor.execute(create_table_query)
                print("✓ Table 'detected_vehicles' created/verified!")
                
                # Show table structure
                print("\nTable Structure:")
                cursor.execute("DESCRIBE detected_vehicles")
                for row in cursor.fetchall():
                    print(f"  {row[0]:20} {row[1]:20} {row[2]:10} {row[3]:10}")
                
                cursor.close()
                
                print("\n" + "="*60)
                print("✓ Database setup completed successfully!")
                print("✓ Connection REMAINS OPEN (persistent)")
                print("="*60)
                print("\nConnection details:")
                print(f"  Host: {self.host}")
                print(f"  Database: {self.database}")
                print(f"  Status: CONNECTED")
                print("="*60)
                
                return self.connection
                
        except Error as e:
            print(f"\n✗ Error: {e}")
            print("\nPlease ensure:")
            print("  1. MySQL server is running")
            print("  2. Username and password are correct")
            print("  3. User has CREATE DATABASE privileges")
            return None
    
    def get_connection(self):
        """Get the persistent connection"""
        return self.connection
    
    def close_connection(self):
        """Only call this when you want to manually close the connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✓ MySQL connection closed")


def test_connection(host='localhost', user='root', password='12345'):
    """Test MySQL connection"""
    print("\nTesting MySQL connection...")
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✓ Successfully connected to MySQL Server version {db_info}")
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup MySQL Database for Vehicle Recognition')
    parser.add_argument('--host', type=str, default='localhost', help='MySQL host (default: localhost)')
    parser.add_argument('--user', type=str, default='root', help='MySQL user (default: root)')
    parser.add_argument('--password', type=str, default='12345', help='MySQL password (default: 12345)')
    parser.add_argument('--database', type=str, default='vehicle_recognition', 
                       help='Database name (default: vehicle_recognition)')
    parser.add_argument('--keep-open', action='store_true', 
                       help='Keep connection open after setup (default: True)')
    
    args = parser.parse_args()
    
    # Test connection first
    if test_connection(args.host, args.user, args.password):
        # Setup database
        db_setup = DatabaseSetup(args.host, args.user, args.password, args.database)
        connection = db_setup.setup_database()
        
        if connection:
            if args.keep_open or True:  # Default to keeping open
                print("\n⚠ CONNECTION IS STILL OPEN")
                print("The connection will remain active for your application to use.")
                print("\nTo use this connection in your app:")
                print("  - Import this module")
                print("  - Use db_setup.get_connection()")
                print("\nTo close manually, press Ctrl+C or run:")
                print("  db_setup.close_connection()")
                
                try:
                    # Keep the script running to maintain connection
                    print("\nPress Ctrl+C to close connection and exit...")
                    import time
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n✓ Closing connection...")
                    db_setup.close_connection()
                    print("✓ Exited successfully")
            else:
                print("\nClosing connection...")
                db_setup.close_connection()
    else:
        print("\n✗ Cannot proceed with database setup due to connection error.")
        print("\nPlease check your MySQL installation and credentials.")