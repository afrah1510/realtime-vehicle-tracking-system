# Suraksha: Automated Vehicle Tracking and Insight Platform

A real-time vehicle detection and license plate recognition system with database logging and web-based analytics dashboard.

## Features

- **Dual YOLO Detection**: Vehicle type classification and license plate detection
- **OCR Recognition**: Tesseract-based license plate reading with Indian plate format validation
- **Real-time Tracking**: Multi-object tracking with unique ID assignment
- **Smart Logging**: Logs vehicles after 3 consecutive frame detections
- **Persistent Database**: MySQL storage with connection pooling
- **Web Dashboard**: Flask-based analytics interface with real-time statistics
- **Video Support**: Works with webcam feeds and video files

## Prerequisites

### System Requirements
- Python 3.8 or higher
- MySQL Server 5.7 or higher
- Tesseract OCR 4.0 or higher
- Webcam (for live detection)

### Install Tesseract OCR

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Installation

1. **Clone or download the project**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up MySQL database:**
```bash
mysql -u root -p
```
Enter your MySQL password, then the database will be created automatically on first run.

4. **Update database credentials** (if different from defaults):
   - Edit `suraksha_main.py` line 83-86
   - Edit `app.py` line 18-21
   - Edit `query_database.py` line 16-19

## Project Structure

```
├── models/
│   ├── license_plate.pt      # YOLO plate detection model
│   └── vehicle_model.pt      # YOLO vehicle classification model
├── output/                   # Saved processed videos
├── templates/
│   └── index.html            # Dashboard HTML
├── static/                   # CSS/JS files
├── suraksha_main.py          # Main detection script
├── app.py                    # Flask web server
├── query_database.py         # Database query tool
└── requirements.txt
```

## Usage

### 1. Run Vehicle Detection (Webcam)

```bash
python suraksha_main.py --camera 0
```

**Common options:**
- `--camera 1` - Use camera ID 1
- `--video path/to/video.mp4` - Process video file
- `--min-frames 3` - Frames needed for logging (default: 3)
- `--no-save` - Don't save output video
- `--display-width 1280` - Window width

**Full example:**
```bash
python suraksha_main.py --video sample.mp4 --min-frames 3 --display-width 1920
```

### 2. Start Web Dashboard

```bash
python app.py
```

Access at: http://localhost:5000

### 3. Query Database (Interactive)

```bash
python query_database.py --interactive
```

## Database Configuration

**Default credentials:**
- Host: `localhost`
- Database: `vehicle_recognition`
- User: `root`
- Password: `12345`

**To change:**
Edit the `DB_CONFIG` dictionary in each Python file or pass as command-line arguments:
```bash
python suraksha_main.py --db-host localhost --db-user root --db-password yourpass
```

## How It Works

1. **Detection Phase**: YOLO models detect vehicles and license plates in each frame
2. **OCR Phase**: Tesseract reads text from detected plates with preprocessing
3. **Validation Phase**: Indian plate format validation and correction
4. **Tracking Phase**: Assigns unique IDs and tracks vehicles across frames
5. **Logging Phase**: After 3 consecutive same-plate detections, logs to database
6. **Display Phase**: Shows vehicle ID, type, and plate number on screen

## Keyboard Controls

- **Q** - Quit application
- Window displays live FPS and detection statistics

## Troubleshooting

**"Tesseract not found" error:**
- Install Tesseract OCR (see Prerequisites)
- Verify installation: `tesseract --version`

**"Cannot connect to MySQL" error:**
- Start MySQL service
- Check credentials in configuration
- Ensure MySQL is running on port 3306

**No detections appearing:**
- Check camera ID (try 0, 1, 2)
- Ensure YOLO model files are in `models/` folder
- Verify models are named correctly: `license_plate.pt` and `vehicle_model.pt`

**Low FPS:**
- Reduce display width: `--display-width 1280`
- Use smaller video resolution
- Check GPU availability for PyTorch

## Database Schema

**Table: `detected_vehicles`**
- `id` - Auto-increment primary key
- `timestamp` - Detection time
- `vehicle_number` - License plate number
- `vehicle_type` - Vehicle classification
- `plate_confidence` - Plate detection confidence
- `vehicle_confidence` - Vehicle detection confidence
- `quality_score` - Overall quality metric
- `state_code` - Indian state code (e.g., MH, DL)
- `state_name` - Full state name
- `detection_type` - Detection method

## Performance Notes

- **Database Connection**: Uses persistent connection pooling for efficiency
- **Frame Processing**: ~15-30 FPS depending on hardware
- **Logging Strategy**: Reduces duplicate entries by requiring 3 consecutive detections
- **OCR Preprocessing**: Multiple preprocessing pipelines for accuracy

## Credits

- YOLO models: Ultralytics
- OCR: Tesseract
- Web Framework: Flask
- Database: MySQL

