# Basketball Form Analyzer Backend API

This is the FastAPI backend for the Basketball Form Analyzer mobile app. It provides endpoints for analyzing basketball shooting videos using computer vision and machine learning.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
```bash
python start_backend.py
```

Or directly with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Verify the Server is Running
- **Health Check**: http://192.168.0.165:8000/health
- **API Documentation**: http://192.168.0.165:8000/docs
- **Root Endpoint**: http://192.168.0.165:8000/

## 📋 API Endpoints

### POST `/analysis/analyze-video`
Analyzes a basketball shooting video and returns form analysis results.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Video file with field name `video`

**Response:**
```json
{
  "player_match": "Kevin Durant",
  "similarity_score": 0.82,
  "phase_scores": {
    "dip": 0.78,
    "setpoint": 0.85,
    "release": 0.81,
    "follow_through": 0.84
  },
  "feedback": [
    "Try to lower your dip slightly for better consistency.",
    "Your release angle is very similar to Kevin Durant's form."
  ]
}
```

### POST `/analysis/analyze-video-test`
Test endpoint that returns mock results (no video processing).

### GET `/health`
Health check endpoint.

## 🔧 Configuration

### Environment Variables
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

### CORS Settings
The API is configured to allow requests from any origin (`*`). In production, you should specify your mobile app's domain.

## 📱 Mobile App Integration

The mobile app should send POST requests to:
```
http://192.168.0.165:8000/analysis/analyze-video
```

With the video file in `multipart/form-data` format.

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   ```

2. **Dependencies not found**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pipeline initialization fails**
   - Check that all required model files are present
   - Verify the `basketball_shooting_integrated_pipeline.py` file exists

### Logs
The server logs will show:
- Pipeline initialization status
- Video analysis progress
- Error messages and stack traces

## 🔒 Security Notes

- The API currently allows all origins (`*`) for CORS
- No authentication is implemented
- In production, add proper authentication and CORS restrictions

## 📊 Performance

- Video analysis typically takes 10-30 seconds depending on video length
- The pipeline uses GPU acceleration if available
- Temporary video files are automatically cleaned up after analysis
