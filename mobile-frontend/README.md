# Basketball Form Analyzer - Mobile App

A React Native mobile app built with Expo that allows users to record basketball shots and analyze their form using the backend API.

## Features

- 📱 iOS camera integration with recording capabilities
- 🎥 5-second video recording with 720p quality
- 👀 Video preview before analysis
- 🔄 Retake functionality
- 📊 Integration with basketball form analysis backend
- 🎨 Modern, intuitive UI following iOS design guidelines

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- iOS Simulator (for testing) or physical iOS device
- Backend server running at `http://192.168.0.165:8000`

## Installation

1. Navigate to the mobile frontend directory:
   ```bash
   cd mobile-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Run on iOS:
   ```bash
   npm run ios
   ```

## Usage

1. **Launch the app** - The camera screen will appear
2. **Grant permissions** - Allow camera and microphone access when prompted
3. **Position yourself** - Stand in frame for your basketball shot
4. **Record** - Tap the record button to start recording (max 5 seconds)
5. **Preview** - Review your recorded shot
6. **Analyze** - Tap "Analyze Shot" to send to backend for analysis
7. **Retake** - If needed, tap "Retake" to record again

## API Integration

The app integrates with the backend API at `http://192.168.0.165:8000`:

- **Endpoint**: `/analysis/analyze-video`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **File Key**: `file`

## Configuration

### Backend URL
Update the backend URL in `CameraScreen.js` if needed:
```javascript
const BACKEND_URL = 'http://192.168.0.165:8000';
```

### Recording Settings
Adjust recording parameters in `CameraScreen.js`:
```javascript
const MAX_RECORDING_TIME = 5; // seconds
// Video quality: Camera.Constants.VideoQuality['720p']
```

## Project Structure

```
mobile-frontend/
├── App.js                 # Main app component
├── CameraScreen.js        # Camera and recording functionality
├── app.json              # Expo configuration
├── package.json          # Dependencies
└── README.md            # This file
```

## Dependencies

- `expo-camera` - Camera functionality
- `expo-av` - Video playback
- `expo-file-system` - File handling
- `axios` - HTTP requests
- `react-native-elements` - UI components

## Troubleshooting

### Camera Permissions
If camera doesn't work:
1. Check device settings for camera permissions
2. Restart the app
3. Ensure you're testing on a physical device (camera may not work in simulator)

### Network Issues
If analysis fails:
1. Verify backend server is running at `http://192.168.0.165:8000`
2. Check network connectivity
3. Ensure backend endpoint `/analysis/analyze-video` is accessible

### Build Issues
If you encounter build problems:
1. Clear Expo cache: `expo r -c`
2. Delete node_modules and reinstall: `rm -rf node_modules && npm install`
3. Reset Metro bundler: `npm start -- --reset-cache`

## Development

### Adding New Features
1. Create new components in separate files
2. Update `App.js` to include new screens
3. Test thoroughly on physical device

### Styling
The app uses React Native StyleSheet for styling. Follow iOS design guidelines for consistency.

## Contributing

1. Create a feature branch
2. Make your changes
3. Test on physical iOS device
4. Submit a pull request

## License

This project is part of the Basketball Form Analyzer system.
