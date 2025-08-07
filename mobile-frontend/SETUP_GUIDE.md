# Basketball Form Analyzer - Mobile App Setup Guide

## Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- iOS device or simulator
- Backend server running at `http://192.168.0.165:8000`

### Installation Steps

1. **Navigate to mobile frontend directory:**
   ```bash
   cd mobile-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Run on iOS device/simulator:**
   ```bash
   npm run ios
   ```

## App Features

### Camera Recording
- **5-second maximum recording** - Optimized for basketball shot capture
- **720p video quality** - Balanced quality and file size
- **Real-time recording indicator** - Shows recording time and status
- **Automatic stop** - Recording stops at 5 seconds

### Video Preview
- **Immediate preview** - Review your shot before analysis
- **Video controls** - Play, pause, and scrub through recording
- **Retake option** - Record again if needed

### Backend Integration
- **Automatic upload** - Sends video to backend for analysis
- **Progress feedback** - Loading indicators during analysis
- **Error handling** - Clear error messages for network issues

## Configuration

### Backend Settings
Edit `config.js` to change backend URL or endpoints:
```javascript
BACKEND: {
  BASE_URL: 'http://192.168.0.165:8000',
  ENDPOINTS: {
    ANALYZE_VIDEO: '/analysis/analyze-video',
  },
  TIMEOUT: 30000, // 30 seconds
}
```

### Recording Settings
Adjust recording parameters in `config.js`:
```javascript
RECORDING: {
  MAX_DURATION: 5, // seconds
  QUALITY: '720p',
  MUTE: false,
}
```

## Troubleshooting

### Common Issues

1. **Camera not working:**
   - Ensure camera permissions are granted
   - Test on physical device (camera may not work in simulator)
   - Restart the app

2. **Network errors:**
   - Verify backend server is running
   - Check network connectivity
   - Ensure backend endpoint is accessible

3. **Build errors:**
   - Clear cache: `expo r -c`
   - Reinstall dependencies: `rm -rf node_modules && npm install`
   - Reset Metro: `npm start -- --reset-cache`

### Testing Checklist

- [ ] Camera permissions granted
- [ ] Recording starts and stops properly
- [ ] Video preview displays correctly
- [ ] Backend connection works
- [ ] Analysis completes successfully
- [ ] Error handling works for network issues

## API Integration Details

### Request Format
```javascript
POST /analysis/analyze-video
Content-Type: multipart/form-data

Form Data:
- file: video file (MP4 format)
```

### Response Handling
- Success: Shows "Analysis Complete" message
- Error: Shows "Analysis Failed" with retry option
- Timeout: 30-second timeout for backend requests

## Development Notes

### File Structure
```
mobile-frontend/
├── App.js              # Main app component
├── CameraScreen.js     # Camera and recording logic
├── config.js          # Configuration settings
├── app.json           # Expo configuration
├── package.json       # Dependencies
└── README.md         # Documentation
```

### Key Dependencies
- `expo-camera` - Camera functionality
- `expo-av` - Video playback
- `axios` - HTTP requests
- `react-native-elements` - UI components

### Code Organization
- **Configuration centralized** in `config.js`
- **Camera logic** in `CameraScreen.js`
- **Error handling** with user-friendly messages
- **Loading states** for better UX

## Next Steps

1. **Test thoroughly** on physical iOS device
2. **Verify backend integration** with actual server
3. **Add analytics** for usage tracking
4. **Implement user authentication** if needed
5. **Add video history** feature
6. **Optimize performance** for different devices

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check Expo documentation for camera issues
4. Verify backend API documentation
