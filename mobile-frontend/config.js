// Configuration file for the Basketball Form Analyzer mobile app

export const CONFIG = {
  // Backend API configuration
  BACKEND: {
    BASE_URL: 'http://192.168.0.165:8000',
    ENDPOINTS: {
      ANALYZE_VIDEO: '/analysis/analyze-video',
    },
    TIMEOUT: 30000, // 30 seconds
  },
  
  // Recording configuration
  RECORDING: {
    MAX_DURATION: 5, // seconds
    QUALITY: '720p',
    MUTE: false,
  },
  
  // UI configuration
  UI: {
    RECORDING_TIMER_INTERVAL: 100, // milliseconds
    BUTTON_DISABLED_OPACITY: 0.5,
  },
};

// Helper function to get full API URL
export const getApiUrl = (endpoint) => {
  return `${CONFIG.BACKEND.BASE_URL}${endpoint}`;
};

// Helper function to get video quality constant
export const getVideoQuality = () => {
  const { Camera } = require('expo-camera');
  return Camera.Constants.VideoQuality[CONFIG.RECORDING.QUALITY.toUpperCase()];
};
