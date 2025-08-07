// Configuration for Basketball Form Analyzer with Synthetic Motion Profiles

export const CONFIG = {
  // Backend Configuration
  BACKEND: {
    BASE_URL: 'http://192.168.0.165:8000', // Update with your backend URL
    ENDPOINTS: {
      ANALYZE_VIDEO: '/analysis/analyze-video',
      COMPARE_WITH_PLAYER: '/analysis/compare-with-player',
      GET_PLAYER_PROFILES: '/synthetic/player-profiles',
      GENERATE_SYNTHETIC: '/synthetic/generate-profile'
    },
    TIMEOUT: 30000, // 30 seconds
  },

  // Recording Configuration
  RECORDING: {
    QUALITY: '720p',
    MAX_DURATION: 5, // seconds
    MUTE: false,
    FIELD_NAME: 'video'
  },

  // UI Configuration
  UI: {
    RECORDING_TIMER_INTERVAL: 100, // milliseconds
    LOADING_TIMEOUT: 10000, // 10 seconds
  },

  // Synthetic Data Configuration
  SYNTHETIC: {
    PLAYERS: [
      {
        id: 'lebron',
        name: 'LeBron James',
        description: 'Power forward with explosive motion',
        style: 'power',
        characteristics: ['Explosive motion', 'Strong follow-through', 'Consistent form'],
        image: 'lebron.jpg'
      },
      {
        id: 'curry',
        name: 'Stephen Curry',
        description: 'Quick release with smooth motion',
        style: 'quick',
        characteristics: ['Fast release', 'Smooth motion flow', 'Quick acceleration'],
        image: 'curry.jpg'
      },
      {
        id: 'durant',
        name: 'Kevin Durant',
        description: 'Tall shooter with smooth motion',
        style: 'smooth',
        characteristics: ['High release point', 'Smooth motion', 'Very consistent'],
        image: 'durant.jpg'
      },
      {
        id: 'kawhi',
        name: 'Kawhi Leonard',
        description: 'Defensive player with controlled motion',
        style: 'linear',
        characteristics: ['Controlled motion', 'Deliberate form', 'Defensive focus'],
        image: 'kawhi.jpg'
      },
      {
        id: 'harden',
        name: 'James Harden',
        description: 'Step-back specialist with unique motion',
        style: 'smooth',
        characteristics: ['Step-back specialist', 'Unique rhythm', 'Smooth variations'],
        image: 'harden.jpg'
      }
    ],
    PHASES: [
      'General',
      'Set-up',
      'Loading',
      'Rising',
      'Release',
      'Follow-through'
    ],
    SIMILARITY_THRESHOLDS: {
      EXCELLENT: 0.8,
      GOOD: 0.6,
      FAIR: 0.4,
      NEEDS_IMPROVEMENT: 0.0
    }
  }
};

// Helper function to get API URL
export const getApiUrl = (endpoint) => {
  return `${CONFIG.BACKEND.BASE_URL}${endpoint}`;
};

// Helper function to get similarity color
export const getSimilarityColor = (score) => {
  if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.EXCELLENT) {
    return '#4CAF50'; // Green
  } else if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.GOOD) {
    return '#FF9800'; // Orange
  } else if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.FAIR) {
    return '#FFC107'; // Yellow
  } else {
    return '#F44336'; // Red
  }
};

// Helper function to get similarity label
export const getSimilarityLabel = (score) => {
  if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.EXCELLENT) {
    return 'Excellent Match';
  } else if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.GOOD) {
    return 'Good Match';
  } else if (score >= CONFIG.SYNTHETIC.SIMILARITY_THRESHOLDS.FAIR) {
    return 'Fair Match';
  } else {
    return 'Needs Improvement';
  }
};

export default CONFIG;
