# Synthetic Motion Profiles Integration

This document describes the integration of synthetic basketball motion profiles with the existing mobile app and backend.

## 🏀 Overview

The synthetic motion profiles system has been successfully integrated with the existing `mobile-app-integrated` branch. This integration allows users to:

1. **Select NBA Players**: Choose from 5 NBA player styles (LeBron, Curry, Durant, Kawhi, Harden)
2. **Compare Shooting Form**: Record videos and compare them with synthetic player data
3. **Get Detailed Analysis**: Receive phase-by-phase similarity scores and recommendations
4. **View Results**: See comprehensive comparison results with visual feedback

## 📱 Mobile App Integration

### New Screens Added

1. **PlayerSelectionScreen** (`mobile/src/screens/PlayerSelectionScreen.jsx`)
   - Displays all 5 NBA player options
   - Shows player characteristics and descriptions
   - Allows user to select a player for comparison

2. **Updated CameraScreen** (`mobile/src/screens/CameraScreen.jsx`)
   - Shows selected player information during recording
   - Sends player data to backend for comparison
   - Supports both basic analysis and player comparison modes

3. **Updated ResultsScreen** (`mobile/src/screens/ResultsScreen.jsx`)
   - Displays overall similarity score
   - Shows phase-by-phase breakdown
   - Provides player comparison details
   - Includes recommendations for improvement

### Navigation Flow

```
LandingScreen
├── "Compare with NBA Players" → PlayerSelectionScreen
│   └── Select Player → CameraScreen (with player data)
│       └── Record Video → ResultsScreen (with comparison)
└── "Basic Analysis" → CameraScreen (basic mode)
    └── Record Video → ResultsScreen (basic results)
```

### Configuration

- **Config File**: `mobile/src/utils/config.js`
  - Backend endpoints
  - Player definitions
  - Similarity thresholds
  - Helper functions

## 🔧 Backend Integration

### New Endpoints

1. **`POST /analysis/compare-with-player`**
   - Compares user video with selected player
   - Returns similarity scores and recommendations

2. **`GET /synthetic/player-profiles`**
   - Returns available player profiles
   - Includes player characteristics and metadata

3. **`POST /synthetic/generate-profile`**
   - Generates synthetic profile for specific player
   - Returns motion data in JSON format

### Integration with Existing Backend

- **Main App**: `backend/main.py` includes synthetic routes
- **Existing Routes**: Model and LLM routes remain unchanged
- **CORS**: Configured to allow mobile app access
- **Error Handling**: Graceful fallback when synthetic profiles unavailable

### Synthetic Profiles System

- **Location**: `synthetic_profiles/motion_profile_generator.py`
- **Players**: 5 NBA player styles with unique characteristics
- **Phases**: 6 shooting phases (General, Set-up, Loading, Rising, Release, Follow-through)
- **Data Format**: JSON with keypoints, ball positions, and confidence scores

## 🎯 Player Styles

| Player | Style | Characteristics | Frame Count |
|--------|-------|-----------------|-------------|
| LeBron James | Power | Explosive motion, strong follow-through | 90 |
| Stephen Curry | Quick | Fast release, smooth motion flow | 85 |
| Kevin Durant | Smooth | High release point, very consistent | 85 |
| Kawhi Leonard | Linear | Controlled motion, deliberate form | 88 |
| James Harden | Smooth | Step-back specialist, unique rhythm | 87 |

## 🚀 Usage

### Starting the Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Starting the Mobile App

```bash
cd mobile
npm install
npx expo start
```

### Testing the Integration

1. **Backend Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Get Player Profiles**:
   ```bash
   curl http://localhost:8000/synthetic/player-profiles
   ```

3. **Generate Player Profile**:
   ```bash
   curl -X POST http://localhost:8000/synthetic/generate-profile \
     -F "player_id=curry"
   ```

## 📊 Analysis Results

The system provides:

- **Overall Similarity**: Percentage match with selected player
- **Phase Scores**: Individual scores for each shooting phase
- **Recommendations**: Specific improvement suggestions
- **Comparison Metrics**: Motion consistency, release timing, follow-through

### Similarity Thresholds

- **Excellent**: ≥80% (Green)
- **Good**: ≥60% (Orange)
- **Fair**: ≥40% (Yellow)
- **Needs Improvement**: <40% (Red)

## 🔄 Data Flow

1. **User selects player** → Mobile app stores player data
2. **User records video** → Video uploaded to backend with player info
3. **Backend processes** → Analyzes video and loads synthetic profile
4. **Comparison executed** → Calculates similarity scores
5. **Results returned** → Mobile app displays analysis

## 🛠️ Technical Details

### Mobile App
- **Framework**: React Native with Expo
- **Navigation**: React Navigation Stack
- **Camera**: Expo Camera for video recording
- **API**: Axios for backend communication

### Backend
- **Framework**: FastAPI
- **Synthetic Data**: Custom motion profile generator
- **Analysis**: Integration with existing basketball analysis pipeline
- **Caching**: In-memory cache for synthetic profiles

### Synthetic Profiles
- **Language**: Python
- **Dependencies**: NumPy, SciPy
- **Output**: JSON format with normalized coordinates
- **Features**: Noise simulation, motion curves, phase distribution

## 🔮 Future Enhancements

- **More Players**: Additional NBA player profiles
- **Custom Profiles**: User-generated motion profiles
- **Real-time Feedback**: Live coaching during recording
- **Advanced Analytics**: Machine learning-based insights
- **Social Features**: Share and compare with friends

## 🆘 Troubleshooting

### Common Issues

1. **Synthetic Profiles Not Available**
   - Check if `synthetic_profiles` directory exists
   - Verify Python dependencies are installed
   - Check import paths in backend

2. **Mobile App Can't Connect**
   - Verify backend URL in `config.js`
   - Check CORS settings
   - Ensure backend is running

3. **Video Upload Fails**
   - Check file size limits
   - Verify video format (MP4)
   - Check network connectivity

### Debug Commands

```bash
# Test synthetic profiles
cd synthetic_profiles
python3 -c "from motion_profile_generator import SyntheticProfileGenerator, create_curry_style; g = SyntheticProfileGenerator(); p = g.generate_profile(create_curry_style()); print(f'Generated {len(p)} frames')"

# Test backend
cd backend
python3 -c "from main import app; print('Backend loaded successfully')"
```

---

**Integration completed successfully! The synthetic motion profiles are now fully integrated with the existing mobile app and backend.**
