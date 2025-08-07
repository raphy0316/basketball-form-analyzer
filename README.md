# Basketball Form Analyzer with Synthetic Motion Profiles

A comprehensive basketball shooting form analysis system that compares user videos with synthetic NBA player motion profiles. This system provides real-time feedback and detailed phase-by-phase analysis of shooting mechanics.

## 🏀 Features

### **Synthetic Motion Profiles**
- **5 NBA Player Styles**: LeBron James, Stephen Curry, Kevin Durant, Kawhi Leonard, James Harden
- **Realistic Biomechanics**: Each player has unique motion characteristics and phase distributions
- **6 Shooting Phases**: General, Set-up, Loading, Rising, Release, Follow-through
- **Motion Curves**: Power, Quick, Smooth, Linear acceleration patterns

### **Mobile App Integration**
- **Player Selection**: Choose from 5 NBA player styles to compare against
- **Real-time Recording**: Capture basketball shots with camera integration
- **Instant Analysis**: Get immediate feedback and similarity scores
- **Phase Breakdown**: Detailed analysis of each shooting phase
- **Recommendations**: Personalized improvement suggestions

### **Backend API**
- **FastAPI Server**: High-performance backend with synthetic profiles integration
- **Video Analysis**: Pose detection and ball tracking
- **Comparison Engine**: Compare user videos with synthetic player data
- **RESTful Endpoints**: Easy integration with mobile and web clients

## 🚀 Quick Start

### **1. Backend Setup**

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Start the server
python main.py
```

The backend will be available at `http://localhost:8000`

### **2. Mobile App Setup**

```bash
# Install dependencies
cd mobile-frontend
npm install

# Start the development server
npx expo start
```

### **3. Generate Synthetic Profiles**

```bash
# Generate profiles for all players
cd synthetic_profiles
python motion_profile_generator.py
```

## 📱 Mobile App Flow

1. **Player Selection**: Choose an NBA player to compare against
2. **Video Recording**: Record your basketball shot
3. **Analysis**: Backend processes video and compares with synthetic data
4. **Results**: View similarity scores and recommendations
5. **Improvement**: Get specific feedback for each shooting phase

## 🔧 API Endpoints

### **Analysis Endpoints**
- `POST /analysis/analyze-video` - Basic video analysis
- `POST /analysis/compare-with-player` - Compare with specific player

### **Synthetic Data Endpoints**
- `GET /synthetic/player-profiles` - Get available player profiles
- `POST /synthetic/generate-profile` - Generate specific player profile

### **Health Check**
- `GET /health` - Server health and capabilities

## 🎯 Player Styles

### **LeBron James**
- **Style**: Power-based, athletic motion
- **Characteristics**: Explosive motion, strong follow-through, consistent form
- **Motion Curve**: Power acceleration
- **Frame Count**: 90 frames

### **Stephen Curry**
- **Style**: Quick release with smooth motion
- **Characteristics**: Fast release, smooth motion flow, quick acceleration
- **Motion Curve**: Quick acceleration
- **Frame Count**: 85 frames

### **Kevin Durant**
- **Style**: Tall shooter with smooth motion
- **Characteristics**: High release point, smooth motion, very consistent
- **Motion Curve**: Smooth S-curve
- **Frame Count**: 85 frames

### **Kawhi Leonard**
- **Style**: Controlled, deliberate motion
- **Characteristics**: Controlled motion, deliberate form, defensive focus
- **Motion Curve**: Linear motion
- **Frame Count**: 88 frames

### **James Harden**
- **Style**: Step-back specialist with unique rhythm
- **Characteristics**: Step-back specialist, unique rhythm, smooth variations
- **Motion Curve**: Smooth S-curve
- **Frame Count**: 87 frames

## 📊 Analysis Results

The system provides:

- **Overall Similarity Score**: Percentage match with selected player
- **Phase-by-Phase Scores**: Individual scores for each shooting phase
- **Recommendations**: Specific improvement suggestions
- **Comparison Metrics**: Motion consistency, release timing, follow-through

## 🛠️ Technical Architecture

### **Frontend (React Native)**
- Camera integration with Expo
- Real-time video recording
- Player selection interface
- Results visualization

### **Backend (FastAPI)**
- Video processing pipeline
- Synthetic profile generation
- Comparison algorithms
- RESTful API endpoints

### **Synthetic Profiles**
- Biomechanically accurate motion data
- Normalized coordinates with noise simulation
- JSON export format
- Caching for performance

## 🔄 Integration Flow

1. **User selects NBA player** → Mobile app sends player ID
2. **User records shot** → Video uploaded to backend
3. **Backend processes video** → Pose detection and phase segmentation
4. **Synthetic profile loaded** → Player-specific motion data retrieved
5. **Comparison executed** → Similarity scores calculated
6. **Results returned** → Mobile app displays analysis

## 📁 Project Structure

```
basketball-form-analyzer/
├── mobile-frontend/          # React Native mobile app
│   ├── CameraScreen.js       # Video recording interface
│   ├── PlayerSelectionScreen.js  # Player selection
│   ├── ResultsScreen.js      # Analysis results display
│   └── config.js            # App configuration
├── backend/                  # FastAPI backend
│   ├── main.py              # Main server file
│   └── requirements.txt     # Python dependencies
├── synthetic_profiles/       # Synthetic data generation
│   ├── motion_profile_generator.py  # Profile generator
│   └── README.md            # Synthetic profiles documentation
└── README.md                # This file
```

## 🎮 Usage Examples

### **Basic Analysis**
```bash
curl -X POST "http://localhost:8000/analysis/analyze-video" \
  -F "video=@shot.mp4"
```

### **Player Comparison**
```bash
curl -X POST "http://localhost:8000/analysis/compare-with-player" \
  -F "video=@shot.mp4" \
  -F "player_id=curry" \
  -F "player_style=quick"
```

### **Get Player Profiles**
```bash
curl "http://localhost:8000/synthetic/player-profiles"
```

## 🔮 Future Enhancements

- **More Players**: Additional NBA player profiles
- **Custom Profiles**: User-generated motion profiles
- **Advanced Analytics**: Machine learning-based insights
- **Real-time Feedback**: Live coaching during recording
- **Social Features**: Share and compare with friends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ for basketball players everywhere** 