import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Dimensions,
  Modal,
  Image,
  Linking,
  Alert,
} from 'react-native';
import { CONFIG, getSimilarityColor, getSimilarityLabel } from '../utils/config';
import DetailedAnalysisScreen from './DetailedAnalysisScreen'; // Adjust path if needed
const { width, height } = Dimensions.get('window');
import { initializeTtsListeners, playTTS } from '../utils/ttsListener';

const ResultsScreen = ({ navigation, route }) => {
  const { analysisResult, selectedPlayer } = route.params || {};
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false);
  const [isImageModalVisible, setIsImageModalVisible] = useState(false);
  const [currentImagePath, setCurrentImagePath] = useState('');
  const [screenDimensions, setScreenDimensions] = useState({
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height
  });

  useEffect(() => {
    initializeTtsListeners();

    setTimeout(() => {
      playTTS(analysisResult?.llm_response); 
    }, 3000);
    
    // Fix for Dimensions API - use the newer subscription-based API
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setScreenDimensions({
        width: window.width,
        height: window.height
      });
    });
    
    // Clean up properly using the subscription object
    return () => {
      if (subscription?.remove) {
        subscription.remove();
      }
    };
  }, [analysisResult]);

  const renderPhaseScore = (phase, score) => {
    const color = getSimilarityColor(score);
    const label = getSimilarityLabel(score);
    
    return (
      <View key={phase} style={styles.phaseScoreContainer}>
        <View style={styles.phaseHeader}>
          <Text style={styles.phaseName}>{phase}</Text>
          <Text style={[styles.phaseScore, { color }]}>
            {(score * 100).toFixed(0)}%
          </Text>
        </View>
        <View style={styles.progressBarContainer}>
          <View style={[styles.progressBar, { width: `${score * 100}%`, backgroundColor: color }]} />
        </View>
        <Text style={[styles.phaseLabel, { color }]}>{label}</Text>
      </View>
    );
  };

  const renderOverallScore = () => {
    const overallScore = analysisResult?.comparison_result?.dtw_analysis?.overall_similarity || 50;
    const color = getSimilarityColor(overallScore);
    const label = getSimilarityLabel(overallScore);
    
    return (
      <View style={styles.overallScoreContainer}>
        <Text style={styles.overallScoreTitle}>Overall Similarity</Text>
        <View style={styles.overallScoreCircle}>
          <Text style={[styles.overallScoreText, { color }]}>
            {(overallScore).toFixed(0)}%
          </Text>
        </View>
        <Text style={[styles.overallScoreLabel, { color }]}>{label}</Text>
      </View>
    );
  };

  const renderPlayerComparison = () => {
    if (!selectedPlayer) return null;
    
    return (
      <View style={styles.playerComparisonContainer}>
        <Text style={styles.playerComparisonTitle}>
          Comparison with {selectedPlayer.name}
        </Text>
        <View style={styles.playerInfoRow}>
          <Text style={styles.playerInfoLabel}>Style:</Text>
          <Text style={styles.playerInfoValue}>{selectedPlayer.description}</Text>
        </View>
        <View style={styles.playerInfoRow}>
          <Text style={styles.playerInfoLabel}>Characteristics:</Text>
          <Text style={styles.playerInfoValue}>
            {selectedPlayer.characteristics.join(', ')}
          </Text>
        </View>
      </View>
    );
  };

  const renderPhaseBreakdown = () => {
    
    const phaseScores = analysisResult?.comparison_result?.dtw_analysis?.phase_similarities || {};
  
    return (
      <View style={styles.phaseBreakdownContainer}>
        <Text style={styles.phaseBreakdownTitle}>Phase-by-Phase Analysis</Text>
        {Object.entries(phaseScores).map(([phase, data]) => {
          const score = (data.similarity ?? 50) / 100; // Convert to 0-1 range for progress bar
          return (
            <View key={phase} style={styles.phaseScoreContainer}>
              {renderPhaseScore(phase, score)}
              {/* <Text style={styles.phaseDetail}>
                Frames: {data.frame_count_1} vs {data.frame_count_2} | Features: {data.feature_count}
              </Text>
              <Text style={styles.phaseNote}>{data.note}</Text> */}
            </View>
          );
        })}
      </View>
    );
  };

  // Function to open image in modal
  const openImageViewer = (imagePath) => {
    setCurrentImagePath(imagePath);
    setIsImageModalVisible(true);
  };

    
  const renderRecommendations = () => {
    const recommendations = analysisResult?.recommendations || [];
    
    if (recommendations.length === 0) {
      return null;
    }
    
    return (
      <View style={styles.recommendationsContainer}>
        <Text style={styles.recommendationsTitle}>Recommendations</Text>
        {recommendations.map((recommendation, index) => (
          <View key={index} style={styles.recommendationItem}>
            <Text style={styles.recommendationText}>• {recommendation}</Text>
          </View>
        ))}
      </View>
    );
  };

  const handleViewAnalysisVideo = () => {
    const videoPath = analysisResult?.normalized_video_path;
    
    if (!videoPath) {
      Alert.alert(
        "Video Not Available",
        "The analysis video is not available for this shot.",
        [{ text: "OK" }]
      );
      return;
    }
    
    // Extract video name from path
    const videoName = videoPath.split('/').pop();
    const videoUrl = `${CONFIG.BACKEND.BASE_URL}/video/normalized-analysis/${videoName}`;
    
    // Try to open the video in the device's default video player
    Linking.openURL(videoUrl).catch((err) => {
      Alert.alert(
        "Error Opening Video",
        "Unable to open the analysis video. Please try again later.",
        [{ text: "OK" }]
      );
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={styles.title}>Analysis Results</Text>
          {selectedPlayer && (
            <Text style={styles.subtitle}>
              Your shot compared to {selectedPlayer.name}'s style
            </Text>
          )}
        </View>

        {renderOverallScore()}
        {renderPlayerComparison()}
        {renderPhaseBreakdown()}

        <View style={styles.actionsContainer}>
          {/* Image viewer button - placed above detailed analysis */}
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => openImageViewer(analysisResult?.image_path)}
          >
            <Text style={styles.actionButtonText}>View Shot Image</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => setIsDetailModalVisible(true)}
          >
            <Text style={styles.actionButtonText}>View Detailed Analysis</Text>
          </TouchableOpacity>
          
          {analysisResult?.normalized_video_path && (
            <TouchableOpacity
              style={[styles.actionButton, styles.videoButton]}
              onPress={handleViewAnalysisVideo}
            >
              <Text style={styles.actionButtonText}>View Analysis Video</Text>
            </TouchableOpacity>
          )}
          
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => navigation.navigate('PlayerSelection')}
          >
            <Text style={styles.actionButtonText}>Try Another Player</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.actionButton, styles.secondaryButton]}
            onPress={() => navigation.navigate('Main')}
          >
            <Text style={styles.secondaryButtonText}>Record New Shot</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Detail Screen Modal */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={isDetailModalVisible}
        onRequestClose={() => setIsDetailModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Detailed Analysis</Text>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setIsDetailModalVisible(false)}
              >
                <Text style={styles.closeButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
            < DetailedAnalysisScreen
              detailedResult={analysisResult} 
              selectedPlayer={selectedPlayer}
            />
          </View>
        </View>
      </Modal>

      {/* Image Viewer Modal */}
      <Modal
        animationType="fade"
        transparent={true}
        visible={isImageModalVisible}
        onRequestClose={() => setIsImageModalVisible(false)}
        supportedOrientations={['portrait', 'landscape']}
      >
        <View style={styles.modalContainer}>
          <View style={[
            styles.imageModalContent, 
            {
              width: screenDimensions.width * 0.95,
              height: screenDimensions.height * 0.85
            }
          ]}>
            <View style={styles.compactModalHeader}>
              <Text style={styles.compactModalTitle}>Pinch to zoom, drag to move</Text>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setIsImageModalVisible(false)}
              >
                <Text style={styles.closeButtonText}>Close</Text>
              </TouchableOpacity>
            </View>
            
            {/* Fix ScrollView layout issue by removing style props from ScrollView and putting them in contentContainerStyle */}
            <ScrollView
              style={styles.imageScrollContainer}
              contentContainerStyle={styles.imageContentContainer}
              maximumZoomScale={5}
              minimumZoomScale={1}
              showsHorizontalScrollIndicator={false}
              showsVerticalScrollIndicator={false}
              pinchGestureEnabled={true}
              scrollEnabled={true}
            >
              {currentImagePath ? (
                <Image
                  source={{ uri: `${CONFIG.BACKEND.BASE_URL}${currentImagePath}` }}
                  style={styles.zoomableImage}
                  resizeMode="contain"
                />
              ) : (
                <Text style={styles.noImageText}>No image to display</Text>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
  },
  overallScoreContainer: {
    alignItems: 'center',
    padding: 20,
    marginBottom: 20,
  },
  overallScoreTitle: {
    fontSize: 18,
    color: 'white',
    marginBottom: 16,
  },
  overallScoreCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#1a1a1a',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#333',
    marginBottom: 12,
  },
  overallScoreText: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  overallScoreLabel: {
    fontSize: 16,
    fontWeight: '600',
  },
  playerComparisonContainer: {
    backgroundColor: '#1a1a1a',
    margin: 20,
    padding: 16,
    borderRadius: 12,
  },
  playerComparisonTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 12,
  },
  playerInfoRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  playerInfoLabel: {
    fontSize: 14,
    color: '#888',
    width: 100,
  },
  playerInfoValue: {
    fontSize: 14,
    color: 'white',
    flex: 1,
  },
  phaseBreakdownContainer: {
    backgroundColor: '#1a1a1a',
    margin: 20,
    padding: 16,
    borderRadius: 12,
  },
  phaseBreakdownTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 16,
  },
  phaseScoreContainer: {
    marginBottom: 16,
  },
  phaseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  phaseName: {
    fontSize: 16,
    color: 'white',
    fontWeight: '500',
  },
  phaseScore: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#333',
    borderRadius: 4,
    marginBottom: 4,
  },
  progressBar: {
    height: '100%',
    borderRadius: 4,
  },
  phaseLabel: {
    fontSize: 12,
    fontWeight: '500',
  },
  recommendationsContainer: {
    backgroundColor: '#1a1a1a',
    margin: 20,
    padding: 16,
    borderRadius: 12,
  },
  recommendationsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 12,
  },
  recommendationItem: {
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 20,
  },
  actionsContainer: {
    padding: 20,
    gap: 12,
  },
  actionButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  actionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  secondaryButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  videoButton: {
    backgroundColor: '#4CAF50', // Green color for video button
  },
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: width * 0.9,
    height: height * 0.8,
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    overflow: 'hidden',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
  closeButton: {
    padding: 8,
  },
  closeButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  imageModalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    overflow: 'hidden',
  },
  compactModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 8,
    paddingVertical: 6,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
  },
  compactModalTitle: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: '500',
  },
  imageScrollContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  imageContentContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  zoomableImage: {
    width: width * 0.95,
    height: height * 0.8,
    resizeMode: 'contain',
  },
  noImageText: {
    color: '#888',
    fontSize: 16,
    fontWeight: '500',
  },
});

export default ResultsScreen;