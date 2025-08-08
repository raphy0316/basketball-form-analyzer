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
} from 'react-native';
import { CONFIG, getSimilarityColor, getSimilarityLabel } from '../utils/config';
import DetailedAnalysisScreen from './DetailedAnalysisScreen'; // Adjust path if needed
const { width, height } = Dimensions.get('window');
import { initializeTtsListeners, playTTS } from '../utils/ttsListener';

const ResultsScreen = ({ navigation, route }) => {
  const { analysisResult, selectedPlayer } = route.params || {};
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false);

  useEffect(() => {
      initializeTtsListeners();

      setTimeout(() => {
        playTTS(analysisResult.llm_response); // or Tts.speak(message)
      }, 1000);
    }, []);

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
    const overallScore = analysisResult?.overall_similarity || 0.5;
    const color = getSimilarityColor(overallScore);
    const label = getSimilarityLabel(overallScore);
    
    return (
      <View style={styles.overallScoreContainer}>
        <Text style={styles.overallScoreTitle}>Overall Similarity</Text>
        <View style={styles.overallScoreCircle}>
          <Text style={[styles.overallScoreText, { color }]}>
            {(overallScore * 100).toFixed(0)}%
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
    const phaseScores = analysisResult?.phase_scores || {};
    
    return (
      <View style={styles.phaseBreakdownContainer}>
        <Text style={styles.phaseBreakdownTitle}>Phase-by-Phase Analysis</Text>
        {CONFIG.SYNTHETIC.PHASES.map(phase => {
          const score = phaseScores[phase] || 0.5;
          return renderPhaseScore(phase, score);
        })}
      </View>
    );
  };

  const renderRecommendations = () => {
    const recommendations = analysisResult?.recommendations || [];
    
    if (recommendations.length === 0) return null;
    
    return (
      <View style={styles.recommendationsContainer}>
        <Text style={styles.recommendationsTitle}>Recommendations</Text>
        {recommendations.map((rec, index) => (
          <View key={index} style={styles.recommendationItem}>
            <Text style={styles.recommendationText}>• {rec}</Text>
          </View>
        ))}
      </View>
    );
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
        {renderRecommendations()}

        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => setIsDetailModalVisible(true)}
          >
            <Text style={styles.actionButtonText}>View Detailed Analysis</Text>
          </TouchableOpacity>
          
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
});

export default ResultsScreen;