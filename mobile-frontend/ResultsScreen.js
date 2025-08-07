import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';

const ResultsScreen = ({ route, navigation }) => {
  const { analysisResult } = route.params;

  const handleRecordAgain = () => {
    navigation.navigate('Camera');
  };

  const renderPhaseScores = () => {
    if (!analysisResult.phase_scores) return null;

    const phases = [
      { key: 'dip', label: 'Dip Phase', color: '#FF6B6B' },
      { key: 'setpoint', label: 'Set Point', color: '#4ECDC4' },
      { key: 'release', label: 'Release', color: '#45B7D1' },
      { key: 'follow_through', label: 'Follow Through', color: '#96CEB4' },
    ];

    return (
      <View style={styles.phaseScoresContainer}>
        <Text style={styles.sectionTitle}>Phase Breakdown</Text>
        {phases.map((phase) => {
          const score = analysisResult.phase_scores[phase.key];
          if (!score) return null;
          
          return (
            <View key={phase.key} style={styles.phaseRow}>
              <Text style={styles.phaseLabel}>{phase.label}</Text>
              <View style={styles.scoreContainer}>
                <View 
                  style={[
                    styles.scoreBar, 
                    { 
                      width: `${score * 100}%`,
                      backgroundColor: phase.color 
                    }
                  ]} 
                />
                <Text style={styles.scoreText}>{Math.round(score * 100)}%</Text>
              </View>
            </View>
          );
        })}
      </View>
    );
  };

  const renderFeedback = () => {
    if (!analysisResult.feedback || analysisResult.feedback.length === 0) return null;

    return (
      <View style={styles.feedbackContainer}>
        <Text style={styles.sectionTitle}>Feedback</Text>
        {analysisResult.feedback.map((feedback, index) => (
          <View key={index} style={styles.feedbackItem}>
            <Text style={styles.feedbackText}>• {feedback}</Text>
          </View>
        ))}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Analysis Complete!</Text>
          <Text style={styles.subtitle}>Here's how your shot compares</Text>
        </View>

        {/* Main Result */}
        <View style={styles.mainResult}>
          <View style={styles.playerMatch}>
            <Text style={styles.playerMatchLabel}>Your form matches</Text>
            <Text style={styles.playerName}>{analysisResult.player_match}</Text>
          </View>
          
          <View style={styles.similarityContainer}>
            <Text style={styles.similarityLabel}>Similarity Score</Text>
            <View style={styles.similarityScore}>
              <Text style={styles.similarityPercentage}>
                {Math.round(analysisResult.similarity_score * 100)}%
              </Text>
            </View>
          </View>
        </View>

        {/* Phase Scores */}
        {renderPhaseScores()}

        {/* Feedback */}
        {renderFeedback()}

        {/* Action Buttons */}
        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={styles.recordAgainButton}
            onPress={handleRecordAgain}
          >
            <Text style={styles.recordAgainText}>Record Another Shot</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
  },
  mainResult: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  playerMatch: {
    alignItems: 'center',
    marginBottom: 20,
  },
  playerMatchLabel: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 8,
  },
  playerName: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#3498db',
  },
  similarityContainer: {
    alignItems: 'center',
  },
  similarityLabel: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 12,
  },
  similarityScore: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#2ecc71',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  similarityPercentage: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
  },
  phaseScoresContainer: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 16,
  },
  phaseRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  phaseLabel: {
    fontSize: 16,
    color: '#2c3e50',
    width: 100,
  },
  scoreContainer: {
    flex: 1,
    height: 24,
    backgroundColor: '#ecf0f1',
    borderRadius: 12,
    overflow: 'hidden',
    position: 'relative',
  },
  scoreBar: {
    height: '100%',
    borderRadius: 12,
  },
  scoreText: {
    position: 'absolute',
    right: 8,
    top: 2,
    fontSize: 12,
    fontWeight: 'bold',
    color: '#2c3e50',
  },
  feedbackContainer: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  feedbackItem: {
    marginBottom: 8,
  },
  feedbackText: {
    fontSize: 16,
    color: '#2c3e50',
    lineHeight: 22,
  },
  actionButtons: {
    marginTop: 20,
    marginBottom: 40,
  },
  recordAgainButton: {
    backgroundColor: '#3498db',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  recordAgainText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
});

export default ResultsScreen;
