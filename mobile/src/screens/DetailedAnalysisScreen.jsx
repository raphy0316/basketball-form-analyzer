import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
  Dimensions,
} from 'react-native';

const { width, height } = Dimensions.get('window');

const DetailedAnalysisScreen = ({ detailedResult, selectedPlayer }) => {
  // const { detailedResult, selectedPlayer } = route.params || {};
  console.log("detailed: ", detailedResult);
 
  const renderPhaseTransitions = () => {
    const transitions = detailedResult.comparison_result.interpretation.phase_transition_analysis || {};
    return (
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Phase Transitions</Text>
        <Text style={styles.phaseTransHeader}>User</Text>
        <Text style={styles.phaseTransText}>{(transitions.video1_pattern?.description || [])}</Text>
        <Text style={styles.phaseTransHeader}>{selectedPlayer.name}</Text>
        <Text style={styles.phaseTransText}>{(transitions.video2_pattern?.description || [])}</Text>
      </View>
    );
  };

  const renderInterpretation = () => {
    const interp = detailedResult?.comparison_result.interpretation || {};
    return (
      <View style={styles.sectionContainer}>
        <Text style={styles.sectionTitle}>Phase Interpretation</Text>
        {Object.entries(interp.text_analysis || {}).map(([phase, details]) => (
          <View key={phase} style={styles.interpPhaseBlock}>
            <Text style={styles.interpPhaseTitle}>{phase.charAt(0).toUpperCase() + phase.slice(1)}</Text>
            {details.differences && details.differences.map((diff, idx) => (
              <Text key={idx} style={styles.interpDiffText}>• {diff}</Text>
            ))}
          </View>
        ))}
        {/* Phase transition analysis */}
        {interp.phase_transition_analysis && (
          <View style={styles.interpPhaseBlock}>
            <Text style={styles.interpPhaseTitle}>Phase Transition Analysis</Text>
            <Text style={styles.interpDiffText}>User: {interp.phase_transition_analysis.video1_pattern?.description}</Text>
            <Text style={styles.interpDiffText}>{selectedPlayer.name}: {interp.phase_transition_analysis.video2_pattern?.description}</Text>
            {interp.phase_transition_analysis.comparison?.differences?.map((diff, idx) => (
              <Text key={idx} style={styles.interpDiffText}>• {diff}</Text>
            ))}
            {interp.phase_transition_analysis.comparison?.recommendations?.map((rec, idx) => (
              <Text key={idx} style={styles.interpRecText}>Recommendation: {rec}</Text>
            ))}
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <Text style={styles.title}>Detailed Analysis</Text>
          {selectedPlayer && (
            <Text style={styles.subtitle}>
              Compared to {selectedPlayer.name}'s style
            </Text>
          )}
        </View>

        {renderPhaseTransitions()}
        {renderInterpretation()}

        <View style={styles.llmContainer}>
          <Text style={styles.llmTitle}>Coach's advice</Text>
          <Text style={styles.llmText}>{detailedResult?.llm_response || 'No response available.'}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  scrollView: { flex: 1 },
  header: { padding: 20, alignItems: 'center' },
  title: { fontSize: 28, fontWeight: 'bold', color: 'white', textAlign: 'center', marginBottom: 8 },
  subtitle: { fontSize: 16, color: '#888', textAlign: 'center' },
  sectionContainer: { backgroundColor: '#1a1a1a', margin: 20, padding: 16, borderRadius: 12 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', color: 'white', marginBottom: 12 },
  phaseStatsRow: { flexDirection: 'row', justifyContent: 'space-between' },
  phaseStatsHeader: { fontSize: 16, color: '#4ECDC4', marginBottom: 6 },
  phaseStatsText: { fontSize: 14, color: 'white', marginBottom: 2 },
  phaseTransHeader: { fontSize: 15, color: '#4ECDC4', marginTop: 8 },
  phaseTransText: { fontSize: 13, color: 'white', marginBottom: 4 },
  interpPhaseBlock: { marginBottom: 12 },
  interpPhaseTitle: { fontSize: 16, color: '#FF6347', marginBottom: 4 },
  interpDiffText: { fontSize: 13, color: '#ccc', marginLeft: 8, marginBottom: 2 },
  interpRecText: { fontSize: 13, color: '#4ECDC4', marginLeft: 8, marginBottom: 2 },
  llmContainer: { backgroundColor: '#1a1a1a', margin: 20, padding: 16, borderRadius: 12 },
  llmTitle: { fontSize: 18, fontWeight: 'bold', color: 'white', marginBottom: 8 },
  llmText: { fontSize: 14, color: '#ccc' },
  actionsContainer: { padding: 20, gap: 12 },
  actionButton: { backgroundColor: '#007AFF', paddingVertical: 16, paddingHorizontal: 24, borderRadius: 12, alignItems: 'center' },
  actionButtonText: { color: 'white', fontSize: 16, fontWeight: '600' },
});

export default DetailedAnalysisScreen;