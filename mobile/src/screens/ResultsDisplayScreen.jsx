import React, {useEffect} from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { initializeTtsListeners, playTTS } from '../utils/ttsListener';

const fetchResult = async () => {

}

const ResultsDisplayScreen = ({ navigation, route }) => {
  const { results } = route.params || {}; // Receive results from navigation params

  useEffect(() => {
    initializeTtsListeners();

    setTimeout(() => {
      playTTS('Hello World! This is text to speech implementation, Keep Coding!!!.'); // or Tts.speak(message)
    }, 1000);
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Results Display</Text>
      <Text style={styles.subtitle}>
        {results ? JSON.stringify(results, null, 2) : 'No results available'}
      </Text>

      {/* Back to Landing Button */}
      <TouchableOpacity
        style={styles.backButton}
        onPress={() => navigation.navigate('Landing')}
      >
        <Text style={styles.backButtonText}>Back to Landing</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1E1E1E',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 16,
    color: '#AAAAAA',
    textAlign: 'center',
    marginBottom: 30,
  },
  backButton: {
    backgroundColor: '#4ECDC4',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 8,
  },
  backButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
});

export default ResultsDisplayScreen;