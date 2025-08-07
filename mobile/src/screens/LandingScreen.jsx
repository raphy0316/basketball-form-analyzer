import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';

const LandingScreen = ({ navigation }) => {
  const handleStartFilming = () => {
    navigation.navigate('Main'); // Navigate to the MainScreen
  };

  return (
    <View style={styles.container}>
      {/* Logo or Banner */}
      <Image
        source={require('../assets/landing.png')}
        style={styles.logo}
        resizeMode="contain"
      />

      {/* Title */}
      <Text style={styles.title}>Basketball Form Analyzer</Text>

      {/* Subtitle */}
      <Text style={styles.subtitle}>
        Analyze your basketball shooting form with real-time feedback.
      </Text>

      {/* Start Filming Button */}
      <TouchableOpacity style={styles.startButton} onPress={handleStartFilming}>
        <Text style={styles.startButtonText}>Start Filming</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1E1E1E', // Dark background
    padding: 20,
  },
  logo: {
    width: 200,
    height: 200,
    marginBottom: 20,
    
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
    textAlign: 'center',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#AAAAAA',
    textAlign: 'center',
    marginBottom: 30,
  },
  startButton: {
    backgroundColor: '#4ECDC4', // Teal color
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 8,
  },
  startButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
});

export default LandingScreen;