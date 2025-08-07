import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';


const ResultsScreen = ({ navigation }) => {
  useEffect(() => {
    // Simulate waiting for results
    const timer = setTimeout(() => {
      // Navigate to another screen or display results
      navigation.navigate('ResultsDisplay'); // Replace with your results screen
    }, 1); // Simulate a 5-second wait

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#4ECDC4" />
      <Text style={styles.text}>Processing Results...</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },
  text: {
    marginTop: 20,
    fontSize: 18,
    color: '#333',
  },
});

export default ResultsScreen;