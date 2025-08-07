import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const Legend = () => {
  return (
    <View style={styles.legend}>
      <Text style={styles.legendTitle}>Detected:</Text>
      <View style={styles.legendRow}>
        <View style={[styles.legendColor, { backgroundColor: '#FF6B6B' }]} />
        <Text style={styles.legendText}>Face</Text>
      </View>
      <View style={styles.legendRow}>
        <View style={[styles.legendColor, { backgroundColor: '#4ECDC4' }]} />
        <Text style={styles.legendText}>Arms</Text>
      </View>
      <View style={styles.legendRow}>
        <View style={[styles.legendColor, { backgroundColor: '#45B7D1' }]} />
        <Text style={styles.legendText}>Legs</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  legend: {
    position: 'absolute',
    bottom: 100,
    right: 20,
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 12,
    borderRadius: 8,
  },
  legendTitle: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  legendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
    borderWidth: 1,
    borderColor: 'white',
  },
  legendText: {
    color: '#ccc',
    fontSize: 12,
  },
});

export default Legend;