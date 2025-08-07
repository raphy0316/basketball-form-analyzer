import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  Dimensions,
} from 'react-native';
import { CONFIG } from '../utils/config';

const { width, height } = Dimensions.get('window');

const PlayerSelectionScreen = ({ navigation }) => {
  const [selectedPlayer, setSelectedPlayer] = useState(null);

  const handlePlayerSelect = (player) => {
    setSelectedPlayer(player);
  };

  const handleContinue = () => {
    if (selectedPlayer) {
      navigation.navigate('Main', { selectedPlayer });
    }
  };

  const renderPlayerCard = (player) => {
    const isSelected = selectedPlayer?.id === player.id;
    
    return (
      <TouchableOpacity
        key={player.id}
        style={[styles.playerCard, isSelected && styles.selectedPlayerCard]}
        onPress={() => handlePlayerSelect(player)}
      >
        <View style={styles.playerImageContainer}>
          <View style={styles.playerImagePlaceholder}>
            <Text style={styles.playerInitials}>
              {player.name.split(' ').map(n => n[0]).join('')}
            </Text>
          </View>
        </View>
        
        <View style={styles.playerInfo}>
          <Text style={[styles.playerName, isSelected && styles.selectedPlayerName]}>
            {player.name}
          </Text>
          <Text style={[styles.playerDescription, isSelected && styles.selectedPlayerDescription]}>
            {player.description}
          </Text>
          
          <View style={styles.characteristicsContainer}>
            {player.characteristics.map((char, index) => (
              <View key={index} style={styles.characteristicTag}>
                <Text style={styles.characteristicText}>{char}</Text>
              </View>
            ))}
          </View>
        </View>
        
        {isSelected && (
          <View style={styles.selectedIndicator}>
            <Text style={styles.selectedIndicatorText}>✓</Text>
          </View>
        )}
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Choose Your NBA Player</Text>
        <Text style={styles.subtitle}>
          Select a player to compare your shooting form with their style
        </Text>
      </View>

      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <View style={styles.playersContainer}>
          {CONFIG.SYNTHETIC.PLAYERS.map(renderPlayerCard)}
        </View>
      </ScrollView>

      <View style={styles.footer}>
        <TouchableOpacity
          style={[
            styles.continueButton,
            !selectedPlayer && styles.disabledButton
          ]}
          onPress={handleContinue}
          disabled={!selectedPlayer}
        >
          <Text style={styles.continueButtonText}>
            {selectedPlayer 
              ? `Continue with ${selectedPlayer.name}` 
              : 'Select a player to continue'
            }
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    padding: 20,
    paddingTop: 40,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    lineHeight: 22,
  },
  scrollView: {
    flex: 1,
  },
  playersContainer: {
    padding: 20,
  },
  playerCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  selectedPlayerCard: {
    borderColor: '#007AFF',
    backgroundColor: '#1a1a1a',
  },
  playerImageContainer: {
    marginRight: 16,
  },
  playerImagePlaceholder: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  playerInitials: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  playerInfo: {
    flex: 1,
  },
  playerName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 4,
  },
  selectedPlayerName: {
    color: '#007AFF',
  },
  playerDescription: {
    fontSize: 14,
    color: '#888',
    marginBottom: 8,
  },
  selectedPlayerDescription: {
    color: '#007AFF',
  },
  characteristicsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  characteristicTag: {
    backgroundColor: '#333',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  characteristicText: {
    fontSize: 12,
    color: '#ccc',
  },
  selectedIndicator: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  selectedIndicatorText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  footer: {
    padding: 20,
    paddingBottom: 40,
  },
  continueButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#333',
  },
  continueButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default PlayerSelectionScreen;
