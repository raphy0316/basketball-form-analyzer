import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
  SafeAreaView,
} from 'react-native';
import * as ExpoCamera from 'expo-camera';
import { Video } from 'expo-video';
import axios from 'axios';
import { CONFIG, getApiUrl, getVideoQuality } from './config';

const { width, height } = Dimensions.get('window');

const CameraScreen = () => {
  const [hasPermission, setHasPermission] = useState(null);
  const [camera, setCamera] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [showPreview, setShowPreview] = useState(false);
  
  const recordingTimerRef = useRef(null);
  const videoRef = useRef(null);



  useEffect(() => {
    (async () => {
      try {
        const { status } = await ExpoCamera.Camera.requestCameraPermissionsAsync();
        const audioStatus = await ExpoCamera.Camera.requestMicrophonePermissionsAsync();
        const hasBothPermissions = status === 'granted' && audioStatus.status === 'granted';
        setHasPermission(hasBothPermissions);
      } catch (error) {
        console.error('Error requesting permissions:', error);
        setHasPermission(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= CONFIG.RECORDING.MAX_DURATION) {
            stopRecording();
            return prev;
          }
          return prev + 0.1;
        });
      }, CONFIG.UI.RECORDING_TIMER_INTERVAL);
    } else {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    }

    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, [isRecording]);

  const startRecording = async () => {
    if (!camera) return;

    try {
      setIsRecording(true);
      setRecordingTime(0);
      setRecordedVideo(null);
      setShowPreview(false);

      // Start recording using the new CameraView API
      await camera.startRecording({
        quality: '720p',
        maxDuration: CONFIG.RECORDING.MAX_DURATION,
        mute: CONFIG.RECORDING.MUTE,
        onRecordingFinished: (video) => {
          setRecordedVideo(video);
          setShowPreview(true);
          setIsRecording(false);
        },
        onRecordingError: (error) => {
          console.error('Recording error:', error);
          Alert.alert('Error', 'Failed to record video. Please try again.');
          setIsRecording(false);
        },
      });
    } catch (error) {
      console.error('Error starting recording:', error);
      Alert.alert('Error', 'Failed to start recording. Please try again.');
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    if (camera && isRecording) {
      try {
        await camera.stopRecording();
      } catch (error) {
        console.error('Error stopping recording:', error);
        setIsRecording(false);
      }
    }
  };

  const retakeVideo = () => {
    setRecordedVideo(null);
    setShowPreview(false);
    setRecordingTime(0);
  };

  const analyzeVideo = async () => {
    if (!recordedVideo) return;

    setIsAnalyzing(true);
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', {
        uri: recordedVideo.uri,
        type: 'video/mp4',
        name: 'basketball_shot.mp4',
      });

      // Send to backend
      const response = await axios.post(
        getApiUrl(CONFIG.BACKEND.ENDPOINTS.ANALYZE_VIDEO),
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: CONFIG.BACKEND.TIMEOUT,
        }
      );

      Alert.alert(
        'Analysis Complete',
        'Your basketball shot has been analyzed successfully!',
        [
          {
            text: 'OK',
            onPress: () => {
              setRecordedVideo(null);
              setShowPreview(false);
              setRecordingTime(0);
            },
          },
        ]
      );
    } catch (error) {
      console.error('Error analyzing video:', error);
      Alert.alert(
        'Analysis Failed',
        'Failed to analyze your shot. Please try again.',
        [
          {
            text: 'OK',
            onPress: () => {
              setRecordedVideo(null);
              setShowPreview(false);
              setRecordingTime(0);
            },
          },
        ]
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading camera...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No access to camera</Text>
        <Text style={styles.errorSubtext}>
          Please enable camera permissions in your device settings.
        </Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => {
            setHasPermission(null);
            // Retry permission request
            (async () => {
              try {
                const { status } = await ExpoCamera.Camera.requestCameraPermissionsAsync();
                const audioStatus = await ExpoCamera.Camera.requestMicrophonePermissionsAsync();
                setHasPermission(status === 'granted' && audioStatus.status === 'granted');
              } catch (error) {
                console.error('Error requesting permissions:', error);
                setHasPermission(false);
              }
            })();
          }}
        >
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (showPreview && recordedVideo) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.previewContainer}>
          <Video
            ref={videoRef}
            source={{ uri: recordedVideo.uri }}
            style={styles.previewVideo}
            useNativeControls
            resizeMode="contain"
            isLooping
          />
          
          <View style={styles.previewControls}>
            <TouchableOpacity
              style={[styles.button, styles.retakeButton]}
              onPress={retakeVideo}
              disabled={isAnalyzing}
            >
              <Text style={styles.buttonText}>Retake</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.button, styles.analyzeButton]}
              onPress={analyzeVideo}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <ActivityIndicator size="small" color="white" />
              ) : (
                <Text style={styles.buttonText}>Analyze Shot</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  // Safety check for Camera component
  if (!ExpoCamera.CameraView) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Camera not available</Text>
        <Text style={styles.errorSubtext}>
          Please restart the app and try again.
        </Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ExpoCamera.CameraView
        style={styles.camera}
        facing="back"
        ref={(ref) => setCamera(ref)}
      />
      
      {/* Overlay positioned absolutely */}
      <View style={styles.overlay}>
        {/* Recording indicator */}
        {isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>
              Recording... {recordingTime.toFixed(1)}s
            </Text>
          </View>
        )}

        {/* Instructions */}
        <View style={styles.instructionsContainer}>
          <Text style={styles.instructionsTitle}>
            Basketball Form Analyzer
          </Text>
          <Text style={styles.instructionsText}>
            Position yourself in the frame and tap record to capture your shot
          </Text>
        </View>

        {/* Recording button */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[
              styles.recordButton,
              isRecording && styles.recordingButton,
            ]}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={isAnalyzing}
          >
            {isRecording ? (
              <View style={styles.stopIcon} />
            ) : (
              <View style={styles.recordIcon} />
            )}
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  recordingIndicator: {
    position: 'absolute',
    top: 60,
    alignSelf: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#FF3B30',
    marginRight: 8,
  },
  recordingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  instructionsContainer: {
    position: 'absolute',
    top: 120,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  instructionsTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  instructionsText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    opacity: 0.9,
  },
  buttonContainer: {
    alignItems: 'center',
    marginBottom: 50,
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#007AFF',
  },
  recordingButton: {
    backgroundColor: '#FF3B30',
    borderColor: '#FF3B30',
  },
  recordIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#007AFF',
  },
  stopIcon: {
    width: 24,
    height: 24,
    backgroundColor: 'white',
    borderRadius: 2,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  previewVideo: {
    flex: 1,
  },
  previewControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
  },
  button: {
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    minWidth: 120,
    alignItems: 'center',
  },
  retakeButton: {
    backgroundColor: '#8E8E93',
  },
  analyzeButton: {
    backgroundColor: '#007AFF',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  errorText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 100,
  },
  errorSubtext: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 20,
    paddingHorizontal: 40,
    opacity: 0.8,
  },
  loadingText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 20,
  },
  retryButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
    marginTop: 30,
    alignSelf: 'center',
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default CameraScreen;
