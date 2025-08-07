// // Keypoint names for reference
export const KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
  ];
//   // Minimum confidence score for keypoints to be considered valid
  export const MIN_KEYPOINT_CONFIDENCE = 0.3;
  
//   /**
//    * Maps keypoints from normalized coordinates to screen coordinates.
//    * @param {Array} keypoints - Array of keypoints with normalized coordinates.
//    * @param {number} screenWidth - Width of the screen.
//    * @param {number} screenHeight - Height of the screen.
//    * @returns {Array} - Array of keypoints mapped to screen coordinates.
//    */
//   export const mapKeypointsToScreen = (keypoints, screenWidth, screenHeight) => {
//     return keypoints.map((keypoint) => {
//       const screenAspect = screenWidth / screenHeight;
//       let scale, offsetX = 0, offsetY = 0;
  
//       if (screenAspect > 1) {
//         // Wider screen (landscape) → fit height
//         scale = screenHeight;
//         offsetX = (screenWidth - screenHeight) / 2;
//       } else {
//         // Taller screen (portrait) → fit width
//         scale = screenWidth;
//         offsetY = (screenHeight - screenWidth) / 2;
//       }
  
//       return {
//         ...keypoint,
//         x: keypoint.x * scale + offsetX,
//         y: keypoint.y * scale + offsetY,
//       };
//     });
//   };
  
//   /**
//    * Filters keypoints based on confidence score.
//    * @param {Array} keypoints - Array of keypoints.
//    * @param {number} minConfidence - Minimum confidence score to include a keypoint.
//    * @returns {Array} - Filtered array of keypoints.
//    */
//   export const filterKeypointsByConfidence = (keypoints, minConfidence = MIN_KEYPOINT_CONFIDENCE) => {
//     return keypoints.filter((keypoint) => keypoint.confidence >= minConfidence);
//   };
  
//   /**
//    * Calculates the average confidence of a set of keypoints.
//    * @param {Array} keypoints - Array of keypoints.
//    * @returns {number} - Average confidence score.
//    */
//   export const calculateAverageConfidence = (keypoints) => {
//     if (keypoints.length === 0) return 0;
//     const totalConfidence = keypoints.reduce((sum, keypoint) => sum + keypoint.confidence, 0);
//     return totalConfidence / keypoints.length;
//   };
  
//   /**
//    * Gets the color for a keypoint based on its name.
//    * @param {string} name - Name of the keypoint.
//    * @returns {string} - Color for the keypoint.
//    */
//   export const getKeypointColor = (name) => {
//     if (name.includes('eye') || name.includes('ear') || name === 'nose') return '#FF6B6B'; // Red for face
//     if (name.includes('shoulder') || name.includes('elbow') || name.includes('wrist')) return '#4ECDC4'; // Teal for arms
//     if (name.includes('hip') || name.includes('knee') || name.includes('ankle')) return '#45B7D1'; // Blue for legs
//     return '#96CEB4'; // Default green
//   };