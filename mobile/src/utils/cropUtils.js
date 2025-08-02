
const MIN_CROP_KEYPOINT_SCORE = 0.3;

function initCropRegion(height, width) {
    'worklet'
    if (width > height) {
      const boxHeight = width / height;
      return {
        yMin: Math.max(0.0, (1.0 - boxHeight) / 2),
        xMin: 0.0,
        yMax: Math.max(0.0, (1.0 - boxHeight) / 2) + boxHeight,
        xMax: 1.0,
        height: boxHeight,
        width: 1.0,
      };
    } else {
      const boxWidth = height / width;
      return {
        yMin: 0.0,
        xMin: Math.max(0.0, (1.0 - boxWidth) / 2),
        yMax: 1.0,
        xMax: Math.max(0.0, (1.0 - boxWidth) / 2) + boxWidth,
        height: 1.0,
        width: boxWidth,
      };
    }
  }
  
function torsoVisible(keypoints) {
    'worklet'
    const score = (name) => keypoints[name]?.confidence || 0;
    return (
      (score('left_hip') > MIN_CROP_KEYPOINT_SCORE || score('right_hip') > MIN_CROP_KEYPOINT_SCORE) &&
      (score('left_shoulder') > MIN_CROP_KEYPOINT_SCORE || score('right_shoulder') > MIN_CROP_KEYPOINT_SCORE)
    );
}

function determineCropRegion(keypoints, height, width) {
    'worklet'
    const jointPos = (name) => ({
      y: keypoints[name].y * height,
      x: keypoints[name].x * width,
    });
  
    const torsoJoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'];
  
    const centerY = (jointPos('left_hip').y + jointPos('right_hip').y) / 2;
    const centerX = (jointPos('left_hip').x + jointPos('right_hip').x) / 2;
  
    let maxTorsoY = 0, maxTorsoX = 0, maxBodyY = 0, maxBodyX = 0;
  
    torsoJoints.forEach(j => {
      const { y, x } = jointPos(j);
      maxTorsoY = Math.max(maxTorsoY, Math.abs(centerY - y));
      maxTorsoX = Math.max(maxTorsoX, Math.abs(centerX - x));
    });
  
    Object.keys(keypoints).forEach(j => {
      if (keypoints[j].confidence < MIN_CROP_KEYPOINT_SCORE) return;
      const { y, x } = jointPos(j);
      maxBodyY = Math.max(maxBodyY, Math.abs(centerY - y));
      maxBodyX = Math.max(maxBodyX, Math.abs(centerX - x));
    });
  
    const cropHalfLen = Math.min(
      Math.max(
        1.9 * Math.max(maxTorsoX, maxTorsoY),
        1.2 * Math.max(maxBodyY, maxBodyX)
      ),
      Math.max(centerX, width - centerX, centerY, height - centerY)
    );
  
    if (cropHalfLen > Math.max(width, height) / 2) {
      return initCropRegion(height, width);
    }
  
    return {
      yMin: (centerY - cropHalfLen) / height,
      xMin: (centerX - cropHalfLen) / width,
      yMax: (centerY + cropHalfLen) / height,
      xMax: (centerX + cropHalfLen) / width,
      height: (2 * cropHalfLen) / height,
      width: (2 * cropHalfLen) / width,
    };
  }

export {initCropRegion, determineCropRegion, torsoVisible};

// const adjustBoxForLetterbox = (box, padInfo) => {
//   'worklet';
//   return {
//     x: (box.x - padInfo.padLeft) / padInfo.scale,
//     y: (box.y - padInfo.padTop) / padInfo.scale,
//     width: box.width / padInfo.scale,
//     height: box.height / padInfo.scale,
//     confidence: box.confidence,
//     classId: box.classId,
//   };
// };

// const resizeWithLetterbox = (frame, resize, targetSize = 736) => {
//   'worklet';

//   const inputW = frame.width;
//   const inputH = frame.height;
//   const scale = Math.min(targetSize / inputW, targetSize / inputH);

//   // Compute padding to fit into square
//   const newW = inputW * scale;
//   const newH = inputH * scale;

//   const padW = targetSize - newW;
//   const padH = targetSize - newH;

//   // Center the original image with padding
//   const crop = {
//     x: 0,
//     y: 0,
//     width: inputW,
//     height: inputH
//   };

//   const resized = resize(frame, {
//     scale: { width: targetSize, height: targetSize },
//     crop,
//     pixelFormat: 'rgb',
//     dataType: 'float32', // this automatically normalizes to [0.0–1.0]
//   });

//   // Save pad ratios if you want to adjust detections later
//   const padInfo = {
//     scale,
//     padLeft: padW / 2 / targetSize,
//     padTop: padH / 2 / targetSize,
//   };

//   return [resized, padInfo];
// };