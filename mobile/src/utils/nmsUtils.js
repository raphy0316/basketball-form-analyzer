export const applyNMS = (boxes, iouThreshold = 0.15) => {
    'worklet';
    const picked = [];
  
    // Sort boxes by confidence in descending order
    boxes.sort((a, b) => b.confidence - a.confidence);
  
    for (let i = 0; i < boxes.length; i++) {
      const boxA = boxes[i];
      let keep = true;
  
      for (let j = 0; j < picked.length; j++) {
        const boxB = picked[j];
  
        // Calculate intersection
        const x1 = Math.max(boxA.x, boxB.x);
        const y1 = Math.max(boxA.y, boxB.y);
        const x2 = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
        const y2 = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
  
        const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  
        // Calculate union
        const areaA = boxA.width * boxA.height;
        const areaB = boxB.width * boxB.height;
        const union = areaA + areaB - interArea;
  
        // Calculate IoU
        const iou = union === 0 ? 0 : interArea / union;
  
        if (iou > iouThreshold) {
          keep = false;
          break;
        }
      }
  
      if (keep) picked.push(boxA);
    }
    return picked;
  };