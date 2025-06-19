package jp.jaxa.iss.kibo.rpc.sampleapk;

import ai.onnxruntime.*;
import android.content.Context;
import android.util.Log;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * Enhanced YOLO Object Detection Service with intelligent NMS
 * Matches functionality of Python yoloraw_postprocessing.py
 */
public class YOLODetectionService {
    private static final String TAG = "YOLODetectionService";
    private static final String MODEL_NAME = "best.onnx";
    private static final int INPUT_SIZE = 320;
    private static final float DEFAULT_CONF_THRESHOLD = 0.3f;
    private static final float DEFAULT_STANDARD_NMS_THRESHOLD = 0.45f;
    private static final float DEFAULT_OVERLAP_NMS_THRESHOLD = 0.8f;

    // Class definitions matching Python code
    private static final String[] CLASS_NAMES = {
            "coin", "compass", "coral", "crystal", "diamond", "emerald",
            "fossil", "key", "letter", "shell", "treasure_box"
    };

    private static final Set<Integer> TREASURE_IDS = new HashSet<Integer>(Arrays.asList(3, 4, 5)); // crystal, diamond, emerald
    private static final Set<Integer> LANDMARK_IDS = new HashSet<Integer>(Arrays.asList(0, 1, 2, 6, 7, 8, 9, 10)); // coin,
    // compass,
    // coral,
    // fossil,
    // key,
    // letter,
    // shell,
    // treasure_box

    private OrtEnvironment env;
    private OrtSession session;
    private Context context;
    private boolean isInitialized = false;

    public YOLODetectionService(Context context) {
        this.context = context;
        initializeModel();
    }

    private void initializeModel() {
        try {
            Log.i(TAG, "Initializing YOLO model...");

            env = OrtEnvironment.getEnvironment();
            File modelFile = copyAssetToFile(MODEL_NAME);

            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

            session = env.createSession(modelFile.getAbsolutePath(), sessionOptions);
            isInitialized = true;
            Log.i(TAG, "YOLO model initialized successfully");

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize YOLO model: " + e.getMessage(), e);
            isInitialized = false;
        }
    }

    private File copyAssetToFile(String assetName) throws IOException {
        InputStream inputStream = context.getAssets().open(assetName);
        File outputFile = new File(context.getFilesDir(), assetName);

        FileOutputStream outputStream = new FileOutputStream(outputFile);
        byte[] buffer = new byte[4096];
        int length;
        while ((length = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, length);
        }

        outputStream.close();
        inputStream.close();

        return outputFile;
    }

    /**
     * Main detection method that matches Python simple_detection_example
     * functionality
     * * @param image                OpenCV Mat image
     * @param imageType            "lost" or "target"
     * @param confThreshold        confidence threshold (default: 0.3)
     * @param standardNmsThreshold standard NMS threshold (default: 0.45)
     * @param overlapNmsThreshold  overlap NMS threshold for intelligent NMS
     * (default: 0.8)
     * @return EnhancedDetectionResult with treasure and landmark quantities
     */
    public EnhancedDetectionResult DetectfromcvImage(Mat image, String imageType,
                                                     float confThreshold,
                                                     float standardNmsThreshold,
                                                     float overlapNmsThreshold) {
        if (!isInitialized) {
            Log.e(TAG, "YOLO model not initialized");
            return new EnhancedDetectionResult();
        }

        try {
            Log.i(TAG, "Starting detection for image type: " + imageType);

            // Preprocess image
            Mat preprocessedImage = preprocessImage(image);
            float[][][][] inputData = matToFloatArray(preprocessedImage);

            // Run inference to get raw tensor
            Map<String, OnnxTensor> inputMap = new HashMap<String, OnnxTensor>();
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
            inputMap.put("images", inputTensor);

            OrtSession.Result result = session.run(inputMap);
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            float[][][] rawOutput = (float[][][]) outputTensor.getValue();

            // Apply intelligent post-processing pipeline
            EnhancedDetectionResult detectionResult = yoloPostprocessPipeline(
                    rawOutput, confThreshold, standardNmsThreshold, overlapNmsThreshold,
                    INPUT_SIZE, imageType, image.width(), image.height());

            // Clean up
            inputTensor.close();
            result.close();
            preprocessedImage.release();

            Log.i(TAG, String.format("Detection completed for %s image", imageType));
            detectionResult.logResults(TAG);

            return detectionResult;

        } catch (Exception e) {
            Log.e(TAG, "Detection failed: " + e.getMessage(), e);
            return new EnhancedDetectionResult();
        }
    }

    /**
     * Convenience method with default parameters
     */
    public EnhancedDetectionResult DetectfromcvImage(Mat image, String imageType) {
        return DetectfromcvImage(image, imageType, DEFAULT_CONF_THRESHOLD,
                DEFAULT_STANDARD_NMS_THRESHOLD, DEFAULT_OVERLAP_NMS_THRESHOLD);
    }

    /**
     * Compatibility method for existing code that expects simple class counts
     * * @param image OpenCV Mat image
     * @return Map of class ID to count using "lost" detection logic
     */
    public Map<Integer, Integer> getItemCounts(Mat image) {
        EnhancedDetectionResult result = DetectfromcvImage(image, "lost");
        return result.getAllQuantities();
    }

    /**
     * Get class names array for external use
     * * @return Array of class names
     */
    public static String[] getClassNames() {
        return CLASS_NAMES.clone();
    }

    /**
     * Get class name by ID
     * * @param classId Class ID (0-based)
     * @return Class name or null if invalid ID
     */
    public static String getClassName(int classId) {
        if (classId >= 0 && classId < CLASS_NAMES.length) {
            return CLASS_NAMES[classId];
        }
        return null;
    }

    /**
     * Enhanced post-processing pipeline matching Python logic
     */
    private EnhancedDetectionResult yoloPostprocessPipeline(float[][][] rawTensor,
                                                            float confThreshold,
                                                            float standardNmsThreshold,
                                                            float overlapNmsThreshold,
                                                            int imgSize,
                                                            String imgType,
                                                            int originalWidth,
                                                            int originalHeight) {
        Log.i(TAG, String.format("Raw tensor shape: [%d, %d, %d]",
                rawTensor.length, rawTensor[0].length, rawTensor[0][0].length));

        // ====================================================================
        // CRITICAL FIX: Transpose tensor from [1, 15, 2100] to [1, 2100, 15]
        // This matches Python: processed_tensor = raw_tensor.transpose(1, 2)
        // ====================================================================

        float[][] processed;
        int numDetections, numFeatures;

        // Check if we need to transpose (matches Python logic)
        if (rawTensor[0].length < rawTensor[0][0].length) {
            // Need to transpose from [15, 2100] to [2100, 15]
            Log.i(TAG, "Transposing tensor from [15, 2100] to [2100, 15]");

            numDetections = rawTensor[0][0].length; // 2100
            numFeatures = rawTensor[0].length; // 15

            processed = new float[numDetections][numFeatures];

            // Transpose: processed[detection][feature] = rawTensor[0][feature][detection]
            for (int det = 0; det < numDetections; det++) {
                for (int feat = 0; feat < numFeatures; feat++) {
                    processed[det][feat] = rawTensor[0][feat][det];
                }
            }

        } else {
            // Already in correct format [2100, 15]
            Log.i(TAG, "Tensor already in correct format");
            processed = rawTensor[0];
            numDetections = processed.length;
            numFeatures = processed[0].length;
        }

        Log.i(TAG, String.format("Processing %d detection proposals with %d features each",
                numDetections, numFeatures));

        // Log min/max for each FEATURE across all detections (matches Python Layer
        // 0-14)
        Log.i(TAG, "Feature min/max values across all detections:");
        for (int featIdx = 0; featIdx < numFeatures; featIdx++) {
            float minValue = Float.MAX_VALUE;
            float maxValue = Float.MIN_VALUE;

            // Find min/max for this feature across all detections
            for (int detIdx = 0; detIdx < numDetections; detIdx++) {
                float value = processed[detIdx][featIdx];
                minValue = Math.min(minValue, value);
                maxValue = Math.max(maxValue, value);
            }

            // This should now match Python's "Layer X: min=..., max=..."
            Log.i(TAG, String.format("Layer %d: min=%.6f, max=%.6f",
                    featIdx, minValue, maxValue));
        }

        List<DetectionCandidate> candidates = new ArrayList<DetectionCandidate>();

        // Step 1: Extract all detection candidates above confidence threshold
        for (int i = 0; i < processed.length; i++) {
            float[] prediction = processed[i];

            if (prediction.length < 5)
                continue;

            // Extract bbox and class scores
            float centerX = prediction[0];
            float centerY = prediction[1];
            float width = prediction[2];
            float height = prediction[3];

            // Check all class scores
            for (int classId = 0; classId < CLASS_NAMES.length; classId++) {
                float classScore = prediction[4 + classId];

                if (classScore > confThreshold) {
                    // Scale coordinates back to original image size
                    float scaleX = (float) originalWidth / imgSize;
                    float scaleY = (float) originalHeight / imgSize;

                    float scaledCenterX = centerX * scaleX;
                    float scaledCenterY = centerY * scaleY;
                    float scaledWidth = width * scaleX;
                    float scaledHeight = height * scaleY;

                    candidates.add(new DetectionCandidate(
                            scaledCenterX, scaledCenterY, scaledWidth, scaledHeight,
                            classScore, classId));
                }
            }
        }

        Log.i(TAG, String.format("Total detection candidates: %d", candidates.size()));

        // Step 2: Separate treasure and landmark candidates
        List<DetectionCandidate> treasureCandidates = new ArrayList<DetectionCandidate>();
        List<DetectionCandidate> landmarkCandidates = new ArrayList<DetectionCandidate>();

        for (DetectionCandidate candidate : candidates) {
            if (TREASURE_IDS.contains(candidate.classId)) {
                treasureCandidates.add(candidate);
            } else if (LANDMARK_IDS.contains(candidate.classId)) {
                landmarkCandidates.add(candidate);
            }
        }

        Log.i(TAG, String.format("Treasure candidates: %d, Landmark candidates: %d",
                treasureCandidates.size(), landmarkCandidates.size()));

        // Step 3: Apply image type constraints with intelligent NMS
        return applyImageTypeConstraints(treasureCandidates, landmarkCandidates,
                imgType, standardNmsThreshold, overlapNmsThreshold);
    }

    private EnhancedDetectionResult applyImageTypeConstraints(List<DetectionCandidate> treasureCandidates,
                                                              List<DetectionCandidate> landmarkCandidates,
                                                              String imgType,
                                                              float standardNmsThreshold,
                                                              float overlapNmsThreshold) {
        List<FinalDetection> finalDetections = new ArrayList<FinalDetection>();
        Map<Integer, Integer> treasureQuantities = new HashMap<Integer, Integer>();
        Map<Integer, Integer> landmarkQuantities = new HashMap<Integer, Integer>();
        Map<Integer, Integer> allQuantities = new HashMap<Integer, Integer>();

        // New variable to store the highest confidence landmark
        FinalDetection highestConfLandmarkResult = null;

        if ("target".equals(imgType)) {
            Log.i(TAG, "TARGET ITEM logic - applying STANDARD NMS");

            // Apply standard NMS to both treasures and landmarks
            List<FinalDetection> treasureFinal = applyStandardNMS(treasureCandidates, standardNmsThreshold);
            List<FinalDetection> landmarkFinal = applyStandardNMS(landmarkCandidates, standardNmsThreshold);

            // Count quantities after NMS
            countQuantities(treasureFinal, treasureQuantities, allQuantities);
            countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

            // Java 7 compatible sort
            Collections.sort(treasureFinal, new Comparator<FinalDetection>() {
                @Override
                public int compare(FinalDetection a, FinalDetection b) {
                    return Float.compare(b.confidence, a.confidence);
                }
            });
            Collections.sort(landmarkFinal, new Comparator<FinalDetection>() {
                @Override
                public int compare(FinalDetection a, FinalDetection b) {
                    return Float.compare(b.confidence, a.confidence);
                }
            });

            // Select exactly 1 treasure + 2 different landmark types
            if (!treasureFinal.isEmpty() && landmarkFinal.size() >= 2) {
                finalDetections.add(treasureFinal.get(0));
                Log.i(TAG, String.format("Selected treasure: %s (conf: %.3f)",
                        CLASS_NAMES[treasureFinal.get(0).classId], treasureFinal.get(0).confidence));

                Set<Integer> selectedLandmarkClasses = new HashSet<Integer>();
                for (FinalDetection landmark : landmarkFinal) {
                    if (!selectedLandmarkClasses.contains(landmark.classId)) {
                        finalDetections.add(landmark);
                        selectedLandmarkClasses.add(landmark.classId);
                        Log.i(TAG, String.format("Selected landmark: %s (conf: %.3f)",
                                CLASS_NAMES[landmark.classId], landmark.confidence));

                        if (selectedLandmarkClasses.size() == 2)
                            break;
                    }
                }
            }

        } else if ("lost".equals(imgType)) {
            Log.i(TAG, "LOST ITEM logic - applying INTELLIGENT NMS");

            if (!treasureCandidates.isEmpty()) {
                // Case 1: 1 landmark + 1 treasure
                Log.i(TAG, "Case 1: Treasure + Landmark detected");

                List<FinalDetection> treasureFinal = applyStandardNMS(treasureCandidates, standardNmsThreshold);
                List<FinalDetection> landmarkFinal = applyLandmarkIntelligentNMS(landmarkCandidates,
                        overlapNmsThreshold);

                countQuantities(treasureFinal, treasureQuantities, allQuantities);
                countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

                // Java 7 compatible sort
                Collections.sort(treasureFinal, new Comparator<FinalDetection>() {
                    @Override
                    public int compare(FinalDetection a, FinalDetection b) {
                        return Float.compare(b.confidence, a.confidence);
                    }
                });
                Collections.sort(landmarkFinal, new Comparator<FinalDetection>() {
                    @Override
                    public int compare(FinalDetection a, FinalDetection b) {
                        return Float.compare(b.confidence, a.confidence);
                    }
                });

                if (!treasureFinal.isEmpty()) {
                    finalDetections.add(treasureFinal.get(0));
                    Log.i(TAG, String.format("Selected treasure: %s (conf: %.3f)",
                            CLASS_NAMES[treasureFinal.get(0).classId], treasureFinal.get(0).confidence));
                }

                if (!landmarkFinal.isEmpty()) {
                    FinalDetection bestLandmark = landmarkFinal.get(0);
                    finalDetections.add(bestLandmark);
                    Log.i(TAG, String.format("Selected landmark: %s (conf: %.3f)",
                            CLASS_NAMES[bestLandmark.classId], bestLandmark.confidence));

                    // 儲存最高信心 landmark 結果
                    highestConfLandmarkResult = bestLandmark;

                } else if (landmarkFinal.isEmpty() && !landmarkCandidates.isEmpty()) {
                    // Fallback to highest confidence candidate if intelligent NMS removed all
                    // Java 7 compatible sort for candidates
                    Collections.sort(landmarkCandidates, new Comparator<DetectionCandidate>() {
                        @Override
                        public int compare(DetectionCandidate a, DetectionCandidate b) {
                            return Float.compare(b.confidence, a.confidence);
                        }
                    });
                    DetectionCandidate best = landmarkCandidates.get(0);
                    FinalDetection fallback = new FinalDetection(
                            best.centerX, best.centerY, best.width, best.height,
                            best.confidence, best.classId);

                    finalDetections.add(fallback);
                    highestConfLandmarkResult = fallback; // 儲存 fallback 結果

                    Log.i(TAG, String.format("Fallback landmark selected: %s (conf: %.3f)",
                            CLASS_NAMES[best.classId], best.confidence));
                }


            } else {
                // Case 2: Only landmarks
                Log.i(TAG, "Case 2: Only landmarks detected");

                List<FinalDetection> landmarkFinal = applyLandmarkIntelligentNMS(landmarkCandidates,
                        overlapNmsThreshold);
                countQuantities(landmarkFinal, landmarkQuantities, allQuantities);

                // Java 7 compatible sort
                Collections.sort(landmarkFinal, new Comparator<FinalDetection>() {
                    @Override
                    public int compare(FinalDetection a, FinalDetection b) {
                        return Float.compare(b.confidence, a.confidence);
                    }
                });

                if (!landmarkFinal.isEmpty()) {
                    FinalDetection bestLandmark = landmarkFinal.get(0);
                    finalDetections.add(bestLandmark);
                    Log.i(TAG, String.format("Selected landmark: %s (conf: %.3f)",
                            CLASS_NAMES[bestLandmark.classId], bestLandmark.confidence));

                    // 儲存最高信心 landmark 結果
                    highestConfLandmarkResult = bestLandmark;

                } else if (landmarkFinal.isEmpty() && !landmarkCandidates.isEmpty()) {
                    // Fallback to highest confidence candidate if intelligent NMS removed all
                    // Java 7 compatible sort for candidates
                    Collections.sort(landmarkCandidates, new Comparator<DetectionCandidate>() {
                        @Override
                        public int compare(DetectionCandidate a, DetectionCandidate b) {
                            return Float.compare(b.confidence, a.confidence);
                        }
                    });
                    DetectionCandidate best = landmarkCandidates.get(0);
                    FinalDetection fallback = new FinalDetection(
                            best.centerX, best.centerY, best.width, best.height,
                            best.confidence, best.classId);

                    finalDetections.add(fallback);
                    highestConfLandmarkResult = fallback; // 儲存 fallback 結果

                    Log.i(TAG, String.format("Fallback landmark selected: %s (conf: %.3f)",
                            CLASS_NAMES[best.classId], best.confidence));
                }

            }
        }

        return new EnhancedDetectionResult(finalDetections, allQuantities, treasureQuantities, landmarkQuantities, highestConfLandmarkResult);
    }

    private List<FinalDetection> applyStandardNMS(List<DetectionCandidate> candidates, float nmsThreshold) {
        if (candidates.size() <= 1) {
            return convertToFinalDetections(candidates);
        }

        // Java 7 compatible sort
        Collections.sort(candidates, new Comparator<DetectionCandidate>() {
            @Override
            public int compare(DetectionCandidate a, DetectionCandidate b) {
                return Float.compare(b.confidence, a.confidence);
            }
        });

        List<DetectionCandidate> kept = new ArrayList<DetectionCandidate>();
        boolean[] suppressed = new boolean[candidates.size()];

        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i])
                continue;

            DetectionCandidate current = candidates.get(i);
            kept.add(current);

            for (int j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j])
                    continue;

                DetectionCandidate other = candidates.get(j);
                if (calculateIoU(current, other) > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return convertToFinalDetections(kept);
    }

    private List<FinalDetection> applyLandmarkIntelligentNMS(List<DetectionCandidate> candidates,
                                                             float overlapThreshold) {
        if (candidates.size() <= 1) {
            return convertToFinalDetections(candidates);
        }

        Log.i(TAG, String.format("Applying intelligent NMS to %d landmark detections", candidates.size()));

        // Find highest confidence detection and its class (Java 7 compatible)
        DetectionCandidate highest = null;
        if (!candidates.isEmpty()) {
            highest = candidates.get(0);
            for (int i = 1; i < candidates.size(); i++) {
                DetectionCandidate currentCandidate = candidates.get(i);
                if (currentCandidate.confidence > highest.confidence) {
                    highest = currentCandidate;
                }
            }
        }

        if (highest == null)
            return new ArrayList<FinalDetection>();

        int selectedClass = highest.classId;
        Log.i(TAG, String.format("Selected class: %d (%s) with confidence: %.3f",
                selectedClass, CLASS_NAMES[selectedClass], highest.confidence));

        // Filter to only detections of the selected class
        List<DetectionCandidate> sameClassCandidates = new ArrayList<DetectionCandidate>();
        for (DetectionCandidate candidate : candidates) {
            if (candidate.classId == selectedClass) {
                sameClassCandidates.add(candidate);
            }
        }

        Log.i(TAG, String.format("Detections of selected class: %d/%d",
                sameClassCandidates.size(), candidates.size()));

        // Apply standard NMS with overlap threshold to same-class detections
        List<FinalDetection> result = applyStandardNMS(sameClassCandidates, overlapThreshold);

        Log.i(TAG, String.format("Landmarks kept after intelligent NMS: %d/%d of class %s",
                result.size(), sameClassCandidates.size(), CLASS_NAMES[selectedClass]));

        return result;
    }

    private List<FinalDetection> convertToFinalDetections(List<DetectionCandidate> candidates) {
        List<FinalDetection> result = new ArrayList<FinalDetection>();
        for (DetectionCandidate candidate : candidates) {
            result.add(new FinalDetection(
                    candidate.centerX, candidate.centerY, candidate.width, candidate.height,
                    candidate.confidence, candidate.classId));
        }
        return result;
    }

    private void countQuantities(List<FinalDetection> detections,
                                 Map<Integer, Integer> specificQuantities,
                                 Map<Integer, Integer> allQuantities) {
        for (FinalDetection detection : detections) {
            // Using getOrDefault requires Java 8. For Java 7, check if key exists.
            Integer specificCount = specificQuantities.get(detection.classId);
            specificQuantities.put(detection.classId,
                    (specificCount == null ? 0 : specificCount) + 1);

            Integer allCount = allQuantities.get(detection.classId);
            allQuantities.put(detection.classId,
                    (allCount == null ? 0 : allCount) + 1);
        }
    }

    private float calculateIoU(DetectionCandidate a, DetectionCandidate b) {
        float x1_a = a.centerX - a.width / 2;
        float y1_a = a.centerY - a.height / 2;
        float x2_a = a.centerX + a.width / 2;
        float y2_a = a.centerY + a.height / 2;

        float x1_b = b.centerX - b.width / 2;
        float y1_b = b.centerY - b.height / 2;
        float x2_b = b.centerX + b.width / 2;
        float y2_b = b.centerY + b.height / 2;

        float intersectionX1 = Math.max(x1_a, x1_b);
        float intersectionY1 = Math.max(y1_a, y1_b);
        float intersectionX2 = Math.min(x2_a, x2_b);
        float intersectionY2 = Math.min(y2_a, y2_b);

        if (intersectionX2 <= intersectionX1 || intersectionY2 <= intersectionY1) {
            return 0.0f;
        }

        float intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float unionArea = areaA + areaB - intersectionArea;

        return intersectionArea / unionArea;
    }

    private Mat preprocessImage(Mat image) {
        Mat processedImage = new Mat();

        // Convert to RGB if needed
        if (image.channels() == 1) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_GRAY2RGB);
        } else if (image.channels() == 4) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_BGRA2RGB);
        } else if (image.channels() == 3) {
            Imgproc.cvtColor(image, processedImage, Imgproc.COLOR_BGR2RGB);
        } else {
            image.copyTo(processedImage);
        }

        // Resize to model input size
        Mat resizedImage = new Mat();
        Imgproc.resize(processedImage, resizedImage, new Size(INPUT_SIZE, INPUT_SIZE));

        processedImage.release();
        return resizedImage;
    }

    private float[][][][] matToFloatArray(Mat image) {
        float[][][][] inputData = new float[1][3][INPUT_SIZE][INPUT_SIZE];

        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                double[] pixel = image.get(y, x);

                // Normalize to [0, 1]
                inputData[0][0][y][x] = (float) (pixel[0] / 255.0);
                inputData[0][1][y][x] = (float) (pixel[1] / 255.0);
                inputData[0][2][y][x] = (float) (pixel[2] / 255.0);
            }
        }

        return inputData;
    }

    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error closing YOLO service: " + e.getMessage(), e);
        }
    }

    // Helper classes
    public static class DetectionCandidate {
        public final float centerX, centerY, width, height;
        public final float confidence;
        public final int classId;

        public DetectionCandidate(float centerX, float centerY, float width, float height,
                                  float confidence, int classId) {
            this.centerX = centerX;
            this.centerY = centerY;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
            this.classId = classId;
        }
    }

    public static class FinalDetection {
        public final float centerX, centerY, width, height;
        public final float confidence;
        public final int classId;

        public FinalDetection(float centerX, float centerY, float width, float height,
                              float confidence, int classId) {
            this.centerX = centerX;
            this.centerY = centerY;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
            this.classId = classId;
        }

        @Override
        public String toString() {
            return String.format("Detection[class=%s, conf=%.2f, center=(%.1f,%.1f), size=(%.1f,%.1f)]",
                    CLASS_NAMES[classId], confidence, centerX, centerY, width, height);
        }
    }

    public static class EnhancedDetectionResult {
        private List<FinalDetection> detections;
        private Map<Integer, Integer> allQuantities;
        private Map<Integer, Integer> treasureQuantities;
        private Map<Integer, Integer> landmarkQuantities;
        // 修改 1: 新增最高信心 landmark 記錄
        private FinalDetection highestConfidenceLandmark = null;

        public EnhancedDetectionResult() {
            this.detections = new ArrayList<FinalDetection>();
            this.allQuantities = new HashMap<Integer, Integer>();
            this.treasureQuantities = new HashMap<Integer, Integer>();
            this.landmarkQuantities = new HashMap<Integer, Integer>();
        }

        public EnhancedDetectionResult(List<FinalDetection> detections,
                                       Map<Integer, Integer> allQuantities,
                                       Map<Integer, Integer> treasureQuantities,
                                       Map<Integer, Integer> landmarkQuantities,
                                       // 修改 1: 接受 highestConfidenceLandmark
                                       FinalDetection highestConfidenceLandmark) {
            this.detections = detections;
            this.allQuantities = allQuantities;
            this.treasureQuantities = treasureQuantities;
            this.landmarkQuantities = landmarkQuantities;
            // 修改 1: 賦值 highestConfidenceLandmark
            this.highestConfidenceLandmark = highestConfidenceLandmark;
        }

        public List<FinalDetection> getDetections() {
            return detections;
        }

        public Map<Integer, Integer> getAllQuantities() {
            return allQuantities;
        }

        public Map<Integer, Integer> getTreasureQuantities() {
            return treasureQuantities;
        }

        public Map<Integer, Integer> getLandmarkQuantities() {
            return landmarkQuantities;
        }

        // 修改 1: 新增 getter 方法
        public FinalDetection getHighestConfidenceLandmark() {
            return highestConfidenceLandmark;
        }

        /**
         * Get result in Python-like format
         * * @return Map containing quantities with class names as keys
         */
        public Map<String, Object> getPythonLikeResult() {
            Map<String, Object> result = new HashMap<String, Object>();

            // Convert all quantities to use class names
            Map<String, Integer> allQuantitiesNamed = new HashMap<String, Integer>();
            for (Map.Entry<Integer, Integer> entry : allQuantities.entrySet()) {
                allQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            // Convert treasure quantities to use class names
            Map<String, Integer> treasureQuantitiesNamed = new HashMap<String, Integer>();
            for (Map.Entry<Integer, Integer> entry : treasureQuantities.entrySet()) {
                treasureQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            // Convert landmark quantities to use class names
            Map<String, Integer> landmarkQuantitiesNamed = new HashMap<String, Integer>();
            for (Map.Entry<Integer, Integer> entry : landmarkQuantities.entrySet()) {
                landmarkQuantitiesNamed.put(CLASS_NAMES[entry.getKey()], entry.getValue());
            }

            result.put("all_quantities", allQuantitiesNamed);
            result.put("treasure_quantities", treasureQuantitiesNamed);
            result.put("landmark_quantities", landmarkQuantitiesNamed);

            // 修改 3: 加入最高信心 landmark 名稱
            if (highestConfidenceLandmark != null) {
                result.put("highest_conf_landmark", CLASS_NAMES[highestConfidenceLandmark.classId]);
            }

            return result;
        }

        public void logResults(String tag) {
            Log.i(tag, String.format("Total detections: %d", detections.size()));

            Map<String, Object> pythonResult = getPythonLikeResult();
            Log.i(tag, "All quantities: " + pythonResult.get("all_quantities"));
            Log.i(tag, "Treasure quantities: " + pythonResult.get("treasure_quantities"));
            Log.i(tag, "Landmark quantities: " + pythonResult.get("landmark_quantities"));
            if (pythonResult.containsKey("highest_conf_landmark")) {
                Log.i(tag, "Highest confidence landmark: " + pythonResult.get("highest_conf_landmark"));
            }
        }
    }
}