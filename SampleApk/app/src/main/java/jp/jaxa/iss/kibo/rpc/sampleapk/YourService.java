package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import android.util.Log;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

// OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.imgproc.CLAHE;

public class YourService extends KiboRpcService {

    private final String TAG = this.getClass().getSimpleName();

    private Set<String> foundTreasures = new HashSet<String>();
    private Set<String> foundLandmarks = new HashSet<String>();
    private Map<String, Map<String, Integer>> areaLandmarks = new HashMap<String, Map<String, Integer>>();

    private Map<Integer, Set<String>> areaTreasure = new HashMap<Integer, Set<String>>();

    // Report points, adjusted by AR tag position. This array will store adjusted points.
    // Index mapping: reportPoints[0] for Area 1, reportPoints[1] for Area 2,
    // reportPoints[2] for Area 3, reportPoints[3] for Area 4.
    private Point[] reportPoints = new Point[4];

    // Area coordinates and orientations for the 3 distinct physical movement points
    private final Point[] MOVE_POINTS = {
            new Point(10.425d, -9.5d, 4.445d),         // Physical Point 0 (for Area 1)
            new Point(10.95d, -9.78d, 5.195d),      // Physical Point 1 (for Area 2 and Area 3)
            new Point(10.666984d, -6.8525d, 4.945d)    // Physical Point 2 (for Area 4)
    };

    private final Quaternion[] MOVE_QUATERNIONS = {
            new Quaternion(0.0617f, 0.0082f, 0.6152f, 0.7071f), // Quaternion for Physical Point 0 (Area 1)
            new Quaternion(0f, 0f, -0.707f, 0.707f),  // Quaternion for Physical Point 1 (Area 2/3)
            new Quaternion(0f, 0f, 1f, 0f)           // Quaternion for Physical Point 2 (Area 4)
    };

    private Set<String> reportedLandmarkNames = new HashSet<String>();
    private List<String> availableLandmarkNames = new ArrayList<String>();
    private Random random = new Random();

    public YourService() {
        for (String name : YOLODetectionService.getClassNames()) {
            if (!("crystal".equals(name) || "diamond".equals(name) || "emerald".equals(name))) {
                availableLandmarkNames.add(name);
            }
        }
    }

    @Override
    protected void runPlan1(){
        Log.i(TAG, "Start mission");
        api.startMission();

        // Initialize areaTreasure for all 4 conceptual areas.
        // Initialize reportPoints with default values, they will be adjusted by AR tag detection.
        for (int i = 0; i < 4; i++) {
            areaTreasure.put(i + 1, new HashSet<String>());
            // Initially, reportPoints could be null or some placeholder,
            // they will be populated during imageEnhanceAndCrop.
            // For now, let's ensure they are not null, perhaps copying from MOVE_POINTS if a direct map exists.
            // A more robust approach might be to ensure imageEnhanceAndCrop always sets it.
            // For Areas 2 and 3, reportPoints[1] and reportPoints[2] need to be handled during dual processing.
        }

        Size cropWarpSize = new Size(640, 480);
        Size resizeSize = new Size(320, 320);

        // Iterate through the 3 distinct physical movement points
        for (int movePointIndex = 0; movePointIndex < MOVE_POINTS.length; movePointIndex++) {
            Point currentMovePoint = MOVE_POINTS[movePointIndex];
            Quaternion targetQuaternion = MOVE_QUATERNIONS[movePointIndex];

            // Determine which conceptual Area(s) this physical point corresponds to
            int[] conceptualAreaIds;
            if (movePointIndex == 0) { // Physical Point 0 -> Conceptual Area 1
                conceptualAreaIds = new int[]{1};
            } else if (movePointIndex == 1) { // Physical Point 1 -> Conceptual Areas 2 & 3
                conceptualAreaIds = new int[]{2, 3};
            } else { // Physical Point 2 -> Conceptual Area 4
                conceptualAreaIds = new int[]{4};
            }

            Log.i(TAG, String.format("Moving to physical point %d for conceptual Area(s) %s: Point(%.3f, %.3f, %.3f)",
                    movePointIndex, java.util.Arrays.toString(conceptualAreaIds),
                    currentMovePoint.getX(), currentMovePoint.getY(), currentMovePoint.getZ()));

            api.moveTo(currentMovePoint, targetQuaternion, false);

            Mat image = api.getMatNavCam();
            Mat undistortedImage = new Mat();
            double[][] intrinsics = api.getNavCamIntrinsics();
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);
            Calib3d.undistort(image, undistortedImage, cameraMatrix, distCoeffs, cameraMatrix);
            api.saveMatImage(undistortedImage, "physical_point_" + movePointIndex + "_undistorted.png");

            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<Mat>();
            Mat ids = new Mat();
            Aruco.detectMarkers(undistortedImage, dictionary, corners, ids);

            boolean processedThisPhysicalPoint = false;

            // Specific handling for the physical point that covers Area 2 and Area 3
            if (movePointIndex == 1) { // This is the physical point for conceptual Areas 2 & 3
                Log.i(TAG, "Attempting dual AR detection for conceptual Areas 2 and 3.");

                // Get two markers sorted by X-position (left to right)
                // This will return an Object array: [List<Mat> sortedCorners, Mat sortedIds]
                Object[] dualMarkers = getTwoMarkersByPosition(corners, ids, undistortedImage);

                if (dualMarkers != null) {
                    @SuppressWarnings("unchecked")
                    List<Mat> twoCorners = (List<Mat>) dualMarkers[0];
                    Mat twoIds = (Mat) dualMarkers[1]; // Contains sorted IDs (left ID, right ID)

                    Log.i(TAG, "Two markers found at physical point " + movePointIndex + ".");
                    Log.i(TAG, "Sorted AR IDs: Left ID=" + (int)twoIds.get(0,0)[0] + ", Right ID=" + (int)twoIds.get(1,0)[0]);

                    // Determine which ID belongs to Area 2 (left) and which to Area 3 (right)
                    int idLeft = (int) twoIds.get(0, 0)[0];
                    int idRight = (int) twoIds.get(1, 0)[0];

                    int area2Id = 2; // Conceptual Area 2
                    int area3Id = 3; // Conceptual Area 3

                    // Process Area 2 (left marker)
                    Log.i(TAG, "Processing conceptual Area " + area2Id + " (left marker, ID: " + idLeft + ")");
                    List<Mat> area2CornerList = new ArrayList<Mat>();
                    area2CornerList.add(twoCorners.get(0));
                    Mat area2SingleIdMat = new Mat(1, 1, CvType.CV_32S);
                    area2SingleIdMat.put(0, 0, idLeft);
                    processSingleArea(area2Id, area2CornerList, area2SingleIdMat, undistortedImage, cropWarpSize, resizeSize);
                    area2SingleIdMat.release();
                    for(Mat c : area2CornerList) c.release();

                    // Process Area 3 (right marker)
                    Log.i(TAG, "Processing conceptual Area " + area3Id + " (right marker, ID: " + idRight + ")");
                    List<Mat> area3CornerList = new ArrayList<Mat>();
                    area3CornerList.add(twoCorners.get(1));
                    Mat area3SingleIdMat = new Mat(1, 1, CvType.CV_32S);
                    area3SingleIdMat.put(0, 0, idRight);
                    processSingleArea(area3Id, area3CornerList, area3SingleIdMat, undistortedImage, cropWarpSize, resizeSize);
                    area3SingleIdMat.release();
                    for(Mat c : area3CornerList) c.release();

                    processedThisPhysicalPoint = true;
                    twoIds.release(); // Release the Mat created in getTwoMarkersByPosition
                } else {
                    Log.w(TAG, "Less than two markers found at physical point for Area 2/3. Cannot perform dual AR processing.");
                    // Fallback to single processing if dual fails, iterate conceptualAreaIds for this point
                    for (int conceptualId : conceptualAreaIds) {
                        Mat claHeBinImage = imageEnhanceAndCrop(undistortedImage, corners, ids, cropWarpSize, resizeSize, conceptualId);
                        handleSingleAreaDetection(conceptualId, claHeBinImage); // Reusing the helper
                    }
                    processedThisPhysicalPoint = true; // Mark as processed to avoid default single processing below
                }
            }

            // Default processing for single areas (Area 1, Area 4) or if dual processing wasn't applicable/failed
            if (!processedThisPhysicalPoint) {
                // For conceptualAreaIds: it will contain {1} for movePointIndex 0, and {4} for movePointIndex 2
                for (int conceptualId : conceptualAreaIds) {
                    Mat claHeBinImage = imageEnhanceAndCrop(undistortedImage, corners, ids, cropWarpSize, resizeSize, conceptualId);
                    handleSingleAreaDetection(conceptualId, claHeBinImage);
                }
            }

            // Release resources
            image.release();
            undistortedImage.release();
            for (Mat corner : corners) {
                corner.release();
            }
            ids.release();
            cameraMatrix.release();
            distCoeffs.release();

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Log.w(TAG, "Sleep interrupted");
            }
        }

        // LOG SUMMARY OF ALL AREAS (still 4 conceptual areas)
        Log.i(TAG, "=== AREA PROCESSING SUMMARY ===");
        for (int i = 1; i <= 4; i++) {
            Log.i(TAG, "Area " + i + " treasures: " + areaTreasure.get(i));
            Log.i(TAG, "Area " + i + " landmarks: " + areaLandmarks.get("area" + i));
        }
        Log.i(TAG, "All found treasures: " + foundTreasures);
        Log.i(TAG, "All found landmarks: " + foundLandmarks);

        // ASTRONAUT INTERACTION
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);

        Log.i(TAG, "Moving to astronaut position");
        api.moveTo(astronautPoint, astronautQuaternion, false);
        api.reportRoundingCompletion();

        boolean astronautMarkersOk = waitForMarkersDetection(2000, 200, "astronaut");

        if (astronautMarkersOk) {
            Log.i(TAG, "Astronaut markers confirmed - proceeding with target detection");
        } else {
            Log.w(TAG, "Astronaut markers not detected - proceeding anyway");
        }

        // TARGET ITEM RECOGNITION
        Mat targetImage = api.getMatNavCam();

        Mat undistortedTargetImage = new Mat();
        Mat cameraMatrixForTarget = new Mat(3, 3, CvType.CV_64F);
        Mat distCoeffsForTarget = new Mat(1, 5, CvType.CV_64F);

        double[][] intrinsicsForTarget = api.getNavCamIntrinsics();
        cameraMatrixForTarget.put(0, 0, intrinsicsForTarget[0]);
        distCoeffsForTarget.put(0, 0, intrinsicsForTarget[1]);
        Calib3d.undistort(targetImage, undistortedTargetImage, cameraMatrixForTarget, distCoeffsForTarget, cameraMatrixForTarget);
        cameraMatrixForTarget.release();
        distCoeffsForTarget.release();

        String targetTreasureType = processTargetImage(undistortedTargetImage, resizeSize);

        if (targetTreasureType != null && !targetTreasureType.equals("unknown")) {
            Log.i(TAG, "Target treasure identified: " + targetTreasureType);

            int targetAreaId = findTreasureInArea(targetTreasureType, areaTreasure);

            if (targetAreaId > 0) {
                Log.i(TAG, "Target treasure '" + targetTreasureType + "' found in Area " + targetAreaId);
                api.notifyRecognitionItem();

                // 優先使用 AR Tag 偵測後的 reportPoints
                Point targetAreaPoint = reportPoints[targetAreaId - 1];
                Quaternion targetAreaQuaternion = null;

                if (targetAreaPoint == null) {
                    Log.w(TAG, "AR Tag-based reportPoint is null, fallback to AREA_POINTS");
                    targetAreaPoint = MOVE_POINTS[targetAreaId - 1];
                    targetAreaQuaternion = MOVE_QUATERNIONS[targetAreaId - 1];
                } else {
                    // 根據 targetAreaId 指定 quaternion
                    if (targetAreaId == 1) {
                        targetAreaQuaternion = MOVE_QUATERNIONS[0];
                    } else if (targetAreaId == 2 || targetAreaId == 3) {
                        targetAreaQuaternion = MOVE_QUATERNIONS[1];
                    } else if (targetAreaId == 4) {
                        targetAreaQuaternion = MOVE_QUATERNIONS[2];
                    } else {
                        targetAreaQuaternion = new Quaternion(0, 0, 0, 1); // default safe
                    }
                }

                Log.i(TAG, String.format("Moving back to Area %d: Point(%.3f, %.3f, %.3f)",
                        targetAreaId, targetAreaPoint.getX(), targetAreaPoint.getY(), targetAreaPoint.getZ()));
                api.moveTo(targetAreaPoint, targetAreaQuaternion, false);
                api.takeTargetItemSnapshot();
                Log.i(TAG, "Mission completed successfully!");

            } else {
                Log.w(TAG, "Target treasure '" + targetTreasureType + "' not found in any area.");
                api.notifyRecognitionItem();
                api.takeTargetItemSnapshot();
            }

        } else {
            Log.w(TAG, "Could not identify target treasure from astronaut.");
            api.notifyRecognitionItem();
            api.takeTargetItemSnapshot();
        }


        targetImage.release();
        undistortedTargetImage.release();
    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }

    private String processTargetImage(Mat targetImage, Size resizeSize) {
        try {
            Log.i(TAG, "Processing target image from astronaut");

            api.saveMatImage(targetImage, "target_astronaut_raw.png");

            Size cropWarpSize = new Size(640, 480);

            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<Mat>();
            Mat ids = new Mat();
            Aruco.detectMarkers(targetImage, dictionary, corners, ids);

            Mat processedTarget = null;
            if (!corners.isEmpty()) {
                Object[] filtered = keepClosestMarker(corners, ids, targetImage);
                @SuppressWarnings("unchecked")
                List<Mat> filteredCorners = (List<Mat>) filtered[0];
                Mat filteredIds = (Mat) filtered[1];

                processedTarget = imageEnhanceAndCrop(targetImage, filteredCorners, filteredIds, cropWarpSize, resizeSize, 0); // areaId=0 for target

                for (Mat corner : filteredCorners) {
                    corner.release();
                }
                filteredIds.release();

            } else {
                Log.w(TAG, "No ArUco markers detected in target image for cropping. Applying simpler enhancement.");
                processedTarget = enhanceTargetImage(targetImage, resizeSize);
            }

            for (Mat corner : corners) {
                corner.release();
            }
            ids.release();

            if (processedTarget != null) {
                Log.i(TAG, "Target image processing successful.");

                Object[] detected_items = detectitemfromcvimg(
                        processedTarget,
                        0.3f,
                        "target",
                        0.45f,
                        0.8f,
                        320
                );

                @SuppressWarnings("unchecked")
                Map<String, Integer> landmark_items = (Map<String, Integer>) detected_items[0];
                @SuppressWarnings("unchecked")
                Set<String> treasure_types = (Set<String>) detected_items[1];

                Log.i(TAG, "Target - Landmark quantities: " + landmark_items);
                Log.i(TAG, "Target - Treasure types: " + treasure_types);

                if (!treasure_types.isEmpty()) {
                    String targetTreasure = treasure_types.iterator().next();
                    Log.i(TAG, "Target treasure detected: " + targetTreasure);
                    processedTarget.release();
                    return targetTreasure;
                }

                processedTarget.release();
            } else {
                Log.w(TAG, "Target image processing failed.");
            }

            Log.w(TAG, "No treasure detected in target image.");
            return "unknown";

        } catch (Exception e) {
            Log.e(TAG, "Error processing target image: " + e.getMessage());
            return "unknown";
        }
    }

    private Mat enhanceTargetImage(Mat image, Size resizeSize) {
        try {
            Mat resized = new Mat();
            Imgproc.resize(image, resized, resizeSize);

            Mat grayImage = new Mat();
            if (resized.channels() == 3) {
                Imgproc.cvtColor(resized, grayImage, Imgproc.COLOR_RGB2GRAY);
            } else {
                resized.copyTo(grayImage);
            }

            Mat enhanced = new Mat();
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);
            clahe.setTilesGridSize(new Size(8, 8));
            clahe.apply(grayImage, enhanced);

            api.saveMatImage(enhanced, "target_astronaut_enhanced.png");

            resized.release();
            grayImage.release();
            return enhanced;

        } catch (Exception e) {
            Log.e(TAG, "Error enhancing target image: " + e.getMessage());
            return null;
        }
    }

    private int findTreasureInArea(String treasureType, Map<Integer, Set<String>> areaTreasure) {
        for (int areaId = 1; areaId <= 4; areaId++) {
            Set<String> treasures = areaTreasure.get(areaId);
            if (treasures != null && treasures.contains(treasureType)) {
                return areaId;
            }
        }
        return 0;
    }

    private Object[] detectitemfromcvimg(Mat image, float conf, String imgtype,
                                         float standard_nms_threshold, float overlap_nms_threshold, int img_size) {
        YOLODetectionService yoloService = null;
        try {
            Log.i(TAG, String.format("Starting YOLO detection - type: %s, conf: %.2f", imgtype, conf));

            yoloService = new YOLODetectionService(this);

            YOLODetectionService.EnhancedDetectionResult result = yoloService.DetectfromcvImage(
                    image, imgtype, conf, standard_nms_threshold, overlap_nms_threshold
            );

            Map<String, Object> pythonLikeResult = result.getPythonLikeResult();

            Map<Integer, Integer> rawLandmarkQuantities = result.getLandmarkQuantities();
            Map<String, Integer> landmarkQuantities = new HashMap<String, Integer>();
            if (rawLandmarkQuantities != null) {
                for (Map.Entry<Integer, Integer> entry : rawLandmarkQuantities.entrySet()) {
                    String className = YOLODetectionService.getClassName(entry.getKey());
                    if (className != null) {
                        landmarkQuantities.put(className, entry.getValue());
                    }
                }
            }

            Map<Integer, Integer> rawTreasureQuantities = result.getTreasureQuantities();
            Set<String> treasureTypes = new HashSet<String>();
            if (rawTreasureQuantities != null) {
                for (Integer classId : rawTreasureQuantities.keySet()) {
                    String className = YOLODetectionService.getClassName(classId);
                    if (className != null) {
                        treasureTypes.add(className);
                    }
                }
            }

            String highestConfLandmarkName = (String) pythonLikeResult.get("highest_conf_landmark");

            Log.i(TAG, "YOLO - Landmark quantities: " + landmarkQuantities);
            Log.i(TAG, "YOLO - Treasure types: " + treasureTypes);
            if (highestConfLandmarkName != null) {
                Log.i(TAG, "YOLO - Highest Confidence Landmark: " + highestConfLandmarkName);
            }

            return new Object[]{landmarkQuantities, treasureTypes, highestConfLandmarkName};

        } catch (Exception e) {
            Log.e(TAG, "Error in detectitemfromcvimg: " + e.getMessage(), e);
            return new Object[]{new HashMap<String, Integer>(), new HashSet<String>(), null};
        } finally {
            if (yoloService != null) {
                yoloService.close();
            }
        }
    }

    private String[] getFirstLandmarkItem(Map<String, Integer> landmarkQuantities) {
        if (landmarkQuantities != null && !landmarkQuantities.isEmpty()) {
            List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<Map.Entry<String, Integer>>(landmarkQuantities.entrySet());
            Collections.sort(sortedEntries, new Comparator<Map.Entry<String, Integer>>() {
                @Override
                public int compare(Map.Entry<String, Integer> a, Map.Entry<String, Integer> b) {
                    int countCompare = b.getValue().compareTo(a.getValue());
                    if (countCompare != 0) {
                        return countCompare;
                    }
                    return a.getKey().compareTo(b.getKey());
                }
            });

            Map.Entry<String, Integer> firstEntry = sortedEntries.get(0);
            String landmarkName = firstEntry.getKey();
            Integer count = firstEntry.getValue();
            return new String[]{landmarkName, String.valueOf(count)};
        }
        return null;
    }

    /**
     * Detects up to two ArUco markers and returns them sorted by their X-position (left to right).
     * If more than two are detected, it prioritizes the two closest to the camera.
     *
     * @param allCorners List of all detected marker corners.
     * @param allIds Mat of all detected marker IDs.
     * @param image The input image for pose estimation.
     * @return An Object array: [List<Mat> sortedCorners, Mat sortedIds] of the two markers,
     * or null if less than two markers are found.
     */
    private Object[] getTwoMarkersByPosition(List<Mat> allCorners, Mat allIds, Mat image) {
        if (allCorners == null || allCorners.size() < 2 || allIds == null || allIds.empty()) {
            Log.d(TAG, "getTwoMarkersByPosition: Less than 2 markers detected.");
            return null;
        }

        List<MarkerInfo> markerInfos = new ArrayList<MarkerInfo>();

        double[][] intrinsics = api.getNavCamIntrinsics();
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
        cameraMatrix.put(0, 0, intrinsics[0]);
        distCoeffs.put(0, 0, intrinsics[1]);

        Mat rvecs = new Mat();
        Mat tvecs = new Mat();
        float markerLength = 0.05f; // Assuming default marker length

        Aruco.estimatePoseSingleMarkers(allCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        for (int i = 0; i < allCorners.size(); i++) {
            int markerId = (int) allIds.get(i, 0)[0];
            double[] tvecData = tvecs.get(i, 0); // Translation vector [x, y, z] in camera frame
            if (tvecData != null && tvecData.length >= 3) {
                // We use tvecData[0] for X-position (horizontal position in camera's view)
                // and tvecData[2] for Z-position (distance from camera)
                markerInfos.add(new MarkerInfo(markerId, allCorners.get(i), tvecData[0], tvecData[2]));
            }
        }

        cameraMatrix.release();
        distCoeffs.release();
        rvecs.release();
        tvecs.release();

        // Sort markers by distance first (closest two)
        Collections.sort(markerInfos, new Comparator<MarkerInfo>() {
            @Override
            public int compare(MarkerInfo m1, MarkerInfo m2) {
                return Double.compare(m1.distanceZ, m2.distanceZ); // Sort by Z-distance (depth)
            }
        });

        // Take the top 2 closest markers
        List<MarkerInfo> closestTwoMarkers = new ArrayList<MarkerInfo>();
        if (markerInfos.size() >= 2) {
            closestTwoMarkers.add(markerInfos.get(0));
            closestTwoMarkers.add(markerInfos.get(1));
        } else {
            // This should not happen if allCorners.size() >= 2, but for robustness
            Log.w(TAG, "getTwoMarkersByPosition: Not enough valid marker poses found after filtering.");
            return null;
        }

        // Now, sort these two markers by their X-position (left to right in the image)
        Collections.sort(closestTwoMarkers, new Comparator<MarkerInfo>() {
            @Override
            public int compare(MarkerInfo m1, MarkerInfo m2) {
                return Double.compare(m1.xPosition, m2.xPosition); // Sort by X-position (left to right)
            }
        });

        List<Mat> twoCorners = new ArrayList<Mat>();
        twoCorners.add(closestTwoMarkers.get(0).cornerMat); // Left marker's corners
        twoCorners.add(closestTwoMarkers.get(1).cornerMat); // Right marker's corners

        Mat twoIds = new Mat(2, 1, CvType.CV_32S);
        twoIds.put(0, 0, closestTwoMarkers.get(0).id); // Left marker's ID
        twoIds.put(1, 0, closestTwoMarkers.get(1).id); // Right marker's ID

        Log.i(TAG, "getTwoMarkersByPosition: Returning two markers. Left ID: " + closestTwoMarkers.get(0).id +
                " (X: " + String.format("%.3f", closestTwoMarkers.get(0).xPosition) +
                "), Right ID: " + closestTwoMarkers.get(1).id +
                " (X: " + String.format("%.3f", closestTwoMarkers.get(1).xPosition) + ")");

        return new Object[]{twoCorners, twoIds};
    }

    /**
     * Helper method to process a single conceptual area (e.g., Area 1, or Area 2/3 if dual detected).
     * This abstracts the common image processing and setAreaInfo logic.
     *
     * @param conceptualAreaId The ID of the conceptual area (1, 2, 3, or 4).
     * @param corners List of corners for the specific marker being processed (usually one).
     * @param ids Mat of IDs for the specific marker being processed (usually one).
     * @param undistortedImage The original undistorted camera image.
     * @param cropWarpSize Target size for cropping/warping.
     * @param resizeSize Target size for final YOLO input.
     */
    private void processSingleArea(int conceptualAreaId, List<Mat> corners, Mat ids, Mat undistortedImage, Size cropWarpSize, Size resizeSize) {
        Log.i(TAG, "Processing conceptual Area " + conceptualAreaId + " with single marker logic.");
        Mat claHeBinImage = imageEnhanceAndCrop(undistortedImage, corners, ids, cropWarpSize, resizeSize, conceptualAreaId);
        handleSingleAreaDetection(conceptualAreaId, claHeBinImage); // Reusing the helper
    }


    private Mat imageEnhanceAndCrop(Mat image, List<Mat> allCorners, Mat allIds, Size cropWarpSize, Size resizeSize, int areaId) {
        try {
            String rawImageFilename = "area_" + areaId + "_raw.png";
            api.saveMatImage(image, rawImageFilename);
            Log.i(TAG, "Raw image saved as " + rawImageFilename);

            List<Mat> filteredCorners = new ArrayList<Mat>();
            Mat filteredIds = new Mat();

            if (allCorners != null && !allCorners.isEmpty() && allIds != null && !allIds.empty()) {
                Log.i(TAG, "Detected " + allCorners.size() + " markers in total for Area " + areaId + ".");

                // Keep only the closest marker if multiple are detected,
                // useful for single area processing or if getTwoMarkersByPosition didn't filter
                Object[] filtered = keepClosestMarker(allCorners, allIds, image);
                @SuppressWarnings("unchecked")
                List<Mat> tempFilteredCorners = (List<Mat>) filtered[0];
                if (tempFilteredCorners != null) {
                    filteredCorners.addAll(tempFilteredCorners);
                }
                filteredIds = (Mat) filtered[1];

            } else {
                Log.w(TAG, "No ArUco markers detected in image for Area " + areaId + ". Cannot perform image enhancement and cropping based on AR tag.");
                return null;
            }

            if (filteredCorners.isEmpty()) {
                Log.w(TAG, "No closest marker found after filtering for Area " + areaId + ".");
                if (filteredIds != null) filteredIds.release();
                return null;
            }

            Log.i(TAG, "Using closest marker for Area " + areaId + ". Remaining markers: " + filteredCorners.size());

            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            double[][] intrinsics = api.getNavCamIntrinsics();
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);

            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            float markerLength = 0.05f;

            Aruco.estimatePoseSingleMarkers(filteredCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            Mat imageWithFrame = image.clone();
            Imgproc.cvtColor(imageWithFrame, imageWithFrame, Imgproc.COLOR_GRAY2RGB);
            Aruco.drawDetectedMarkers(imageWithFrame, filteredCorners, filteredIds);

            if (rvecs.rows() > 0 && tvecs.rows() > 0) {
                Mat rvec = new Mat(3, 1, CvType.CV_64F);
                Mat tvec = new Mat(3, 1, CvType.CV_64F);

                rvecs.row(0).copyTo(rvec);
                tvecs.row(0).copyTo(tvec);

                Calib3d.drawFrameAxes(imageWithFrame, cameraMatrix, distCoeffs, rvec, tvec, 0.1f);

                String markerFilename = "area_" + areaId + "_marker_0_with_frame.png";
                api.saveMatImage(imageWithFrame, markerFilename);
                Log.i(TAG, "Marker image saved as " + markerFilename);

                // OPTIMIZE REPORTING LOCATION BASED ON ARUCO POSITION
                double[] offset = new double[3];
                tvec.get(0, 0, offset); // Get the translation vector from marker detection

                Point basePointForArea = null;
                // Map conceptual area ID back to the closest MOVE_POINTS index for its base
                if (areaId == 1) {
                    basePointForArea = MOVE_POINTS[0];
                } else if (areaId == 2 || areaId == 3) {
                    basePointForArea = MOVE_POINTS[1];
                } else if (areaId == 4) {
                    basePointForArea = MOVE_POINTS[2];
                }

                if (basePointForArea != null) {
                    // Apply the offset (from camera to marker) to the base Astrobee point
                    // Note: This mapping of tvec (x,y,z) to Astrobee's (x,y,z) needs careful validation
                    // depending on your camera and Astrobee's coordinate systems.
                    // Typically, camera X is right, Y is down, Z is forward.
                    // Astrobee X is forward, Y is left, Z is up.
                    // A common mapping if camera is looking forward:
                    // Astrobee_X_new = base_X + camera_Z_offset
                    // Astrobee_Y_new = base_Y - camera_X_offset (negated for left/right mapping)
                    // Astrobee_Z_new = base_Z - camera_Y_offset (negated for up/down mapping)
                    Point adjusted = new Point(
                            basePointForArea.getX() + offset[2],  // Camera Z -> Astrobee X
                            basePointForArea.getY() - offset[0],  // Camera X -> Astrobee Y (negated)
                            basePointForArea.getZ() - offset[1]   // Camera Y -> Astrobee Z (negated)
                    );
                    reportPoints[areaId - 1] = adjusted; // Store adjusted point for future use

                    Log.i(TAG, String.format("Area %d AR Tag Adjusted Point: (%.3f, %.3f, %.3f)",
                            areaId, adjusted.getX(), adjusted.getY(), adjusted.getZ()));
                } else {
                    Log.w(TAG, "Could not determine base point for Area " + areaId + " to adjust AR tag position.");
                }


                Mat processedImage = processCropRegion(filteredCorners.get(0), image, cameraMatrix, distCoeffs, rvec, tvec, cropWarpSize, resizeSize, areaId);

                rvec.release();
                tvec.release();
                imageWithFrame.release();
                cameraMatrix.release();
                distCoeffs.release();
                rvecs.release();
                tvecs.release();

                filteredIds.release();
                for (Mat corner : filteredCorners) {
                    corner.release();
                }

                return processedImage;
            }

            imageWithFrame.release();
            cameraMatrix.release();
            distCoeffs.release();
            rvecs.release();
            tvecs.release();
            filteredIds.release();
            for (Mat corner : filteredCorners) {
                corner.release();
            }
            return null;
        } catch (Exception e) {
            Log.e(TAG, "Error in imageEnhanceAndCrop for Area " + areaId + ": " + e.getMessage(), e);
            return null;
        }
    }

    private Mat processCropRegion(Mat singleCorner, Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvec, Mat tvec, Size cropWarpSize, Size resizeSize, int areaId) {
        MatOfDouble distCoeffsDouble = null;
        MatOfPoint3f cropCornersMat = null;
        MatOfPoint2f cropCorners2D = null;

        try {
            // Define the 3D coordinates of the target cropping region relative to the AR tag.
            // These points form a rectangle in the plane of the AR tag.
            org.opencv.core.Point3[] cropCorners3D = {
                    new org.opencv.core.Point3(-0.0265, 0.0420, 0),    // Top-left of the target area
                    new org.opencv.core.Point3(-0.2385, 0.0420, 0),   // Top-right
                    new org.opencv.core.Point3(-0.2385, -0.1170, 0),  // Bottom-right
                    new org.opencv.core.Point3(-0.0265, -0.1170, 0)   // Bottom-left
            };

            cropCornersMat = new MatOfPoint3f(cropCorners3D);
            cropCorners2D = new MatOfPoint2f();

            distCoeffsDouble = new MatOfDouble(distCoeffs.rows(), distCoeffs.cols());
            distCoeffs.copyTo(distCoeffsDouble);

            // Project these 3D points onto the 2D image plane using the camera pose and intrinsics
            Calib3d.projectPoints(cropCornersMat, rvec, tvec, cameraMatrix, distCoeffsDouble, cropCorners2D);
            org.opencv.core.Point[] cropPoints2D = cropCorners2D.toArray();

            if (cropPoints2D.length == 4) {
                Mat processedImage = cropEnhanceAndBinarize(image, cropPoints2D, cropWarpSize, resizeSize, areaId);
                return processedImage;
            }

            return null;

        } catch (Exception e) {
            Log.e(TAG, "Error in processCropRegion for Area " + areaId + ": " + e.getMessage(), e);
            return null;
        } finally {
            if (distCoeffsDouble != null) {
                distCoeffsDouble.release();
            }
            if (cropCornersMat != null) {
                cropCornersMat.release();
            }
            if (cropCorners2D != null) {
                cropCorners2D.release();
            }
        }
    }

    private void handleSingleAreaDetection(int areaId, Mat claHeBinImage) {
        Map<String, Integer> landmark_items = new HashMap<String, Integer>();
        Set<String> treasure_types = new HashSet<String>();
        String highest_conf_landmark_name = null;

        // Try to infer landmark name from ARUCO ID if possible, as a primary source.
        // This assumes a convention: specific ARUCO IDs map to specific landmarks.
        // You'll need to define this mapping if it's not already in YOLODetectionService.
        String inferredLandmarkName = YOLODetectionService.getClassName(areaId); // Assuming Area ID maps to ARUCO ID

        if (claHeBinImage != null) {
            Log.i(TAG, "Area " + areaId + ": Image enhancement and cropping successful.");

            Object[] detected_items = detectitemfromcvimg(
                    claHeBinImage,
                    0.5f,
                    "lost",
                    0.45f,
                    0.8f,
                    320
            );

            @SuppressWarnings("unchecked")
            Map<String, Integer> tempLandmarkItems = (Map<String, Integer>) detected_items[0];
            if (tempLandmarkItems != null) {
                landmark_items.putAll(tempLandmarkItems);
            }

            @SuppressWarnings("unchecked")
            Set<String> tempTreasureTypes = (Set<String>) detected_items[1];
            if (tempTreasureTypes != null) {
                treasure_types.addAll(tempTreasureTypes);
            }

            highest_conf_landmark_name = (String) detected_items[2];

            Log.i(TAG, "Area " + areaId + " - YOLO Landmark quantities: " + landmark_items);
            Log.i(TAG, "Area " + areaId + " - YOLO Treasure types: " + treasure_types);
            if (highest_conf_landmark_name != null) {
                Log.i(TAG, "Area " + areaId + " - YOLO Highest Confidence Landmark: " + highest_conf_landmark_name);
            }

            areaLandmarks.put("area" + areaId, landmark_items);
            foundTreasures.addAll(treasure_types);

            if (highest_conf_landmark_name != null) {
                foundLandmarks.add(highest_conf_landmark_name);
            } else {
                foundLandmarks.addAll(landmark_items.keySet());
            }

            areaTreasure.get(areaId).addAll(treasure_types);

            Log.i(TAG, "Area " + areaId + " treasure types: " + areaTreasure.get(areaId));

            claHeBinImage.release();
        } else {
            Log.w(TAG, "Area " + areaId + ": Image enhancement failed - no markers detected or processing error.");
        }

        String currentlandmark_item_name = "unknown";
        int landmark_item_count = 0;

        // Prioritize inferred landmark name from ARUCO ID if available and valid
        if (inferredLandmarkName != null && !inferredLandmarkName.isEmpty() && !"unknown".equals(inferredLandmarkName)) {
            currentlandmark_item_name = inferredLandmarkName;
            landmark_item_count = 1; // Assuming AR tag implies 1 item
            Log.i(TAG, String.format("Area %d: Reporting landmark inferred from ARUCO ID: %s x %d", areaId, currentlandmark_item_name, landmark_item_count));
        }
        // Fallback to highest confidence YOLO detection
        else if (highest_conf_landmark_name != null && !highest_conf_landmark_name.isEmpty() && !highest_conf_landmark_name.equals("unknown")) {
            currentlandmark_item_name = highest_conf_landmark_name;
            landmark_item_count = landmark_items.getOrDefault(highest_conf_landmark_name, 1);
            if (landmark_item_count == 0) {
                landmark_item_count = 1;
            }
            Log.i(TAG, String.format("Area %d: Reporting highest confidence landmark from YOLO: %s x %d", areaId, currentlandmark_item_name, landmark_item_count));
        }
        // Fallback to first available landmark from YOLO quantities
        else {
            String[] firstLandmark = getFirstLandmarkItem(landmark_items);
            if (firstLandmark != null) {
                currentlandmark_item_name = firstLandmark[0];
                landmark_item_count = Integer.parseInt(firstLandmark[1]);
                Log.i(TAG, String.format("Area %d: Reporting first available landmark from YOLO quantities: %s x %d", areaId, currentlandmark_item_name, landmark_item_count));
            }
            // Final fallback to a random unreported landmark
            else {
                String fallbackLandmark = getRandomUnreportedLandmark();
                currentlandmark_item_name = fallbackLandmark;
                landmark_item_count = 1;
                Log.w(TAG, "Area " + areaId + ": No landmark detected, reporting fallback: " + fallbackLandmark);
            }
        }
        api.setAreaInfo(areaId, currentlandmark_item_name, landmark_item_count);
        reportedLandmarkNames.add(currentlandmark_item_name);
    }

    private String getRandomUnreportedLandmark() {
        List<String> unreported = new ArrayList<String>();
        for (String landmark : availableLandmarkNames) {
            if (!reportedLandmarkNames.contains(landmark)) {
                unreported.add(landmark);
            }
        }

        if (!unreported.isEmpty()) {
            return unreported.get(random.nextInt(unreported.size()));
        } else {
            reportedLandmarkNames.clear();
            if (!availableLandmarkNames.isEmpty()) {
                return availableLandmarkNames.get(random.nextInt(availableLandmarkNames.size()));
            }
        }
        return "unknown_landmark";
    }

    private boolean waitForMarkersDetection(long timeoutMillis, long intervalMillis, String context) {
        long startTime = System.currentTimeMillis();
        while (System.currentTimeMillis() - startTime < timeoutMillis) {
            Mat image = api.getMatNavCam();
            Mat undistortedImage = new Mat();
            double[][] intrinsics = api.getNavCamIntrinsics();
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            cameraMatrix.put(0, 0, intrinsics[0]);
            distCoeffs.put(0, 0, intrinsics[1]);
            Calib3d.undistort(image, undistortedImage, cameraMatrix, distCoeffs, cameraMatrix);

            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<Mat>();
            Mat ids = new Mat();
            Aruco.detectMarkers(undistortedImage, dictionary, corners, ids);

            int detectedCount = ids.rows();

            Log.d(TAG, "Waiting for markers (" + context + "): Detected " + detectedCount + " markers.");

            image.release();
            undistortedImage.release();
            cameraMatrix.release();
            distCoeffs.release();
            for (Mat corner : corners) {
                corner.release();
            }
            ids.release();

            if (detectedCount > 0) {
                Log.i(TAG, "Markers detected for " + context + ".");
                return true;
            }

            try {
                Thread.sleep(intervalMillis);
            } catch (InterruptedException e) {
                Log.w(TAG, "Wait for markers interrupted.");
                Thread.currentThread().interrupt();
                return false;
            }
        }
        Log.w(TAG, "Timeout waiting for markers (" + context + ").");
        return false;
    }

    private Mat cropEnhanceAndBinarize(Mat image, org.opencv.core.Point[] cropPoints2D, Size cropWarpSize, Size resizeSize, int areaId) {
        Mat grayImage = new Mat();
        Mat warpedImage = new Mat();
        Mat claheOutput = new Mat();
        Mat binaryImage = new Mat();
        MatOfPoint2f srcPoints = null;
        MatOfPoint2f dstPoints = null;

        try {
            if (image.channels() == 3) {
                Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_RGB2GRAY);
            } else {
                image.copyTo(grayImage);
            }

            org.opencv.core.Point[] dstPointsArray = {
                    new org.opencv.core.Point(0, 0),
                    new org.opencv.core.Point(cropWarpSize.width, 0),
                    new org.opencv.core.Point(cropWarpSize.width, cropWarpSize.height),
                    new org.opencv.core.Point(0, cropWarpSize.height)
            };

            srcPoints = new MatOfPoint2f(cropPoints2D);
            dstPoints = new MatOfPoint2f(dstPointsArray);

            Mat perspectiveTransform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
            Imgproc.warpPerspective(grayImage, warpedImage, perspectiveTransform, cropWarpSize);

            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(2.0);
            clahe.setTilesGridSize(new Size(8, 8));
            clahe.apply(warpedImage, claheOutput);

            Imgproc.adaptiveThreshold(claheOutput, binaryImage, 255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

            Mat finalImage = new Mat();
            Imgproc.resize(binaryImage, finalImage, resizeSize);

            if (areaId > 0) {
                api.saveMatImage(warpedImage, "area_" + areaId + "_warped.png");
                api.saveMatImage(claheOutput, "area_" + areaId + "_clahe.png");
                api.saveMatImage(binaryImage, "area_" + areaId + "_binary.png");
                api.saveMatImage(finalImage, "area_" + areaId + "_final.png");
            } else {
                api.saveMatImage(warpedImage, "target_warped.png");
                api.saveMatImage(claheOutput, "target_clahe.png");
                api.saveMatImage(binaryImage, "target_binary.png");
                api.saveMatImage(finalImage, "target_final.png");
            }

            grayImage.release();
            warpedImage.release();
            claheOutput.release();
            binaryImage.release();
            perspectiveTransform.release();

            return finalImage;

        } catch (Exception e) {
            Log.e(TAG, "Error in cropEnhanceAndBinarize for Area " + areaId + ": " + e.getMessage(), e);
            return null;
        } finally {
            if (srcPoints != null) {
                srcPoints.release();
            }
            if (dstPoints != null) {
                dstPoints.release();
            }
        }
    }

    private Object[] keepClosestMarker(List<Mat> corners, Mat ids, Mat image) {
        if (corners == null || corners.isEmpty() || ids == null || ids.empty()) {
            return new Object[]{new ArrayList<Mat>(), new Mat()};
        }

        double[][] intrinsics = api.getNavCamIntrinsics();
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
        cameraMatrix.put(0, 0, intrinsics[0]);
        distCoeffs.put(0, 0, intrinsics[1]);

        Mat rvecs = new Mat();
        Mat tvecs = new Mat();
        float markerLength = 0.05f;

        Aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

        double minDistance = Double.MAX_VALUE;
        int closestIndex = -1;

        for (int i = 0; i < tvecs.rows(); i++) {
            double[] tvecData = tvecs.get(i, 0);
            if (tvecData != null && tvecData.length >= 3) {
                // Distance is typically sqrt(x^2 + y^2 + z^2) for 3D Euclidean distance
                // Or just Z (tvecData[2]) if comparing depth directly
                double distance = tvecData[2]; // Using Z-coordinate for depth/distance
                if (distance < minDistance) {
                    minDistance = distance;
                    closestIndex = i;
                }
            }
        }

        cameraMatrix.release();
        distCoeffs.release();
        rvecs.release();
        tvecs.release();

        if (closestIndex != -1) {
            List<Mat> closestCorner = new ArrayList<Mat>();
            closestCorner.add(corners.get(closestIndex));
            Mat closestId = new Mat(1, 1, CvType.CV_32S);
            closestId.put(0, 0, (int) ids.get(closestIndex, 0)[0]);
            Log.d(TAG, "keepClosestMarker: Found closest marker with ID " + (int)ids.get(closestIndex, 0)[0] + " at distance " + String.format("%.3f", minDistance));
            return new Object[]{closestCorner, closestId};
        }
        Log.d(TAG, "keepClosestMarker: No closest marker found.");
        return new Object[]{new ArrayList<Mat>(), new Mat()};
    }

    // A helper class for sorting markers based on their position or distance
    private static class MarkerInfo {
        int id;
        Mat cornerMat;
        double xPosition; // Camera X-coordinate (left/right in image)
        double distanceZ; // Camera Z-coordinate (depth/distance from camera)

        MarkerInfo(int id, Mat cornerMat, double xPosition, double distanceZ) {
            this.id = id;
            this.cornerMat = cornerMat;
            this.xPosition = xPosition;
            this.distanceZ = distanceZ;
        }
    }
}