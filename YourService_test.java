package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.core.Mat;

// new imports
import android.util.Log;

import java.util.List;
import java.util.ArrayList;


// new OpenCV imports
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.Aruco;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import java.io.InputStream;
import java.io.IOException;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.Core;


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */

public class YourService extends KiboRpcService {

    // The TAG is used for logging.
// You can use it to check the log in the Android Studio.
    private final String TAG = this.getClass().getSimpleName();

    // 圖片模板
    private final String[] TEMPLATE_FILE_NAMES = {
            "coin.png",
            "compass.png",
            "coral.png",
            "crystal.png",
            "diamond.png",
            "emerald.png",
            "fossil.png",
            "key.png",
            "letter.png",
            "shell.png",
            "treasure_box.png"
    };

    private final String[] TEMPLATE_NAMES = {
            "coin",
            "compass",
            "coral",
            "crystal",
            "diamond",
            "emerald",
            "fossil",
            "key",
            "letter",
            "shell",
            "treasure_box"
    };

    private final Point[] AREA_POINTS = {
            new Point(10.9d, -9.92284d, 5.195d),         // point 1
            new Point(10.925d, -8.875d, 4.602d),            // point 2
            new Point(10.925d, -7.925d, 4.60093d),          // point 3
            new Point(10.766d, -6.852d, 4.945d)             // point 4
    };

    private final Quaternion[] AREA_QUATERNIONS = {
            new Quaternion(0f, 0f, -0.707f, 0.707f),        // 轉向 1
            new Quaternion(0f, 0.707f, 0f, 0.707f),         // 轉向 2
            new Quaternion(0f, 0.707f, 0f, 0.707f),         // 轉向 3
            new Quaternion(0f, 0f, 1f, 0f)                  // 轉向 4
    };

    @Override
    protected void runPlan1() {
        Log.i(TAG, "Start mission");
        api.startMission();

        String[] foundItems = new String[AREA_POINTS.length];
        int[] itemCounts = new int[AREA_POINTS.length];

        // 載入模板
        Mat[] templates = new Mat[TEMPLATE_FILE_NAMES.length];
        for (int i = 0; i < TEMPLATE_FILE_NAMES.length; i++) {
            try {
                InputStream inputStream = getAssets().open(TEMPLATE_FILE_NAMES[i]);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                templates[i] = mat;
                inputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error loading template image: " + TEMPLATE_FILE_NAMES[i], e);
            }
        }

        for (int areaId = 0; areaId < AREA_POINTS.length; areaId++) {
            api.moveTo(AREA_POINTS[areaId], AREA_QUATERNIONS[areaId], false);

            // Step 1: 取得原始影像
            Mat rawImage = api.getMatNavCam();

            // Step 2: 相機校正 - 使用內參與畸變係數校正影像
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);

            Mat undistortedImage = new Mat();
            Calib3d.undistort(rawImage, undistortedImage, cameraMatrix, distCoeffs);
            api.saveMatImage(undistortedImage, "area_" + (areaId + 1) + ".png");

            // Step 3: ArUco 偵測（可略過不使用）
            Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
            List<Mat> corners = new ArrayList<>();
            Mat ids = new Mat();
            Aruco.detectMarkers(undistortedImage, dictionary, corners, ids);

            // Step 4: Template Matching
            int[] numMatches = new int[TEMPLATE_FILE_NAMES.length];
            for (int tempNum = 0; tempNum < templates.length; tempNum++) {
                int matchCount = 0;
                List<org.opencv.core.Point> matchLocations = new ArrayList<>();
                Mat template = templates[tempNum].clone();
                Mat targetImg = undistortedImage.clone();

                for (int size = 20; size <= 100; size += 5) {
                    for (int angle = 0; angle < 360; angle += 45) {
                        Mat resizedTemplate = scalingresizeImage(template, size);
                        Mat rotatedTemplate = rotImg(resizedTemplate, angle);

                        Mat result = new Mat();
                        Imgproc.matchTemplate(targetImg, rotatedTemplate, result, Imgproc.TM_CCOEFF_NORMED);
                        double threshold = 0.7;
                        double maxVal = Core.minMaxLoc(result).maxVal;

                        if (maxVal >= threshold) {
                            Mat thresholdResult = new Mat();
                            Imgproc.threshold(result, thresholdResult, threshold, 1, Imgproc.THRESH_TOZERO);

                            for (int y = 0; y < thresholdResult.rows(); y++) {
                                for (int x = 0; x < thresholdResult.cols(); x++) {
                                    if (thresholdResult.get(y, x)[0] > 0) {
                                        matchLocations.add(new org.opencv.core.Point(x, y));
                                    }
                                }
                            }
                            thresholdResult.release();
                        }

                        result.release();
                        rotatedTemplate.release();
                        resizedTemplate.release();
                    }
                }

                List<org.opencv.core.Point> filteredMatches = removeDuplicates(matchLocations);
                matchCount += filteredMatches.size();
                numMatches[tempNum] = matchCount;
                template.release();
                targetImg.release();
            }

            int mostMatchTemplateNum = getMxIndex(numMatches);
            foundItems[areaId] = TEMPLATE_NAMES[mostMatchTemplateNum];
            itemCounts[areaId] = numMatches[mostMatchTemplateNum];

            api.setAreaInfo(areaId + 1, foundItems[areaId], itemCounts[areaId]);
            Log.i(TAG, "Area " + (areaId + 1) + ": Found " + itemCounts[areaId] + " of " + foundItems[areaId]);

            // 清除資源
            rawImage.release();
            undistortedImage.release();
            cameraMatrix.release();
            distCoeffs.release();
            ids.release();
            for (Mat corner : corners) corner.release();
        }

        // 與太空人互動
        // 先與太空人互動
        Point astronautPoint = new Point(11.143, -6.7607, 4.9654);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(astronautPoint, astronautQuaternion, false);
        api.reportRoundingCompletion();

        // 再靠近一點拍清楚照片
        Point photoPoint = new Point(11.00, -6.7607, 4.9654);
        api.moveTo(photoPoint, astronautQuaternion, false);
        Mat clearerImage = api.getMatNavCam();
        api.saveMatImage(clearerImage, "target_item.png");

        Mat targetImage = api.getMatNavCam();
        api.saveMatImage(targetImage, "target_item.png");
        String targetItem = identifyTargetItem(targetImage, templates);
        api.notifyRecognitionItem();

        int targetArea = 0;
        for (int i = 0; i < foundItems.length; i++) {
            if (foundItems[i].equals(targetItem)) {
                targetArea = i;
                break;
            }
        }

        api.moveTo(AREA_POINTS[targetArea], AREA_QUATERNIONS[targetArea], false);
        api.takeTargetItemSnapshot();

        targetImage.release();
        for (Mat template : templates) {
            if (template != null) template.release();
        }
    }

    // 輔助函式：目標物自動辨識
    private String identifyTargetItem(Mat targetImage, Mat[] templates) {
        double threshold = 0.7;
        int bestIndex = -1;
        double bestScore = 0;

        for (int i = 0; i < templates.length; i++) {
            Mat template = templates[i];
            Mat result = new Mat();
            Imgproc.matchTemplate(targetImage, template, result, Imgproc.TM_CCOEFF_NORMED);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

            if (mmr.maxVal > bestScore && mmr.maxVal >= threshold) {
                bestScore = mmr.maxVal;
                bestIndex = i;
            }

            result.release();
        }

        if (bestIndex != -1) {
            return TEMPLATE_NAMES[bestIndex];
        } else {
            return "unknown";
        }
    }


    /*@Override
        protected void runPlan2 (){
            // write your plan 2 here.
        }

        @Override
        protected void runPlan3(){
            // write your plan 3 here.
        }*/

    private Mat scalingresizeImage(Mat image, int width) {
        int height = (int) ((double) image.rows() / image.cols() * width);
        // Create a new Mat object to hold the resized image
        Mat resizedImage = new Mat();

        // Resize the image using the specified width and height
        Imgproc.resize(image, resizedImage, new Size(width, height));

        return resizedImage;
    }

    private Mat rotImg(Mat img, int angle) {
        // Get the center of the image
        org.opencv.core.Point center = new org.opencv.core.Point(img.cols() / 2, img.rows() / 2);

        // Create a rotation matrix
        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        // Create a new Mat object to hold the rotated image
        Mat rotatedImg = new Mat();

        // Rotate the image using the rotation matrix
        Imgproc.warpAffine(img, rotatedImg, rotMat, img.size());

        // Release resources
        rotMat.release();

        return rotatedImg;
    }


    /**
     * 將匹配點中距離小於指定門檻的視為重複，僅保留彼此間距較遠的唯一點。
     *
     * @param points 原始匹配點列表（可能包含重複或重疊）
     * @return 去除重複後的唯一點列表
     */
    private List<org.opencv.core.Point> removeDuplicates(List<org.opencv.core.Point> points) {
        double length = 10; // 10 px 內視為重複
        List<org.opencv.core.Point> uniquePoints = new ArrayList<>();
        for (org.opencv.core.Point point : points) {
            boolean isIncluded = false;
            for (org.opencv.core.Point uniquePoint : uniquePoints) {
                double distance = calculateDistance(point, uniquePoint);
                if (distance <= length){
                    isIncluded = true;
                    break;
                }
            }
            if (!isIncluded) {
                uniquePoints.add(point);
            }
        }
        return uniquePoints;
    }


    //  計算兩個 2D 點（OpenCV Point）之間的歐式距離
    private double calculateDistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        return Math.sqrt(dx * dx + dy * dy);
    }



    // 回傳整數陣列中最大值的索引位置  (選出哪一個模板最匹配)
    private int getMxIndex(int[] array) {
        int max = array[0];
        int maxIndex = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

}