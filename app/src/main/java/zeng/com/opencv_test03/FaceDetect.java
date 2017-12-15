package zeng.com.opencv_test03;

import android.graphics.Bitmap;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by Administrator on 2017/12/15.
 */

public class FaceDetect {
    public static final int LANDMARK_NUM = 68;

    public class FaceRect {
        public int left;
        public int top;
        public int bottom;
        public int right;
    }

    public class PixelPonit {
        public int x;
        public int y;
    }

    public class FaceLandmark {
        public PixelPonit[]  points;
        FaceLandmark() { points = new PixelPonit[LANDMARK_NUM]; }
    }

    List<FaceRect> face_rect_list;
    //List<FaceLandmark> face_landmark_list;

    public void init(String modelPath) {
        nativeInit(modelPath);
    }

    public int detect(Bitmap bitmapImage) {
        int w = bitmapImage.getWidth(), h=bitmapImage.getHeight();
        int[] pix = new int[w*h];
        bitmapImage.getPixels(pix,0, w, 0, 0, w, h);

        int faceNum = nativeFaceDetect(pix,w,h); // Face detect
        return faceNum;

    }

    public List<FaceLandmark> getFaceLandmark() {
        List<FaceLandmark> face_landmark_list = new LinkedList<FaceLandmark>();
        int[] flmArray = new int[LANDMARK_NUM*2];

        int faceNum = nativeGetFaceNum();
        for(int i=0; i<faceNum; i++) {
            nativeGetFaceLandmark(i, flmArray);
            FaceLandmark faceLandmark = new FaceLandmark();
            for(int j=0; j<LANDMARK_NUM; j++) {
                faceLandmark.points[j].x = flmArray[j*2];
                faceLandmark.points[j].y = flmArray[j*2+1];
            }
            face_landmark_list.add(faceLandmark);
        }
        return face_landmark_list;

    }


    public void tracker(Bitmap bitmapImage) {

    }

    public void trackerClean() {

    }

    public native int nativeInit(String modelPath);
    public native int nativeFaceDetect(int[] srcImg,  int width, int height);
    public native int nativeGetFaceNum();
    public native int nativeGetFaceLandmark(int index, int[] faceLandmark);
}
