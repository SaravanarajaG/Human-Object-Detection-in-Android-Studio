package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 1;
    private static final int REQUEST_STORAGE_PERMISSION = 100;

    private ImageView imageView;
    private TextView textViewResult;
    private Interpreter tfliteInterpreter;
    private List<String> flowerLabels;
    private ObjectDetector mlKitObjectDetector;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        Button buttonUpload = findViewById(R.id.buttonUpload);
        textViewResult = findViewById(R.id.textViewResult);

        // Load TFLite model and labels
        try {
            tfliteInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "flower_model.tflite"));
            flowerLabels = FileUtil.loadLabels(this, "labels.txt");
        } catch (IOException e) {
            Log.e("ModelError", "Error loading model or labels: " + e.getMessage());
            textViewResult.setText("Error loading model or labels");
        }

        // Configure ML Kit Object Detector (without classification)
        ObjectDetectorOptions options = new ObjectDetectorOptions.Builder()
                .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                .enableMultipleObjects()
                .build();
        mlKitObjectDetector = ObjectDetection.getClient(options);

        buttonUpload.setOnClickListener(view -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQUEST_STORAGE_PERMISSION);
            } else {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openGallery();
        } else {
            textViewResult.setText("Permission denied");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            if (imageUri != null) {
                try {
                    Bitmap bitmap = resizeImage(imageUri); // Resize the image
                    analyzeImage(bitmap);
                } catch (IOException e) {
                    Log.e("ImageError", "Error resizing image: " + e.getMessage());
                    textViewResult.setText("Error resizing image");
                }
            } else {
                textViewResult.setText("Error: Image selection failed");
            }
        }
    }

    private Bitmap resizeImage(Uri imageUri) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        InputStream inputStream = getContentResolver().openInputStream(imageUri);
        BitmapFactory.decodeStream(inputStream, null, options);

        options.inSampleSize = calculateInSampleSize(options, 1024, 1024); // Target size
        options.inJustDecodeBounds = false;

        inputStream = getContentResolver().openInputStream(imageUri);
        return BitmapFactory.decodeStream(inputStream, null, options);
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        int height = options.outHeight;
        int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    private void analyzeImage(Bitmap bitmap) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        // Initialize result text
        StringBuilder resultText = new StringBuilder();

        // Custom model detection
        resultText.append(detectObjectsWithCustomModel(bitmap, mutableBitmap));

        // ML Kit object detection
        detectObjectsWithMLKit(bitmap, mutableBitmap);

        // Update TextView with the combined results
        textViewResult.setText(resultText.toString());

        // Display the annotated image
        imageView.setImageBitmap(mutableBitmap);
    }

    private void detectObjectsWithMLKit(Bitmap bitmap, Bitmap mutableBitmap) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);

        mlKitObjectDetector.process(image)
                .addOnSuccessListener(detectedObjects -> {
                    Paint paint = new Paint();
                    paint.setColor(Color.GREEN);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(5);

                    Canvas canvas = new Canvas(mutableBitmap);

                    for (DetectedObject obj : detectedObjects) {
                        RectF boundingBox = new RectF(obj.getBoundingBox());
                        canvas.drawRect(boundingBox, paint);
                    }

                    imageView.setImageBitmap(mutableBitmap);
                })
                .addOnFailureListener(e -> Log.e("MLKitError", "Error in ML Kit detection: " + e.getMessage()));
    }

    private String detectObjectsWithCustomModel(Bitmap bitmap, Bitmap mutableBitmap) {
        if (tfliteInterpreter == null || flowerLabels == null) {
            return "Error: Custom model or labels not loaded properly.\n";
        }

        StringBuilder result = new StringBuilder();

        try {
            ByteBuffer inputBuffer = preprocessImage(bitmap);

            int outputBoxCount = 10;
            int[] outputShape = {1, outputBoxCount, 6};
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputShape, org.tensorflow.lite.DataType.FLOAT32);

            tfliteInterpreter.run(inputBuffer, outputBuffer.getBuffer());

            float[] output = outputBuffer.getFloatArray();

            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5);

            Paint textPaint = new Paint();
            textPaint.setColor(Color.WHITE);
            textPaint.setTextSize(40);

            Canvas canvas = new Canvas(mutableBitmap);

            for (int i = 0; i < output.length / 6; i++) {
                float score = output[i * 6 + 4];
                int classId = (int) output[i * 6 + 5];

                if (score > 0.5f) {
                    float left = output[i * 6] * mutableBitmap.getWidth();
                    float top = output[i * 6 + 1] * mutableBitmap.getHeight();
                    float right = output[i * 6 + 2] * mutableBitmap.getWidth();
                    float bottom = output[i * 6 + 3] * mutableBitmap.getHeight();

                    RectF boundingBox = new RectF(left, top, right, bottom);
                    canvas.drawRect(boundingBox, paint);

                    String label = flowerLabels.get(classId) + " (" + String.format("%.2f", score) + ")";
                    canvas.drawText(label, left, top - 10, textPaint);

                    result.append(label).append("\n");
                }
            }
        } catch (Exception e) {
            Log.e("DetectionError", "Error during detection: " + e.getMessage());
        }
        return result.toString();
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true); // Ensure size matches the model
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[224 * 224];
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return byteBuffer;
    }
}