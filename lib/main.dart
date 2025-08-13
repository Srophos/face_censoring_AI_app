import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

// Data class for passing info to the isolate
class YoloInput {
  final String modelPath;
  final Uint8List imageBytes;
  final RootIsolateToken? token;

  YoloInput({required this.modelPath, required this.imageBytes, required this.token});
}

// Data class for receiving info from the isolate
class YoloOutput {
  final List<Map<String, dynamic>> results;
  final Size imageSize;

  YoloOutput({required this.results, required this.imageSize});
}

/// This function runs in a separate isolate.
Future<YoloOutput?> runYoloInIsolate(YoloInput input) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(input.token!);

  final yolo = YOLO(modelPath: input.modelPath, task: YOLOTask.detect);
  await yolo.loadModel();

  img.Image? originalImage = img.decodeImage(input.imageBytes);
  if (originalImage == null) return null;
  
  // Apply EXIF orientation
  final orientationTag = originalImage.exif.getTag(0x0112);
  final int orientation = orientationTag?.toInt() ?? 1;
  switch (orientation) {
      case 2: originalImage = img.flipHorizontal(originalImage); break;
      case 3: originalImage = img.copyRotate(originalImage, angle: 180); break;
      case 4: originalImage = img.flipVertical(originalImage); break;
      case 5: originalImage = img.copyRotate(originalImage, angle: 90); originalImage = img.flipHorizontal(originalImage); break;
      case 6: originalImage = img.copyRotate(originalImage, angle: 90); break;
      case 7: originalImage = img.copyRotate(originalImage, angle: -90); originalImage = img.flipHorizontal(originalImage); break;
      case 8: originalImage = img.copyRotate(originalImage, angle: -90); break;
  }
  
  final resizedImage = img.copyResize(originalImage, width: 640);
  final correctedImageBytes = img.encodeJpg(resizedImage);
  final rawResults = await yolo.predict(correctedImageBytes);

  // --- THE FIX: Return the dimensions of the RESIZED image ---
  final resizedImageSize = Size(resizedImage.width.toDouble(), resizedImage.height.toDouble());

  final results = (rawResults?['boxes'] as List<dynamic>?)
      ?.map((e) => Map<String, dynamic>.from(e))
      .toList() ?? [];

  return YoloOutput(results: results, imageSize: resizedImageSize);
}

void main() => runApp(const YOLODemo());

class YOLODemo extends StatefulWidget {
  const YOLODemo({super.key});

  @override
  State<YOLODemo> createState() => _YOLODemoState();
}

class _YOLODemoState extends State<YOLODemo> {
  static const String modelPath = 'assets/models/yolov8s-face_final.tflite';

  File? selectedImage;
  List<Map<String, dynamic>> detectionResults = [];
  bool isLoading = false;
  Size? sourceImageSize; // This will now hold the RESIZED image's dimensions
  String? errorMessage;

  Future<void> pickImageAndDetect() async {
    final picker = ImagePicker();
    final imageFile = await picker.pickImage(source: ImageSource.gallery);
    if (imageFile == null) return;

    setState(() {
      isLoading = true;
      errorMessage = null;
      selectedImage = File(imageFile.path);
      detectionResults = [];
      sourceImageSize = null;
    });

    try {
      final imageBytes = await selectedImage!.readAsBytes();
      
      final token = ServicesBinding.rootIsolateToken;
      final yoloInput = YoloInput(modelPath: modelPath, imageBytes: imageBytes, token: token);
      
      final YoloOutput? output = await compute(runYoloInIsolate, yoloInput);
      
      setState(() {
        if (output != null) {
          detectionResults = output.results;
          sourceImageSize = output.imageSize; // Get the RESIZED size
        }
      });
    } catch (e) {
      setState(() => errorMessage = 'Failed to process image: $e');
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Face Detection Demo')),
        body: Column(
          children: [
            Expanded(
              child: _buildContentView(),
            ),
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: ElevatedButton(
                onPressed: isLoading ? null : pickImageAndDetect,
                child: const Text('Pick Image & Detect'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContentView() {
    if (isLoading && selectedImage == null) {
      return const Center(child: Text('App Ready. Please pick an image.'));
    }
    if (errorMessage != null) {
      return Center(
          child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text('Error: $errorMessage',
                  style: const TextStyle(color: Colors.red))));
    }
    if (selectedImage == null) {
      return const Center(child: Text('Please pick an image.'));
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        Container(
          margin: const EdgeInsets.all(16.0),
          decoration: BoxDecoration(border: Border.all(color: Colors.grey)),
          child: CustomPaint(
            foregroundPainter: FacePainter(
              imageSize: sourceImageSize, // Pass the RESIZED size
              results: detectionResults,
            ),
            child: Image.file(selectedImage!, fit: BoxFit.contain),
          ),
        ),
        if (isLoading)
          Container(
            color: Colors.black.withAlpha(128),
            child: const Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }
}

/// The painter now correctly uses the RESIZED image's dimensions as its source.
class FacePainter extends CustomPainter {
  final Size? imageSize; // This is the size of the source image for the coordinates (e.g., 640x480)
  final List<Map<String, dynamic>> results;

  FacePainter({required this.imageSize, required this.results});

  @override
  void paint(Canvas canvas, Size size) { // `size` is the size of the widget on screen
    if (imageSize == null) return;

    final imageAspectRatio = imageSize!.width / imageSize!.height;
    final canvasAspectRatio = size.width / size.height;
    double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (canvasAspectRatio > imageAspectRatio) {
      scale = size.height / imageSize!.height;
      offsetX = (size.width - imageSize!.width * scale) / 2;
    } else {
      scale = size.width / imageSize!.width;
      offsetY = (size.height - imageSize!.height * scale) / 2;
    }

    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    for (final res in results) {
      final x1 = res['x1'] ?? 0.0;
      final y1 = res['y1'] ?? 0.0;
      final x2 = res['x2'] ?? 0.0;
      final y2 = res['y2'] ?? 0.0;
      
      // Scale the coordinates from the resized image space to the canvas space
      final left = x1 * scale + offsetX;
      final top = y1 * scale + offsetY;
      final right = x2 * scale + offsetX;
      final bottom = y2 * scale + offsetY;
      
      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}