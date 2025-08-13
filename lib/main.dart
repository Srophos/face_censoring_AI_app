import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/yolo.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'package:saver_gallery/saver_gallery.dart';
import 'package:share_plus/share_plus.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// --- ISOLATE COMMUNICATION CLASSES ---

/// Data class to hold arguments for spawning the inference isolate.
class IsolateSetup {
  final SendPort mainSendPort;
  final RootIsolateToken token;
  // FIX: Pass the raw bytes for the age model, not the path.
  final Uint8List ageModelBytes;

  IsolateSetup({
    required this.mainSendPort,
    required this.token,
    required this.ageModelBytes,
  });
}

/// Message to send to the inference isolate for prediction.
class InferenceMessage {
  final Uint8List imageBytes;
  final SendPort replyPort; // Port to send the results back to

  InferenceMessage({required this.imageBytes, required this.replyPort});
}

/// Wraps the output from the inference isolate.
class YoloOutput {
  final List<Map<String, dynamic>> results;
  final img.Image image;

  YoloOutput({required this.results, required this.image});
}

/// Input for the save/blur isolate.
class SaveInput {
  final Uint8List imageBytes;
  final List<Map<String, dynamic>> results;
  final List<int> blurIndices;
  final RootIsolateToken? token;

  SaveInput({required this.imageBytes, required this.results, required this.blurIndices, required this.token});
}


// --- ISOLATE ENTRY POINTS ---

/// The entry point for the long-lived inference isolate.
/// Loads the models once and then listens for messages to process.
Future<void> inferenceIsolateEntry(IsolateSetup setup) async {
  // Initialize the messenger to allow access to platform channels (for assets).
  BackgroundIsolateBinaryMessenger.ensureInitialized(setup.token);

  final isolateReceivePort = ReceivePort();

  // Load models ONCE.
  final yolo = YOLO(modelPath: 'yolov8s-face.tflite', task: YOLOTask.detect);
  await yolo.loadModel();

  // FIX: Use the age model bytes passed directly from the main thread.
  final ageInterpreter = Interpreter.fromBuffer(setup.ageModelBytes);


  // --- WARMUP STEP ---
  final dummyImage = img.Image(width: 640, height: 640);
  final dummyBytes = img.encodeJpg(dummyImage);
  await yolo.predict(dummyBytes);


  // Send the isolate's SendPort to the main thread AFTER warmup is complete.
  setup.mainSendPort.send(isolateReceivePort.sendPort);

  // Listen for incoming messages (images to process).
  await for (final message in isolateReceivePort) {
    if (message is InferenceMessage) {
      final imageBytes = message.imageBytes;
      img.Image? originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) continue;

      // --- Image Orientation Correction ---
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

      // --- Stage 1: Face Detection ---
      final resizedImageForYolo = img.copyResize(originalImage, width: 640);
      final yoloInputBytes = img.encodeJpg(resizedImageForYolo);
      final rawResults = await yolo.predict(yoloInputBytes);
      final results = (rawResults['boxes'] as List<dynamic>?)
          ?.map((e) => Map<String, dynamic>.from(e))
          .toList() ?? [];

      // --- Stage 2: Age Classification ---
      final scale = originalImage.width / 640.0;
      for (var res in results) {
        final x1 = (res['x1'] ?? 0.0) * scale;
        final y1 = (res['y1'] ?? 0.0) * scale;
        final x2 = (res['x2'] ?? 0.0) * scale;
        final y2 = (res['y2'] ?? 0.0) * scale;

        final faceCrop = img.copyCrop(originalImage, x: x1.toInt(), y: y1.toInt(), width: (x2 - x1).toInt(), height: (y2 - y1).toInt());
        res['face_crop_bytes'] = img.encodeJpg(faceCrop);

        final ageModelInputImage = img.copyResize(faceCrop, width: 133, height: 133);
        var inputTensor = List.generate(1, (i) => List.generate(133, (j) => List.generate(133, (k) => List.generate(3, (l) => 0.0))));
        for (var y = 0; y < 133; y++) {
          for (var x = 0; x < 133; x++) {
            final pixel = ageModelInputImage.getPixel(x, y);
            inputTensor[0][y][x][0] = pixel.b / 255.0;
            inputTensor[0][y][x][1] = pixel.g / 255.0;
            inputTensor[0][y][x][2] = pixel.r / 255.0;
          }
        }
        var outputTensor = List.generate(1, (i) => [0.0]);
        ageInterpreter.run(inputTensor, outputTensor);

        final adultConfidence = outputTensor[0][0];
        res['age_prediction'] = adultConfidence > 0.5 ? "Teen/Adult (14+)" : "Child (0-13)";
        res['age_confidence'] = adultConfidence;
      }

      // Send the result back to the main isolate.
      message.replyPort.send(YoloOutput(results: results, image: originalImage));
    }
  }
}

/// Isolate for creating the final blurred image.
Future<Uint8List?> createFinalImageInIsolate(SaveInput input) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(input.token!);
  try {
    img.Image? imageToSave = img.decodeImage(input.imageBytes);
    if (imageToSave == null) return null;

    final scale = imageToSave.width / 640.0;

    for (var i = 0; i < input.results.length; i++) {
      if (!input.blurIndices.contains(i)) continue;

      final res = input.results[i];
      final x1 = (res['x1'] ?? 0.0) * scale;
      final y1 = (res['y1'] ?? 0.0) * scale;
      final x2 = (res['x2'] ?? 0.0) * scale;
      final y2 = (res['y2'] ?? 0.0) * scale;

      final face = img.copyCrop(imageToSave, x: x1.toInt(), y: y1.toInt(), width: (x2 - x1).toInt(), height: (y2 - y1).toInt());
      const int maxBlurDimension = 64;
      final img.Image smallFace = img.copyResize(face, width: maxBlurDimension, height: maxBlurDimension, interpolation: img.Interpolation.linear);
      final int blurRadius = (maxBlurDimension / 6).round();
      final img.Image blurredSmallFace = img.gaussianBlur(smallFace, radius: blurRadius);
      final img.Image blurredFace = img.copyResize(blurredSmallFace, width: face.width, height: face.height, interpolation: img.Interpolation.nearest);
      img.compositeImage(imageToSave, blurredFace, dstX: x1.toInt(), dstY: y1.toInt());
    }
    imageToSave.exif.clear();
    return img.encodeJpg(imageToSave);
  } catch (e) {
    debugPrint('Error in createFinalImageInIsolate: $e');
    return null;
  }
}


void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const YOLODemoApp());
}

class YOLODemoApp extends StatefulWidget {
  const YOLODemoApp({super.key});
  @override
  State<YOLODemoApp> createState() => _YOLODemoAppState();
}

class _YOLODemoAppState extends State<YOLODemoApp> {
  ThemeMode _themeMode = ThemeMode.dark;

  @override
  void initState() {
    super.initState();
    _loadTheme();
  }

  Future<void> _loadTheme() async {
    final prefs = await SharedPreferences.getInstance();
    final isDarkMode = prefs.getBool('isDarkMode') ?? true;
    setState(() {
      _themeMode = isDarkMode ? ThemeMode.dark : ThemeMode.light;
    });
  }

  Future<void> _setThemeMode(ThemeMode mode) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('isDarkMode', mode == ThemeMode.dark);
    setState(() {
      _themeMode = mode;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: ThemeData(
        brightness: Brightness.light,
        scaffoldBackgroundColor: Colors.grey[200],
        primaryColor: Colors.blue,
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.blue,
          elevation: 4,
          titleTextStyle: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
          iconTheme: IconThemeData(color: Colors.white),
        ),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF1a2533),
        primaryColor: const Color(0xFF3498db),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF2c3e50),
          elevation: 4,
          titleTextStyle: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
          iconTheme: IconThemeData(color: Colors.white),
        ),
        cardTheme: CardThemeData(
          color: const Color(0xFF2c3e50),
          elevation: 2,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
      home: YOLODemoPage(
        themeMode: _themeMode,
        onThemeChanged: _setThemeMode,
      ),
    );
  }
}

class YOLODemoPage extends StatefulWidget {
  final ThemeMode themeMode;
  final ValueChanged<ThemeMode> onThemeChanged;

  const YOLODemoPage({
    super.key,
    required this.themeMode,
    required this.onThemeChanged,
  });

  @override
  State<YOLODemoPage> createState() => _YOLODemoPageState();
}

class _YOLODemoPageState extends State<YOLODemoPage> with SingleTickerProviderStateMixin {
  static const String ageModelAssetPath = 'assets/best_model.tflite';

  File? selectedImage;
  Uint8List? _selectedImageBytes;
  String? _originalFileName;
  List<Map<String, dynamic>> detectionResults = [];
  bool isLoading = false;
  bool _isSaving = false;
  img.Image? originalImage;
  String? errorMessage;
  PermissionStatus _permissionStatus = PermissionStatus.denied;

  late AnimationController _animationController;
  final List<int> _blurIndices = [];

  bool _isPreviewingBlur = false;
  Uint8List? _previewImageBytes;

  // --- Isolate Management ---
  Isolate? _inferenceIsolate;
  SendPort? _inferenceSendPort;
  bool _isolateReady = false;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _startInferenceIsolate();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
  }

  /// Starts the background isolate and sets up communication.
  Future<void> _startInferenceIsolate() async {
    final mainReceivePort = ReceivePort();
    final token = ServicesBinding.rootIsolateToken!;

    // FIX: Load the age model bytes on the main thread before spawning.
    final ageModelAsset = await rootBundle.load(ageModelAssetPath);
    final ageModelBytes = ageModelAsset.buffer.asUint8List();

    // FIX: Pass the loaded bytes to the isolate.
    final setup = IsolateSetup(
      mainSendPort: mainReceivePort.sendPort,
      token: token,
      ageModelBytes: ageModelBytes,
    );
    _inferenceIsolate = await Isolate.spawn(inferenceIsolateEntry, setup);

    // Wait for the isolate to send back its SendPort.
    mainReceivePort.listen((message) {
      if (message is SendPort) {
        setState(() {
          _inferenceSendPort = message;
          _isolateReady = true;
        });
      }
    });
  }

  @override
  void dispose() {
    _inferenceIsolate?.kill(priority: Isolate.immediate);
    _animationController.dispose();
    super.dispose();
  }

  void _toggleTheme() {
    final newTheme = widget.themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    widget.onThemeChanged(newTheme);
  }

  void _clearSelection() {
    setState(() {
      selectedImage = null;
      _selectedImageBytes = null;
      _originalFileName = null;
      detectionResults = [];
      originalImage = null;
      _blurIndices.clear();
      _isPreviewingBlur = false;
      _previewImageBytes = null;
    });
  }

  void _selectAllForBlur() {
    setState(() {
      _blurIndices.clear();
      _blurIndices.addAll(List.generate(detectionResults.length, (index) => index));
      _previewImageBytes = null;
      _isPreviewingBlur = false;
    });
  }

  void _clearAllBlurSelections() {
    setState(() {
      _blurIndices.clear();
      _previewImageBytes = null;
      _isPreviewingBlur = false;
    });
  }

  Future<void> _requestPermissions() async {
    final status = await Permission.photos.request();
    setState(() {
      _permissionStatus = status;
    });
  }

  Future<void> pickImageAndDetect() async {
    if (!_isolateReady) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Models are still loading, please wait...'))
      );
      return;
    }

    if (_permissionStatus != PermissionStatus.granted) {
      await showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Permission Required'),
          content: const Text('This app needs access to your photo library to select images.'),
          actions: [
            TextButton(child: const Text('Cancel'), onPressed: () => Navigator.of(context).pop()),
            TextButton(child: const Text('Open Settings'), onPressed: () { openAppSettings(); Navigator.of(context).pop(); }),
          ],
        ),
      );
      return;
    }

    final picker = ImagePicker();
    final imageFile = await picker.pickImage(source: ImageSource.gallery, maxHeight: 1600 , maxWidth: 1600);
    if (imageFile == null) return;

    final imageBytes = await imageFile.readAsBytes();

    setState(() {
      isLoading = true;
      errorMessage = null;
      selectedImage = File(imageFile.path);
      _selectedImageBytes = imageBytes;
      _originalFileName = imageFile.name;
      detectionResults = [];
      originalImage = null;
      _blurIndices.clear();
      _isPreviewingBlur = false;
      _previewImageBytes = null;
      _animationController.reset();
    });

    try {
      // Create a port to receive the result from the isolate.
      final replyPort = ReceivePort();
      _inferenceSendPort?.send(InferenceMessage(imageBytes: _selectedImageBytes!, replyPort: replyPort.sendPort));

      // Wait for the result.
      final output = await replyPort.first as YoloOutput;

      if (mounted) {
        setState(() {
          detectionResults = output.results;
          originalImage = output.image;
        });
        _animationController.forward();
        _checkForChildrenAndPrompt();
      }
    } catch (e, s) {
      debugPrint('Error processing image: $e\n$s');
      if (mounted) {
        setState(() => errorMessage = 'Failed to process image: $e');
      }
    } finally {
      if (mounted) {
        setState(() => isLoading = false);
      }
    }
  }

  Future<void> _checkForChildrenAndPrompt() async {
    final childIndices = <int>[];
    for (var i = 0; i < detectionResults.length; i++) {
      if (detectionResults[i]['age_prediction'] == 'Child (0-13)') {
        childIndices.add(i);
      }
    }

    if (childIndices.isNotEmpty && mounted) {
      await showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('${childIndices.length} face(s) under 13 detected'),
          content: const Text('Do you want to blur them out?'),
          actions: [
            TextButton(child: const Text('No'), onPressed: () => Navigator.of(context).pop()),
            TextButton(
              child: const Text('Yes'),
              onPressed: () {
                Navigator.of(context).pop();
                setState(() {
                  _blurIndices.addAll(childIndices);
                  _previewImageBytes = null;
                });
                _toggleBlurPreview();
              },
            ),
          ],
        ),
      );
    }
  }

  Future<void> _toggleBlurPreview() async {
    if (_isPreviewingBlur) {
      setState(() => _isPreviewingBlur = false);
      return;
    }
    if (_previewImageBytes != null) {
      setState(() => _isPreviewingBlur = true);
      return;
    }
    if (_selectedImageBytes == null) return;

    setState(() => _isSaving = true);
    try {
      final token = ServicesBinding.rootIsolateToken;
      final saveInput = SaveInput(imageBytes: _selectedImageBytes!, results: detectionResults, blurIndices: _blurIndices, token: token);
      final Uint8List? finalImageBytes = await compute(createFinalImageInIsolate, saveInput);
      if (mounted) {
        setState(() {
          _previewImageBytes = finalImageBytes;
          _isPreviewingBlur = true;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error creating preview: $e')));
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  Future<void> _saveOrShareImage({required bool isShare}) async {
    if (_selectedImageBytes == null) return;
    setState(() => _isSaving = true);
    try {
      Uint8List? finalImageBytes = _previewImageBytes;
      if (finalImageBytes == null) {
        final token = ServicesBinding.rootIsolateToken;
        final saveInput = SaveInput(imageBytes: _selectedImageBytes!, results: detectionResults, blurIndices: _blurIndices, token: token);
        finalImageBytes = await compute(createFinalImageInIsolate, saveInput);
      }

      if (finalImageBytes != null) {
        final newFileName = 'censored_${_originalFileName ?? 'image_${DateTime.now().millisecondsSinceEpoch}.jpg'}';
        final tempDir = await getTemporaryDirectory();
        final filePath = '${tempDir.path}/$newFileName';
        final file = await File(filePath).writeAsBytes(finalImageBytes);

        if (isShare) {
          final xFile = XFile(file.path, name: newFileName);
          await Share.shareXFiles([xFile], text: 'Check out this picture!');
        } else {
          await SaverGallery.saveFile(file: file.path, name: newFileName, androidRelativePath: 'DCIM/FaceDetections', androidExistNotSave: true);
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Image saved!')));
          }
        }
      } else {
        throw Exception('Failed to create final image.');
      }
    } catch (e, s) {
      debugPrint('Error during save/share: $e\n$s');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Detector AI'),
        actions: _buildAppBarActions(),
      ),
      body: Column(
        children: [
          Expanded(child: AnimatedSwitcher(duration: const Duration(milliseconds: 300), child: _buildContentView())),
          _buildPickerButton(),
        ],
      ),
    );
  }

  List<Widget> _buildAppBarActions() {
    if (_isSaving) {
      return [
        const Padding(
          padding: EdgeInsets.all(16.0),
          child: SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3, color: Colors.white)),
        ),
      ];
    }
    return [
      if (selectedImage != null && detectionResults.isNotEmpty && !isLoading) ...[
        if (_blurIndices.isNotEmpty)
          IconButton(
            icon: Icon(_isPreviewingBlur ? Icons.visibility_off_outlined : Icons.visibility_outlined),
            onPressed: _toggleBlurPreview,
            tooltip: _isPreviewingBlur ? 'Hide Preview' : 'Preview Blur',
          ),
        if (_blurIndices.length < detectionResults.length)
          IconButton(icon: const Icon(Icons.select_all), onPressed: _selectAllForBlur, tooltip: 'Select All for Blurring'),
        if (_blurIndices.isNotEmpty)
          IconButton(icon: const Icon(Icons.deselect_outlined), onPressed: _clearAllBlurSelections, tooltip: 'Clear Blur Selections'),
        IconButton(icon: const Icon(Icons.share), onPressed: () => _saveOrShareImage(isShare: true), tooltip: 'Share Image'),
        IconButton(icon: const Icon(Icons.save_alt_outlined), onPressed: () => _saveOrShareImage(isShare: false), tooltip: 'Save Image'),
      ],
      if (selectedImage != null)
        IconButton(icon: const Icon(Icons.clear), onPressed: _clearSelection, tooltip: 'Clear Image'),
      IconButton(
        icon: Icon(widget.themeMode == ThemeMode.dark ? Icons.light_mode_outlined : Icons.dark_mode_outlined),
        onPressed: _toggleTheme,
        tooltip: 'Toggle Theme',
      ),
    ];
  }

  Widget _buildPickerButton() {
    final isDark = widget.themeMode == ThemeMode.dark;
    final buttonGradient = isDark ? [const Color(0xFF3498db), const Color(0xFF2980b9)] : [Colors.blue, Colors.lightBlueAccent];
    final disabledGradient = [Colors.grey.shade700, Colors.grey.shade600];
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(15),
        child: InkWell(
          onTap: isLoading || !_isolateReady ? null : pickImageAndDetect,
          borderRadius: BorderRadius.circular(15),
          splashColor: Colors.white.withAlpha(50),
          highlightColor: Colors.white.withAlpha(30),
          child: Ink(
            height: 60,
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: isLoading || !_isolateReady ? disabledGradient : buttonGradient, begin: Alignment.topLeft, end: Alignment.bottomRight),
              borderRadius: BorderRadius.circular(15),
              boxShadow: [BoxShadow(color: isDark ? Colors.black.withAlpha(77) : Colors.grey.withAlpha(100), blurRadius: 10, offset: const Offset(0, 5))],
            ),
            child: Center(
              child: _isolateReady
                  ? const Text('Pick Image & Detect', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold))
                  : const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(width: 16),
                  Text('Loading Models...', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContentView() {
    final isDark = widget.themeMode == ThemeMode.dark;
    final placeholderColor = isDark ? Colors.white.withAlpha(128) : Colors.grey.shade600;

    if (errorMessage != null) {
      return Center(child: Padding(padding: const EdgeInsets.all(16.0), child: Text('Error: $errorMessage', textAlign: TextAlign.center, style: const TextStyle(color: Colors.redAccent, fontSize: 16))));
    }
    if (selectedImage == null) {
      return Center(
        key: const ValueKey('placeholder'),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.photo_library_outlined, size: 80, color: placeholderColor),
            const SizedBox(height: 20),
            Text('Select an image to begin', style: TextStyle(fontSize: 18, color: placeholderColor)),
          ],
        ),
      );
    }
    return Stack(
      key: const ValueKey('results'),
      children: [
        Column(
          children: [
            Expanded(
              flex: 3,
              child: Container(
                margin: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: LayoutBuilder(builder: (context, constraints) {
                  return Stack(
                    fit: StackFit.expand,
                    children: [
                      Container(
                        decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), border: Border.all(color: isDark ? Colors.grey.shade800 : Colors.grey.shade300)),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(11),
                          child: AnimatedSwitcher(
                            duration: const Duration(milliseconds: 300),
                            child: (_isPreviewingBlur && _previewImageBytes != null)
                                ? Image.memory(_previewImageBytes!, key: const ValueKey('preview'), fit: BoxFit.contain)
                                : CustomPaint(
                              key: const ValueKey('live'),
                              foregroundPainter: FacePainter(image: originalImage, results: detectionResults, animation: _animationController, blurIndices: _blurIndices),
                              child: Image.file(selectedImage!, fit: BoxFit.contain),
                            ),
                          ),
                        ),
                      ),
                      if (!_isPreviewingBlur) ..._buildTappableBoxes(context, constraints.biggest),
                    ],
                  );
                }),
              ),
            ),
            Expanded(
              flex: 2,
              child: ListView.builder(
                padding: const EdgeInsets.all(8),
                itemCount: detectionResults.length,
                itemBuilder: (context, index) {
                  final detection = detectionResults[index];
                  final isSelectedForBlur = _blurIndices.contains(index);
                  final agePrediction = detection['age_prediction'] ?? 'N/A';
                  final ageConfidence = detection['age_confidence'] ?? 0.0;
                  final confidenceString = 'Confidence (Adult): ${(ageConfidence * 100).toStringAsFixed(1)}%';
                  final faceCropBytes = detection['face_crop_bytes'] as Uint8List?;
                  return Card(
                    color: isSelectedForBlur ? Theme.of(context).primaryColor.withAlpha(100) : null,
                    child: ListTile(
                      onTap: () => setState(() {
                        if (isSelectedForBlur) {_blurIndices.remove(index);}
                        else {_blurIndices.add(index);}
                        _previewImageBytes = null; _isPreviewingBlur = false;
                      }),
                      leading: (faceCropBytes != null) ? CircleAvatar(backgroundImage: MemoryImage(faceCropBytes), radius: 30) : const Icon(Icons.face, size: 40),
                      title: Text('FACE #${index + 1} - $agePrediction'),
                      subtitle: Text(confidenceString),
                      trailing: Checkbox(
                        value: isSelectedForBlur,
                        onChanged: (bool? value) => setState(() {
                          if (value == true)
                            {
                              _blurIndices.add(index);
                            }
                          else {
                            _blurIndices.remove(index);
                          }
                          _previewImageBytes = null; _isPreviewingBlur = false;
                        }),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
        if (isLoading) Container(color: Colors.black.withAlpha(180), child: const Center(child: CircularProgressIndicator())),
      ],
    );
  }

  List<Widget> _buildTappableBoxes(BuildContext context, Size canvasSize) {
    if (originalImage == null) return [];
    final imageAspectRatio = originalImage!.width / originalImage!.height;
    final canvasAspectRatio = canvasSize.width / canvasSize.height;
    Rect imageRect;
    if (canvasAspectRatio > imageAspectRatio) {
      final scaledWidth = canvasSize.height * imageAspectRatio;
      final offsetX = (canvasSize.width - scaledWidth) / 2;
      imageRect = Rect.fromLTWH(offsetX, 0, scaledWidth, canvasSize.height);
    } else {
      final scaledHeight = canvasSize.width / imageAspectRatio;
      final offsetY = (canvasSize.height - scaledHeight) / 2;
      imageRect = Rect.fromLTWH(0, offsetY, canvasSize.width, scaledHeight);
    }
    return detectionResults.asMap().entries.map((entry) {
      final index = entry.key;
      final res = entry.value;
      final x1 = res['x1'] ?? 0.0; final y1 = res['y1'] ?? 0.0;
      final x2 = res['x2'] ?? 0.0; final y2 = res['y2'] ?? 0.0;
      final scale = originalImage!.width / 640.0;
      final scaledX1 = x1 * scale; final scaledY1 = y1 * scale;
      final scaledX2 = x2 * scale; final scaledY2 = y2 * scale;
      final finalScaleX = imageRect.width / originalImage!.width;
      final finalScaleY = imageRect.height / originalImage!.height;
      final left = imageRect.left + scaledX1 * finalScaleX;
      final top = imageRect.top + scaledY1 * finalScaleY;
      final right = imageRect.left + scaledX2 * finalScaleX;
      final bottom = imageRect.top + scaledY2 * finalScaleY;
      final tappableRect = Rect.fromLTRB(left, top, right, bottom);
      final isHighlighted = _blurIndices.contains(index);
      return Positioned.fromRect(
        rect: tappableRect,
        child: GestureDetector(
          onTap: () => setState(() {
            if (isHighlighted) {
              _blurIndices.remove(index);
            }
            else {
              _blurIndices.add(index);
            }
            _previewImageBytes = null; _isPreviewingBlur = false;
          }),
          child: Container(decoration: BoxDecoration(color: isHighlighted ? Colors.white.withAlpha(50) : Colors.transparent, border: isHighlighted ? Border.all(color: Colors.white, width: 2) : null)),
        ),
      );
    }).toList();
  }
}

class FacePainter extends CustomPainter {
  final img.Image? image;
  final List<Map<String, dynamic>> results;
  final Animation<double> animation;
  final List<int> blurIndices;

  FacePainter({required this.image, required this.results, required this.animation, required this.blurIndices}) : super(repaint: animation);

  @override
  void paint(Canvas canvas, Size size) {
    if (image == null) return;
    final imageAspectRatio = image!.width / image!.height;
    final canvasAspectRatio = size.width / size.height;
    Rect imageRect;
    if (canvasAspectRatio > imageAspectRatio) {
      final scaledWidth = size.height * imageAspectRatio;
      final offsetX = (size.width - scaledWidth) / 2;
      imageRect = Rect.fromLTWH(offsetX, 0, scaledWidth, size.height);
    } else {
      final scaledHeight = size.width / imageAspectRatio;
      final offsetY = (size.height - scaledHeight) / 2;
      imageRect = Rect.fromLTWH(0, offsetY, size.width, scaledHeight);
    }
    for (var i = 0; i < results.length; i++) {
      final res = results[i];
      final x1 = res['x1'] ?? 0.0; final y1 = res['y1'] ?? 0.0;
      final x2 = res['x2'] ?? 0.0; final y2 = res['y2'] ?? 0.0;
      final scale = image!.width / 640.0;
      final scaledX1 = x1 * scale; final scaledY1 = y1 * scale;
      final scaledX2 = x2 * scale; final scaledY2 = y2 * scale;
      final finalScaleX = imageRect.width / image!.width;
      final finalScaleY = imageRect.height / image!.height;
      final left = imageRect.left + scaledX1 * finalScaleX;
      final top = imageRect.top + scaledY1 * finalScaleY;
      final right = imageRect.left + scaledX2 * finalScaleX;
      final bottom = imageRect.top + scaledY2 * finalScaleY;
      final rect = Rect.fromLTRB(left, top, right, bottom);
      final animatedRect = Rect.lerp(Rect.fromCenter(center: rect.center, width: 0, height: 0), rect, animation.value);
      if (animatedRect != null) {
        final paint = Paint()
          ..color = const Color(0xFF00FF00)
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke;
        canvas.drawRect(animatedRect, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant FacePainter oldDelegate) {
    return oldDelegate.results != results || oldDelegate.blurIndices != blurIndices;
  }
}
