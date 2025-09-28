# Vertical Jump Analyzer with MoveNet

A comprehensive vertical jump analysis tool that uses Google's MoveNet pose detection to analyze jumping performance and calculate lower body explosive strength metrics.

## 🎯 Features

- **Accurate Pose Detection**: Uses MoveNet Thunder model for precise skeletal tracking
- **Jump Detection**: Advanced algorithm to identify and count vertical jumps
- **Explosive Strength Analysis**: Calculate power scores and strength ratings
- **Real-time Visualization**: Live skeleton overlay during video processing
- **Comprehensive Analytics**: Professional-grade analysis graphs and metrics
- **Performance Assessment**: Detailed lower body strength evaluation

## 📊 Analysis Output

The analyzer generates:

### Visual Analysis
- **Hip Height Tracking**: Movement patterns over time with jump detection markers
- **Vertical Velocity Analysis**: Speed and acceleration during jumps
- **Knee Flexion Angles**: Joint angle analysis for technique assessment
- **Performance Summary**: Overall metrics and scoring

### Metrics Calculated
- Total jumps detected
- Maximum velocity and acceleration
- Average explosive power score
- Peak power output
- Power consistency rating
- Overall strength assessment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Video file in MP4 format

### Installation

1. Clone this repository:
```bash
git clone https://github.com/MPkS1/vertical-jump-analyzer.git
cd vertical-jump-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your video:
   - Name your video file `Vertical Jump.mp4`
   - Place it in the project directory

4. Run the analysis:
```bash
python jump_analyzer_enhanced.py
```

## 🎮 Controls During Processing

- **SPACE**: Pause/Resume
- **Q**: Quit analysis
- **S**: Skip 60 frames

## 📈 Understanding Results

### Power Consistency
- **90%+**: Very consistent technique
- **70-90%**: Good consistency
- **50-70%**: Moderate variation
- **<50%**: Highly variable technique

## 🎥 Best Practices for Video Recording

For optimal analysis results:

### Video Quality
- 30+ FPS for smooth tracking
- 1080p or higher resolution
- Stable camera position
- Person fully visible in frame

### Environment
- Good, even lighting
- Simple background (minimal clutter)
- Contrasting colors between person and background

### Jump Technique
- Perform clear, distinct jumps
- Allow 2-3 seconds between jumps
- Full crouch before each jump
- Complete landing before next jump

## 🔧 Technical Details

### AI Model
- **MoveNet Thunder**: Google's high-accuracy pose detection model
- **17 Keypoints**: Full body skeletal tracking
- **Real-time Processing**: ~10-20 FPS depending on hardware

### Analysis Algorithm
1. **Frame Preprocessing**: Resize and normalize for MoveNet input
2. **Pose Detection**: Extract keypoint coordinates and confidence scores
3. **Kinematic Calculation**: Compute hip height, knee angles, velocity, acceleration
4. **Jump Detection**: Pattern recognition for jump identification
5. **Metrics Calculation**: Explosive strength and performance analysis
6. **Visualization**: Real-time overlay and graph generation

## 📋 Output Files

After analysis completion, you'll find:
- `enhanced_jump_analysis_output.mp4` - Processed video with skeleton tracking
- `jump_analysis_enhanced.png` - Comprehensive analysis graphs
- Console output with detailed performance metrics

## ⚙️ Dependencies

- TensorFlow 2.8+
- TensorFlow Hub
- OpenCV
- NumPy
- Matplotlib
- SciPy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Google Research for the MoveNet model
- TensorFlow team for the ML framework
- OpenCV community for computer vision tools

---

**Note**: This tool is designed for fitness and sports analysis. For professional athletic assessment, consult with certified sports scientists.
