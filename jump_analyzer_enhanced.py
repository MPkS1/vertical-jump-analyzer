import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import math
from collections import deque
import time
import os

# Try to import TensorFlow and TensorFlow Hub for MoveNet
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported for MoveNet")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

class EnhancedJumpAnalyzer:
    def __init__(self):
        print("Initializing Enhanced Jump Analyzer with MoveNet...")
        
        self.movenet_available = TENSORFLOW_AVAILABLE
        
        if self.movenet_available:
            try:
                print("Loading MoveNet model... (this may take a moment)")
                # Use MoveNet Thunder for better accuracy
                self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
                self.movenet = self.model.signatures['serving_default']
                print("✅ MoveNet model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load MoveNet: {e}")
                self.movenet_available = False
        
        # MoveNet keypoint indices
        self.KEYPOINT_DICT = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Enhanced skeleton connections for better visualization
        self.CONNECTIONS = [
            ('left_ear', 'left_eye'), ('left_eye', 'nose'), ('nose', 'right_eye'), ('right_eye', 'right_ear'),
            ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        # Analysis variables
        self.keypoints_history = []
        self.jump_count = 0
        self.confidence_threshold = 0.3
        
        # Enhanced jump detection variables
        self.hip_heights = []
        self.knee_flexion_angles = []
        self.jump_phases = []
        self.jump_timestamps = []
        self.jump_heights = []
        self.jump_durations = []
        
        # Movement analysis
        self.velocity_history = deque(maxlen=30)
        self.acceleration_history = deque(maxlen=30)
        self.vertical_displacement = []
        
        # Jump detection parameters
        self.min_jump_height = 0.03  # Minimum vertical movement
        self.min_crouch_depth = 20   # Minimum knee bend in degrees
        self.jump_detection_buffer = []
        self.last_jump_frame = -50
        
        # Performance metrics
        self.max_velocity = 0
        self.max_acceleration = 0
        self.explosive_power_scores = []
        
        print(f"Enhanced Analyzer initialized. MoveNet available: {self.movenet_available}")
    
    def preprocess_for_movenet(self, frame):
        """Preprocess frame for MoveNet input"""
        input_size = 256
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = tf.image.resize_with_pad(frame_rgb, input_size, input_size)
        frame_resized = tf.cast(frame_resized, dtype=tf.int32)
        return tf.expand_dims(frame_resized, axis=0)
    
    def detect_pose_movenet(self, frame):
        """Detect pose using MoveNet"""
        if not self.movenet_available:
            return None
        
        try:
            input_tensor = self.preprocess_for_movenet(frame)
            outputs = self.movenet(input_tensor)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]
            return keypoints
        except Exception as e:
            print(f"MoveNet detection error: {e}")
            return None
    
    def draw_enhanced_skeleton(self, frame, keypoints, frame_num):
        """Draw enhanced skeleton with better visualization"""
        if keypoints is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Create a copy for overlay
        overlay = frame.copy()
        
        # Draw connections with varying thickness based on confidence
        for start_point, end_point in self.CONNECTIONS:
            start_idx = self.KEYPOINT_DICT[start_point]
            end_idx = self.KEYPOINT_DICT[end_point]
            
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            if start_kp[2] > self.confidence_threshold and end_kp[2] > self.confidence_threshold:
                start_pos = (int(start_kp[1] * width), int(start_kp[0] * height))
                end_pos = (int(end_kp[1] * width), int(end_kp[0] * height))
                
                # Color coding based on body part
                if 'hip' in start_point or 'hip' in end_point or 'knee' in start_point or 'knee' in end_point:
                    color = (0, 255, 255)  # Yellow for legs (focus area)
                    thickness = 4
                elif 'shoulder' in start_point or 'shoulder' in end_point:
                    color = (255, 0, 0)  # Blue for torso
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for other parts
                    thickness = 2
                
                cv2.line(overlay, start_pos, end_pos, color, thickness)
        
        # Draw keypoints with labels
        for i, (y, x, confidence) in enumerate(keypoints):
            if confidence > self.confidence_threshold:
                pos = (int(x * width), int(y * height))
                
                # Enhanced color coding
                if i in [11, 12]:  # Hips
                    color = (0, 0, 255)  # Red
                    radius = 10
                elif i in [13, 14]:  # Knees
                    color = (255, 0, 255)  # Magenta
                    radius = 8
                elif i in [15, 16]:  # Ankles
                    color = (0, 255, 255)  # Yellow
                    radius = 6
                else:
                    color = (0, 255, 0)  # Green
                    radius = 4
                
                # Draw keypoint
                cv2.circle(overlay, pos, radius, color, -1)
                cv2.circle(overlay, pos, radius + 2, (255, 255, 255), 2)
                
                # Add confidence score for key points
                if i in [11, 12, 13, 14, 15, 16]:  # Lower body points
                    cv2.putText(overlay, f"{confidence:.2f}", 
                               (pos[0] + 15, pos[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Blend overlay with original frame
        result = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        return result
    
    def calculate_enhanced_metrics(self, keypoints):
        """Calculate enhanced jump analysis metrics"""
        if keypoints is None:
            return None, None, None
        
        # Get hip height
        hip_height = self.get_hip_height(keypoints)
        
        # Get knee flexion angles for both legs
        left_knee_angle = self.calculate_knee_flexion_angle(keypoints, 'left')
        right_knee_angle = self.calculate_knee_flexion_angle(keypoints, 'right')
        
        # Average knee angle
        if left_knee_angle is not None and right_knee_angle is not None:
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        elif left_knee_angle is not None:
            avg_knee_angle = left_knee_angle
        elif right_knee_angle is not None:
            avg_knee_angle = right_knee_angle
        else:
            avg_knee_angle = None
        
        return hip_height, avg_knee_angle, keypoints
    
    def calculate_knee_flexion_angle(self, keypoints, side='left'):
        """Calculate knee flexion angle for specified side"""
        if keypoints is None:
            return None
        
        hip_idx = self.KEYPOINT_DICT[f'{side}_hip']
        knee_idx = self.KEYPOINT_DICT[f'{side}_knee']
        ankle_idx = self.KEYPOINT_DICT[f'{side}_ankle']
        
        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]
        
        if hip[2] < self.confidence_threshold or knee[2] < self.confidence_threshold or ankle[2] < self.confidence_threshold:
            return None
        
        # Calculate vectors
        thigh_vector = np.array([hip[1] - knee[1], hip[0] - knee[0]])
        shin_vector = np.array([ankle[1] - knee[1], ankle[0] - knee[0]])
        
        # Calculate angle
        cos_angle = np.dot(thigh_vector, shin_vector) / (np.linalg.norm(thigh_vector) * np.linalg.norm(shin_vector))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def get_hip_height(self, keypoints):
        """Get average hip height"""
        if keypoints is None:
            return None
        
        left_hip = keypoints[self.KEYPOINT_DICT['left_hip']]
        right_hip = keypoints[self.KEYPOINT_DICT['right_hip']]
        
        if left_hip[2] > self.confidence_threshold and right_hip[2] > self.confidence_threshold:
            return (left_hip[0] + right_hip[0]) / 2
        
        return None
    
    def enhanced_jump_detection(self, frame_num):
        """Enhanced jump detection algorithm"""
        if len(self.hip_heights) < 30:
            return False
        
        # Smooth the hip height data
        if len(self.hip_heights) >= 10:
            smoothed_heights = savgol_filter(self.hip_heights[-30:], 
                                           window_length=min(9, len(self.hip_heights[-30:])), 
                                           polyorder=3)
        else:
            smoothed_heights = self.hip_heights[-30:]
        
        # Calculate velocity and acceleration
        if len(smoothed_heights) >= 3:
            velocity = np.gradient(smoothed_heights)
            acceleration = np.gradient(velocity)
            
            current_velocity = velocity[-1]
            current_acceleration = acceleration[-1] if len(acceleration) > 0 else 0
            
            self.velocity_history.append(current_velocity)
            self.acceleration_history.append(current_acceleration)
            
            # Update max values
            self.max_velocity = max(self.max_velocity, abs(current_velocity))
            self.max_acceleration = max(self.max_acceleration, abs(current_acceleration))
        
        # Jump detection logic
        if len(self.hip_heights) >= 20 and frame_num - self.last_jump_frame > 30:
            recent_heights = np.array(self.hip_heights[-20:])
            height_change = np.max(recent_heights[:10]) - np.min(recent_heights[-10:])
            
            # Check for significant upward movement
            if height_change > self.min_jump_height:
                # Verify with knee flexion
                if len(self.knee_flexion_angles) >= 20:
                    recent_knee_angles = self.knee_flexion_angles[-20:]
                    min_angle = np.min(recent_knee_angles)
                    
                    # Check for proper crouch (knee flexion)
                    if min_angle < 140:  # Significant knee bend
                        self.jump_count += 1
                        self.last_jump_frame = frame_num
                        self.jump_timestamps.append(frame_num)
                        self.jump_heights.append(height_change)
                        
                        # Calculate explosive power for this jump
                        if len(self.velocity_history) > 0 and len(self.acceleration_history) > 0:
                            power_score = (abs(current_velocity) * 1000 + 
                                         abs(current_acceleration) * 500 + 
                                         (180 - min_angle) * 2)
                            self.explosive_power_scores.append(power_score)
                        
                        print(f"Jump #{self.jump_count} detected at frame {frame_num}")
                        print(f"  Height change: {height_change:.3f}")
                        print(f"  Min knee angle: {min_angle:.1f}°")
                        print(f"  Velocity: {current_velocity:.4f}")
                        
                        return True
        
        return False
    
    def calculate_explosive_strength(self):
        """Calculate comprehensive explosive strength metrics"""
        if len(self.explosive_power_scores) == 0:
            return {
                'avg_explosive_power': 0,
                'peak_power': 0,
                'power_consistency': 0,
                'strength_rating': 'Insufficient Data'
            }
        
        avg_power = np.mean(self.explosive_power_scores)
        peak_power = np.max(self.explosive_power_scores)
        power_std = np.std(self.explosive_power_scores) if len(self.explosive_power_scores) > 1 else 0
        power_consistency = max(0, 100 - (power_std / avg_power * 100)) if avg_power > 0 else 0
        
        # Strength rating based on explosive power
        if avg_power > 800:
            strength_rating = 'Excellent'
        elif avg_power > 600:
            strength_rating = 'Very Good'
        elif avg_power > 400:
            strength_rating = 'Good'
        elif avg_power > 200:
            strength_rating = 'Fair'
        else:
            strength_rating = 'Needs Improvement'
        
        return {
            'avg_explosive_power': round(avg_power, 2),
            'peak_power': round(peak_power, 2),
            'power_consistency': round(power_consistency, 1),
            'strength_rating': strength_rating
        }
    
    def create_comprehensive_analysis_graphs(self, output_path="jump_analysis_enhanced.png"):
        """Create comprehensive analysis graphs matching the reference format"""
        if len(self.hip_heights) < 20:
            print("Not enough data for analysis graphs")
            return
        
        # Create figure with specific layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        # 1. Hip Height Over Time with Jump Detection
        ax1 = fig.add_subplot(gs[0, 0])
        frames = range(len(self.hip_heights))
        ax1.plot(frames, self.hip_heights, 'b-', linewidth=2, label='Hip Height')
        
        # Mark jump points
        for jump_frame in self.jump_timestamps:
            if jump_frame < len(self.hip_heights):
                ax1.axvline(x=jump_frame, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax1.plot(jump_frame, self.hip_heights[jump_frame], 'ro', markersize=8)
        
        ax1.set_title('Hip Height Over Time with Jump Detection', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Hip Height (normalized)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Invert so up is up
        
        # 2. Vertical Velocity Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.velocity_history) > 0:
            velocity_frames = range(len(list(self.velocity_history)))
            velocity_data = list(self.velocity_history)
            ax2.plot(velocity_frames, velocity_data, 'g-', linewidth=2)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Mark peak velocities
            if len(velocity_data) > 10:
                peaks, _ = find_peaks(np.abs(velocity_data), height=np.std(velocity_data))
                for peak in peaks:
                    ax2.plot(peak, velocity_data[peak], 'go', markersize=6)
        
        ax2.set_title('Vertical Velocity Analysis', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Velocity (up is positive)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Knee Flexion Angles
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.knee_flexion_angles) > 0:
            knee_frames = range(len(self.knee_flexion_angles))
            ax3.plot(knee_frames, self.knee_flexion_angles, 'r-', linewidth=2)
            
            # Add reference lines
            ax3.axhline(y=160, color='orange', linestyle='--', alpha=0.7, label='Standing')
            ax3.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Deep Crouch')
            
            # Mark minimum angles (deepest crouches)
            if len(self.knee_flexion_angles) > 10:
                min_peaks, _ = find_peaks(-np.array(self.knee_flexion_angles), height=20)
                for peak in min_peaks:
                    ax3.plot(peak, self.knee_flexion_angles[peak], 'ro', markersize=6)
        
        ax3.set_title('Knee Flexion Angles', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Angle (degrees)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Summary
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Get explosive strength metrics
        strength_metrics = self.calculate_explosive_strength()
        
        metric_names = ['Jumps', 'Max Vel\n(×1000)', 'Max Acc\n(×100)', 'Consistency']
        metric_values = [
            self.jump_count,
            self.max_velocity * 1000,
            self.max_acceleration * 100,
            strength_metrics['power_consistency']
        ]
        
        colors = ['blue', 'green', 'red', 'orange']
        bars = ax4.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(metric_values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Performance Summary', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add overall title
        fig.suptitle('Accurate Jump Analysis with MoveNet Pose Detection', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced analysis graphs saved to {output_path}")
        
        # Print detailed metrics
        print("\nDetailed Performance Metrics:")
        print(f"Total Jumps: {self.jump_count}")
        print(f"Average Explosive Power: {strength_metrics['avg_explosive_power']}")
        print(f"Peak Power: {strength_metrics['peak_power']}")
        print(f"Power Consistency: {strength_metrics['power_consistency']}%")
        print(f"Strength Rating: {strength_metrics['strength_rating']}")
    
    def process_video_enhanced(self, video_path, output_path=None):
        """Process video with enhanced analysis"""
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        paused = False
        
        print("Processing with Enhanced MoveNet Analysis...")
        print("Controls: 'q' to quit, SPACE to pause, 's' to skip 60 frames")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                
                # Progress updates
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames} - "
                          f"Time: {elapsed:.1f}s - Jumps: {self.jump_count}")
                
                # Detect pose
                keypoints = self.detect_pose_movenet(frame)
                
                # Calculate metrics
                hip_height, knee_angle, _ = self.calculate_enhanced_metrics(keypoints)
                
                # Store data
                if hip_height is not None:
                    self.hip_heights.append(hip_height)
                if knee_angle is not None:
                    self.knee_flexion_angles.append(knee_angle)
                
                # Detect jumps
                new_jump = self.enhanced_jump_detection(frame_count)
                
                # Draw enhanced skeleton
                frame_with_analysis = self.draw_enhanced_skeleton(frame, keypoints, frame_count)
                
                # Add comprehensive info overlay
                self.add_info_overlay(frame_with_analysis, frame_count, total_frames, 
                                    hip_height, knee_angle, new_jump)
                
                if out:
                    out.write(frame_with_analysis)
            else:
                frame_with_analysis = frame.copy()
                cv2.putText(frame_with_analysis, "PAUSED - Press SPACE to continue",
                           (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            # Display
            cv2.imshow('Enhanced Jump Analysis', frame_with_analysis)
            
            # Handle controls
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s') and not paused:
                frame_count = min(frame_count + 60, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Generate final analysis
        processing_time = time.time() - start_time
        strength_metrics = self.calculate_explosive_strength()
        
        print("\n" + "="*80)
        print("ENHANCED JUMP ANALYSIS RESULTS")
        print("="*80)
        print(f"📊 Total Jumps Detected: {self.jump_count}")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"🚀 Max Velocity: {self.max_velocity:.4f}")
        print(f"⚡ Max Acceleration: {self.max_acceleration:.4f}")
        print(f"💪 Average Explosive Power: {strength_metrics['avg_explosive_power']}")
        print(f"🏆 Peak Power: {strength_metrics['peak_power']}")
        print(f"📈 Power Consistency: {strength_metrics['power_consistency']}%")
        print(f"⭐ Strength Rating: {strength_metrics['strength_rating']}")
        
        # Create comprehensive graphs
        self.create_comprehensive_analysis_graphs()
        
        return {
            'jumps_detected': self.jump_count,
            'processing_time': processing_time,
            'strength_metrics': strength_metrics,
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration
        }
    
    def add_info_overlay(self, frame, frame_count, total_frames, hip_height, knee_angle, new_jump):
        """Add comprehensive information overlay"""
        overlay_info = [
            "Enhanced MoveNet Jump Analysis",
            f"Jumps Detected: {self.jump_count}",
            f"Frame: {frame_count}/{total_frames}",
            f"Hip Height: {hip_height:.3f}" if hip_height else "Hip: N/A",
            f"Knee Angle: {knee_angle:.1f}°" if knee_angle else "Knee: N/A",
            f"Velocity: {list(self.velocity_history)[-1]:.4f}" if self.velocity_history else "Vel: N/A",
        ]
        
        # Get explosive strength
        strength_metrics = self.calculate_explosive_strength()
        overlay_info.append(f"Power Score: {strength_metrics['avg_explosive_power']:.1f}")
        overlay_info.append(f"Rating: {strength_metrics['strength_rating']}")
        
        if new_jump:
            overlay_info.append("🚀 JUMP DETECTED!")
        
        # Draw overlay with enhanced styling
        y_pos = 30
        for i, line in enumerate(overlay_info):
            # Color coding
            if "JUMP DETECTED" in line:
                color, thickness, font_scale = (0, 255, 0), 3, 0.8
            elif i == 0:  # Title
                color, thickness, font_scale = (255, 255, 0), 2, 0.7
            elif "Rating:" in line:
                rating = line.split(": ")[1]
                if rating in ["Excellent", "Very Good"]:
                    color = (0, 255, 0)
                elif rating in ["Good", "Fair"]:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                thickness, font_scale = 2, 0.6
            else:
                color, thickness, font_scale = (255, 255, 255), 1, 0.6
            
            # Add background for better readability
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(frame, (5, y_pos + i * 35 - 20), 
                         (text_size[0] + 15, y_pos + i * 35 + 5), (0, 0, 0), -1)
            
            cv2.putText(frame, line, (10, y_pos + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def main():
    print("="*80)
    print("ENHANCED VERTICAL JUMP ANALYZER WITH MOVENET")
    print("="*80)
    
    analyzer = EnhancedJumpAnalyzer()
    
    if not analyzer.movenet_available:
        print("⚠️  MoveNet not available, please install TensorFlow and TensorFlow Hub")
        return
    
    video_path = "Vertical Jump.mp4"
    output_path = "enhanced_jump_analysis_output.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    results = analyzer.process_video_enhanced(video_path, output_path)
    
    if results:
        print(f"\n✅ Enhanced Analysis Complete!")
        print(f"🎥 Output video: {output_path}")
        print(f"📊 Analysis graphs: jump_analysis_enhanced.png")
        print(f"💪 Lower Body Explosive Strength Rating: {results['strength_metrics']['strength_rating']}")
    else:
        print("\n❌ Analysis failed")

if __name__ == "__main__":
    main()