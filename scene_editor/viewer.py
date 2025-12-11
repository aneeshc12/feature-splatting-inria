#!/usr/bin/env python3
"""
Minimal Gaussian Splat Viewer Prototype
Loads a single .ply splat file and provides interactive camera controls.
Usage: python viewer.py <path_to_splat.ply>
"""

import sys
import os
import torch
import json
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap
from gaussian_renderer import GaussianModel
from render import render


class SimpleCamera:
    """Minimal camera class for rendering gaussian splats."""
    
    def __init__(self, width, height, fovx, fovy, position, look_at, up):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        
        # Build view matrix
        self.position = torch.tensor(position, dtype=torch.float32, device="cuda")
        self.look_at = torch.tensor(look_at, dtype=torch.float32, device="cuda")
        self.up = torch.tensor(up, dtype=torch.float32, device="cuda")
        
        self._update_matrices()
    
    def _update_matrices(self):
        """Compute view and projection matrices."""
        # View matrix (world to camera)
        z_axis = (self.position - self.look_at)
        z_axis = z_axis / torch.norm(z_axis)
        
        x_axis = torch.cross(self.up, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        
        y_axis = torch.cross(z_axis, x_axis)
        
        # Build view matrix
        view = torch.eye(4, dtype=torch.float32, device="cuda")
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[:3, 3] = -torch.stack([
            torch.dot(x_axis, self.position),
            torch.dot(y_axis, self.position),
            torch.dot(z_axis, self.position)
        ])
        
        self.world_view_transform = view.transpose(0, 1)
        
        # Projection matrix
        znear = 0.01
        zfar = 100.0
        
        tanhalffovx = np.tan(self.FoVx * 0.5)
        tanhalffovy = np.tan(self.FoVy * 0.5)
        
        top = tanhalffovy * znear
        bottom = -top
        right = tanhalffovx * znear
        left = -right
        
        proj = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
        proj[0, 0] = 2.0 * znear / (right - left)
        proj[1, 1] = 2.0 * znear / (top - bottom)
        proj[0, 2] = (right + left) / (right - left)
        proj[1, 2] = (top + bottom) / (top - bottom)
        proj[2, 2] = -(zfar + znear) / (zfar - znear)
        proj[2, 3] = -2.0 * zfar * znear / (zfar - znear)
        proj[3, 2] = -1.0
        
        self.projection_matrix = proj.transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.position


class PipelineParams:
    """Minimal pipeline parameters for rendering."""
    def __init__(self):
        self.debug = False
        self.convert_SHs_python = False
        self.compute_cov3D_python = False


class ViewerWindow(QMainWindow):
    """Main viewer window with interactive camera controls."""
    
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Gaussian Splat Viewer")
        
        ply_path = args.ply_path

        # Set args and load 3DGS
        print(f"Loading {ply_path}...")
        self.gaussians = GaussianModel(sh_degree=3, distill_feature_dim=32)
        self.gaussians.load_ply(ply_path)
        print(f"Loaded {len(self.gaussians.get_xyz)} gaussians")
        
        # Rendering parameters
        self.width = 1024
        self.height = 768
        self.pipe = PipelineParams()
        self.bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        
        # Camera parameters
        fov_y = np.radians(50.0)
        fov_x = 2.0 * np.arctan(np.tan(fov_y / 2.0) * self.width / self.height)
        
        # Load camera parameters from file if it exists
        self.camera_file = "camera_pos.txt"
        camera_params = self._load_camera_params()
        
        if camera_params:
            print(f"Loaded camera parameters from {self.camera_file}")
            self.distance = camera_params['distance']
            self.azimuth = camera_params['azimuth']
            self.elevation = camera_params['elevation']
            self.target = np.array(camera_params['target'])
            up_vector = camera_params['up']
        else:
            print("Using default camera parameters")
            self.distance = 2.0
            self.azimuth = 0.0
            self.elevation = 0.0
            self.target = np.array([0.0, 0.0, 0.0])
            up_vector = [0.0, 1.0, 0.0]
        
        self.camera = SimpleCamera(
            self.width, self.height,
            fov_x, fov_y,
            self._compute_camera_position(),
            self.target,
            up_vector
        )
        
        # UI setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(self.width, self.height)
        layout.addWidget(self.image_label)

        self.cam_rotation_label = QLabel(f"Rot: {self.azimuth:.2f}, {self.elevation:.2f}")
        layout.addWidget(self.cam_rotation_label, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Mouse interaction state
        self.last_mouse_pos = None
        self.mouse_button = None
        
        # Keyboard movement state
        self.keys_pressed = set()
        self.movement_timer = QTimer()
        self.movement_timer.timeout.connect(self._process_movement)
        self.movement_timer.start(16)  # ~60 FPS
        self.movement_speed = 0.05  # Units per frame
        
        # Initial render
        self.render_and_display()
        
        print("\nControls:")
        print("  Left mouse + drag: Rotate camera")
        print("  Right mouse + drag: Pan camera")
        print("  Mouse wheel: Zoom in/out")
        print("  WASD: Move camera forward/left/back/right")
        print("  Space/Shift: Move camera up/down")
        print("  U: Set up vector to current screen up direction")
        print("  P: Print current camera parameters (for caching)")
    
    def _compute_camera_position(self):
        """Compute camera position from spherical coordinates."""
        x = self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.distance * np.sin(self.elevation)
        z = self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        return self.target + np.array([x, y, z])
    
    def _load_camera_params(self):
        """Load camera parameters from file if it exists."""
        if not os.path.exists(self.camera_file):
            return None
        
        try:
            with open(self.camera_file, 'r') as f:
                params = json.load(f)
            return params
        except Exception as e:
            print(f"Warning: Could not load camera parameters: {e}")
            return None
    
    def _save_camera_params(self):
        """Save current camera parameters to file."""
        position = self._compute_camera_position()
        up = self.camera.up.cpu().numpy()
        
        params = {
            'position': position.tolist(),
            'target': self.target.tolist(),
            'up': up.tolist(),
            'distance': float(self.distance),
            'azimuth': float(self.azimuth),
            'elevation': float(self.elevation)
        }
        
        try:
            with open(self.camera_file, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"\nCamera parameters saved to {self.camera_file}")
            return True
        except Exception as e:
            print(f"Error saving camera parameters: {e}")
            return False
    
    def _update_camera(self):
        """Update camera after parameter changes."""
        position = self._compute_camera_position()
        self.camera.position = torch.tensor(position, dtype=torch.float32, device="cuda")
        self.camera.look_at = torch.tensor(self.target, dtype=torch.float32, device="cuda")
        self.camera._update_matrices()

        self.cam_rotation_label.setText(f"Rot: {self.azimuth:.2f}, {self.elevation:.2f}")

    
    def _get_camera_vectors(self):
        """Get camera forward, right, and up vectors."""
        forward = self.target - self._compute_camera_position()
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.camera.up.cpu().numpy())
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        return forward, right, up
    
    def _process_movement(self):
        """Process keyboard movement inputs."""
        if not self.keys_pressed:
            return
        
        forward, right, up = self._get_camera_vectors()
        movement = np.zeros(3)
        
        if Qt.Key.Key_W in self.keys_pressed:
            movement += forward * self.movement_speed
        if Qt.Key.Key_S in self.keys_pressed:
            movement -= forward * self.movement_speed
        if Qt.Key.Key_A in self.keys_pressed:
            movement += right * self.movement_speed
        if Qt.Key.Key_D in self.keys_pressed:
            movement -= right * self.movement_speed
        if Qt.Key.Key_Space in self.keys_pressed:
            movement += up * self.movement_speed
        if Qt.Key.Key_Shift in self.keys_pressed:
            movement -= up * self.movement_speed
        
        if np.any(movement != 0):
            # Move both camera position and target together
            cam_pos = self._compute_camera_position()
            new_cam_pos = cam_pos + movement
            self.target = self.target + movement
            
            # Update distance and angles based on new position
            offset = new_cam_pos - self.target
            self.distance = np.linalg.norm(offset)
            
            if self.distance > 0.01:
                self.azimuth = np.arctan2(offset[2], offset[0])
                self.elevation = np.arcsin(np.clip(offset[1] / self.distance, -1.0, 1.0))
            
            self._update_camera()
            self.render_and_display()
    
    def render_and_display(self):
        """Render the scene and display the result."""
        with torch.no_grad():
            render_pkg = render(
                self.camera,
                self.gaussians,
                self.pipe,
                self.bg_color
            )
        
        # Convert rendered image to displayable format
        image = render_pkg["render"]  # Shape: (3, H, W)
        image = torch.clamp(image, 0.0, 1.0)

        image = (image * 255).byte().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
        
        # Ensure contiguous array for QImage
        image = np.ascontiguousarray(image)
        
        # Convert to QPixmap
        h, w, c = image.shape
        bytes_per_line = 3 * w
        
        # Create QImage from bytes (not memoryview)
        q_image = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Keep a reference to prevent garbage collection
        pixmap = QPixmap.fromImage(q_image.copy())
        
        self.image_label.setPixmap(pixmap)
    
    def mousePressEvent(self, event):
        """Handle mouse press for camera control."""
        self.last_mouse_pos = event.pos()
        self.mouse_button = event.button()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag for camera control."""
        if self.last_mouse_pos is None:
            return
        
        delta = event.pos() - self.last_mouse_pos
        dx = delta.x()
        dy = delta.y()
        
        if self.mouse_button == Qt.MouseButton.LeftButton:
            # Rotate camera
            self.azimuth += dx * 0.01
            self.elevation = np.clip(self.elevation - dy * 0.01, -np.pi/2 + 0.1, np.pi/2 - 0.1)
            self._update_camera()
            self.render_and_display()
        
        elif self.mouse_button == Qt.MouseButton.RightButton:
            # Pan camera
            sensitivity = 0.001 * self.distance
            right = np.array([np.cos(self.azimuth + np.pi/2), 0, np.sin(self.azimuth + np.pi/2)])
            up = np.array([0, 1, 0])
            self.target -= right * dx * sensitivity
            self.target += up * dy * sensitivity
            self._update_camera()
            self.render_and_display()
        
        self.last_mouse_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.last_mouse_pos = None
        self.mouse_button = None
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        self.distance *= zoom_factor
        self.distance = np.clip(self.distance, 0.1, 10.0)
        self._update_camera()
        self.render_and_display()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        key = event.key()
        
        # Handle 'U' key to reset up vector
        if key == Qt.Key.Key_U:
            _, right, up = self._get_camera_vectors()
            # Set up vector to current screen up direction
            self.camera.up = torch.tensor(up, dtype=torch.float32, device="cuda")
            self.camera.azimuth = 0.
            self.camera.elevation = 0.
            self._update_camera()
            self.render_and_display()
            print(f"Up vector set to: {up}")
        
        # Handle 'P' key to print camera parameters
        elif key == Qt.Key.Key_P:
            position = self._compute_camera_position()
            up = self.camera.up.cpu().numpy()
            
            # Print to console
            print("\n" + "="*60)
            print("Camera Parameters:")
            print("="*60)
            print(f"Position: {position.tolist()}")
            print(f"Target:   {self.target.tolist()}")
            print(f"Up:       {up.tolist()}")
            print(f"Distance: {self.distance}")
            print(f"Azimuth:  {self.azimuth}")
            print(f"Elevation: {self.elevation}")
            print("="*60)
            
            # Save to file
            if self._save_camera_params():
                print(f"Parameters saved to {self.camera_file}")
            print()
        
        else:
            # Add key to pressed set for movement
            self.keys_pressed.add(key)
    
    def keyReleaseEvent(self, event):
        """Handle key release events."""
        key = event.key()
        self.keys_pressed.discard(key)

