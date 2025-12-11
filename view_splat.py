import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from scene_editor.viewer import ViewerWindow

# include argparse parsing for ply_path
from argparse import ArgumentParser

# set up argparse

def main():
    parser = ArgumentParser(description="Gaussian Splat Viewer")
    parser.add_argument("--ply_path", type=str, help="Path to the .ply file containing the splat data")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    window = ViewerWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
