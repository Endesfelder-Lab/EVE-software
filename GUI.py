from GUI_main import MyGUI
import sys,argparse,colorsys
from PyQt5.QtWidgets import QApplication
import multiprocessing
from PyQt5.QtGui import QIcon
import os

def adjust_color_brightness(hex_color, percent):
    # Convert hexadecimal color to RGB
    rgb_color = tuple(int(hex_color[1+i:1+i+2], 16) for i in (0, 2, 4))
    # Convert HSL color back to RGB 
    rgb_color = tuple(int(round(i + i * percent / 100)) for i in rgb_color)
    # Bound each element between 0 and 255
    rgb_color = tuple(max(0, min(255, i)) for i in rgb_color)
    # Convert RGB color to hexadecimal
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
    
    return hex_color

def get_stylesheet():
    # External variables for colors
    background_color = '#f8f8f8'
    tab_pane_background_color = '#585858'
    accent_color = '#d5d5e5'  
    accent_color_darker = adjust_color_brightness(accent_color, -15)#'#b0b0e5'
    text_color = '#333333'
    border_radius = '0px'
    border_width = '1px'
    padding_small = '1px'#'2px'
    padding_medium = '2px'#'4px'
    padding_large = '3px'#'6px'
    margin_small = '1px'#'2px'
    margin_medium = '3px'#'5px'
    margin_between_groupbox_and_entry = '8px'  # Adjust as needed

    # Stylesheet
    stylesheet = (
        f"QWidget {{ margin: {margin_small}; }}"
        f"QGroupBox {{ background-color: {background_color}; }}"
        f"QLayout {{ background-color: {background_color}; }}"

        f"QLabel, QLineEdit, QComboBox, QCheckBox {{ color: {text_color}; }}"
        f"QPushButton {{ background-color: {accent_color}; color: {text_color}; "
        f"border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_large}; }}"
        f"QPushButton:hover {{ background-color: {accent_color_darker}; }}"
        f"QPushButton:pressed {{ background-color: {accent_color}; }}"
        f"QLineEdit {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
        f"QComboBox {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
        f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right;}}"
        f"QComboBox QAbstractItemView {{ selection-background-color: #0078D7; selection-color: {text_color};}}" 
        f"QTabWidget::pane {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; margin: 0; }}"
        f"QTabBar::tab {{ background-color: {background_color}; color: {text_color}; border: {border_width} solid {accent_color}; border-top: none; border-bottom-left-radius: {border_radius}; border-bottom-right-radius: {border_radius}; padding: {padding_medium}; margin-right: {margin_small}; }}"
        f"QTabBar::tab:selected {{ background-color: {accent_color_darker}; color: {text_color}; border-top: {border_width} solid {accent_color}; }}"
        f"QTabBar::tab:hover {{ background-color: {accent_color}; color: {text_color}; }}"
        f"QGroupBox {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; margin-top: {margin_between_groupbox_and_entry}; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: {margin_small}; padding: 0 {padding_medium}; background-color: {accent_color}; color: {text_color}; border: {border_width} solid {accent_color}; border-radius: {border_radius}; }}"
        
    )

    return stylesheet

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    gui = MyGUI()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui.setWindowIcon(QIcon(current_dir + os.sep + 'Eve.png'))
    gui.setStyleSheet(get_stylesheet())
    gui.show()
    sys.exit(app.exec_())