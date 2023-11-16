from GUI_main import MyGUI
import sys,argparse
from PyQt5.QtWidgets import QApplication


def get_stylesheet():
    # External variables for colors
    background_color = '#f8f8f8'
    tab_pane_background_color = '#585858'
    accent_color = '#d5d5e5'  # A shade of gray
    accent_color_darker = '#b0b0e5'
    text_color = '#333333'
    border_radius = '0px'
    border_width = '1px'
    padding_small = '1px'#'2px'
    padding_medium = '2px'#'4px'
    padding_large = '3px'#'6px'
    margin_small = '1px'#'2px'
    margin_medium = '1px'#'5px'
    margin_between_groupbox_and_entry = '8px'  # Adjust as needed

    # Stylesheet
    stylesheet = (
        f"QWidget {{ background-color: {background_color}; margin: {margin_small}; }}"
        f"QLabel, QLineEdit {{ color: {text_color}; }}"
        f"QPushButton {{ background-color: {accent_color}; color: {text_color}; "
        f"border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_large}; }}"
        f"QPushButton:hover {{ background-color: {accent_color_darker}; }}"
        f"QPushButton:pressed {{ background-color: {accent_color}; }}"
        f"QLineEdit {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
        f"QComboBox {{ border: {border_width} solid {accent_color}; border-radius: {border_radius}; padding: {padding_small}; }}"
        f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right;}}"
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
    gui.setStyleSheet(get_stylesheet())
    gui.show()
    sys.exit(app.exec_())