from GUI_main import MyGUI
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    gui = MyGUI()
    gui.show()
    sys.exit(app.exec_())