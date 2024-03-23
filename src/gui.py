import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import random
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
import pandas as pd


class LivePlotCanvas(FigureCanvas):
     def __init__(self, parent=None, width=18.51, height=6.51):
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super(LivePlotCanvas, self).__init__(fig)

class Ui_MainWindow(object):
        
    #Setting up main window to hold widgets
    def setupUi(self, MainWindow):
        self.root_dir = Path(__file__).resolve().parent
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1920, 1080)
        # Opening Stylesheet for the GUI
        with open(Path(self.root_dir / "guistyle.css"), "r") as f:
                MainWindow.setStyleSheet(f.read())

        self.data = list(pd.read_csv(self.root_dir / 'data.csv').iloc[:, -1])
        #variable which will hold ai output
        ai_output = ""
                
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        #Frame widget which contains the Graph
        self.frame = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(32, 30, 1851, 651))
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        #create a horizontal layout to align the graph in the frame
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        #integrating graph into the frame
        self.canvas = LivePlotCanvas(self)
        self.y_data = [375 for x in range(500*4)]
        self.x_data = [x/500 for x in range(0, 500*4)]
        self._plot_ref = None
        self.canvas.axes.set_facecolor('#2a2e32')
        self.canvas.figure.set_facecolor('#2a2e32')
        self.canvas.axes.set_ylim(0, 800)
        self.canvas.axes.xaxis.set_visible(False)
        self.canvas.axes.yaxis.set_visible(False)
        self.update_plot()
        self.horizontalLayout_4.addWidget(self.canvas)
        self.timer = QtCore.QTimer(self.centralwidget)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.check_for_plot_updates)
        self.timer.start()
        self.timer2 = QtCore.QTimer(self.centralwidget)
        self.timer2.setInterval(100)
        self.timer2.timeout.connect(self.ai_indicator)
        self.timer2.start()

        #Setting up frame that will contain the AI output
        self.frame_2 = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(60, 700, 881, 331))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setObjectName("frame_2")
        #elipses to signify that the AI is loading
        self.label = QtWidgets.QLabel(parent=self.frame_2)
        self.label.setGeometry(QtCore.QRect(18, 15, 851, 301))
        self.label.setText(ai_output)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(450, 280, 31, 41))
        self.label_2.setStyleSheet("font-size: 36px")
        self.label_2.setObjectName("label_2")
        #Patient info frame
        self.frame_3 = QtWidgets.QFrame(parent=self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(980, 700, 881, 331))
        self.frame_3.setStyleSheet("")
        self.frame_3.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_3.setObjectName("frame_3")
        #Adding Form Entries
        self.textEdit = QtWidgets.QTextEdit(parent=self.frame_3)
        self.textEdit.setGeometry(QtCore.QRect(580, 20, 281, 201))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(parent=self.frame_3)
        self.pushButton.setGeometry(QtCore.QRect(350, 260, 241, 51))
        self.pushButton.setObjectName("pushButton")
        #Calling reset function on click
        self.pushButton.clicked.connect(self.reset)
        #Form layout settings and fields
        self.formLayoutWidget = QtWidgets.QWidget(parent=self.frame_3)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 20, 551, 231))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setVerticalSpacing(37)
        self.formLayout.setObjectName("formLayout")
        self.patientNameLabel = QtWidgets.QLabel(parent=self.formLayoutWidget)
        self.patientNameLabel.setObjectName("patientNameLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.patientNameLabel)
        self.sexLabel = QtWidgets.QLabel(parent=self.formLayoutWidget)
        self.sexLabel.setObjectName("sexLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.sexLabel)
        self.sexLineEdit = QtWidgets.QLineEdit(parent=self.formLayoutWidget)
        self.sexLineEdit.setObjectName("sexLineEdit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.sexLineEdit)
        self.AgeLabel = QtWidgets.QLabel(parent=self.formLayoutWidget)
        self.AgeLabel.setObjectName("AgeLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.AgeLabel)
        self.ageLineEdit = QtWidgets.QLineEdit(parent=self.formLayoutWidget)
        self.ageLineEdit.setObjectName("ageLineEdit")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.ageLineEdit)
        self.DoB = QtWidgets.QLabel(parent=self.formLayoutWidget)
        self.DoB.setObjectName("DoB")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.DoB)
        self.patientNameLineEdit = QtWidgets.QLineEdit(parent=self.formLayoutWidget)
        self.patientNameLineEdit.setObjectName("patientNameLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.patientNameLineEdit)
        self.DoBlineEdit = QtWidgets.QLineEdit(parent=self.formLayoutWidget)
        self.DoBlineEdit.setObjectName("DoBlineEdit")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.DoBlineEdit)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        #label widget which contains BPM data
        self.BPM = QtWidgets.QLabel(parent=self.frame)
        self.BPM.setGeometry(QtCore.QRect(1650, 545, 85, 100))
        self.BPM.setAlignment(QtCore.Qt.AlignRight)
        self.BPM.setObjectName("BPM")
        self.BPM.setText("000")
        self.BPM.adjustSize()

    #function to change colour of bordor when ai has an output currently configured to a button for testing
    def ai_indicator(self):
        inference = pd.read_csv(self.root_dir / "inference.csv")
        if inference.iloc[0, 0] > -1:
            self.frame_2.setStyleSheet("border-color: #00cc00;")
            self.label_2.hide()
            if inference.iloc[0, 0] == 0:
                self.label.setText(f'\tDiagnosis: Normal\n\tConfidence: {inference.iloc[0, 1] * 100:.1f}%\n\n\tTo maintain a healthy heart, the NHS recommends:\n\t   \u2022 Regular exercise\n\t   \u2022 Avoiding fatty foods\n\t   \u2022 Try to reduce stress')
            if inference.iloc[0, 0] == 1:
                self.label.setText(f'\tDiagnosis: Extrasystole\n\tConfidence: {inference.iloc[0, 1] * 100:.1f}%\n\n\tExtra heart sounds can be caused by a variety of factors such as:\n\t     Anxiety, stress and fatigue\n\tSymptoms include:\n\t     Palpitations, dizziness, weakness and difficulty breaathing\n\tTreatments include:\n\t     Exercise, beta-blockers and treatment for underlying cause of extrasystole')
            if inference.iloc[0, 0] == 2:
                self.label.setText(f'\tDiagnosis: Murmur\n\tConfidence: {inference.iloc[0, 1] * 100:.1f}%\n\n\tMurmurs can be caused by a variety of factors such as:\n\t     Fever, anemia, hyperthyroidism, rapid growth, exercise, pregnancy, congenital heart\n\t     defects, degenerative valve disease, endocarditis and rheumatic fever\n\tSymptoms include:\n\t     Discolouration of fingernails and/or lips, chest pain, persistent cough, dizziness, swelling\n\t     of liver and/or neck veins, fainting, heavy sweating, shortness of breath and sudden\n\t     weight gain\n\tTreatments include:\n\t     Anti-arrhythmic medications, ACE inhibitors or ARBs, blood thinners, antibiotics, surgery\n\t     or cardiac catheterisation')    
    
    #reset fields fucntion
    def reset(self):
        self.ageLineEdit.clear()
        self.patientNameLineEdit.clear()
        self.sexLineEdit.clear()
        self.DoBlineEdit.clear()
        self.textEdit.clear()
        
    def check_for_plot_updates(self):
        try:
            if (data := list(pd.read_csv(self.root_dir / 'data.csv').iloc[:, -1])) != self.data:
                self.data = data
                bpm = int(self.data[-1])
                self.BPM.setText(f'{bpm: >3}')
                self.update_plot()
        except pd.errors.EmptyDataError:
            pass
    #Initiate translations during UI setup 
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "..."))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'Manrope\'; font-size:16pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Manrope\';\"><br /></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Save"))
        self.patientNameLabel.setText(_translate("MainWindow", "Name"))
        self.sexLabel.setText(_translate("MainWindow", "Sex"))
        self.AgeLabel.setText(_translate("MainWindow", "Age"))
        self.DoB.setText(_translate("MainWindow", "DoB"))
     
    #Plotting Graph function
    def update_plot(self):
        if self._plot_ref is None:
            plot_refs = self.canvas.axes.plot(self.x_data, self.y_data, 'r')
            self._plot_ref = plot_refs[0]

        else:
            self.y_data = self.y_data[400:] + self.data[:400]
            self._plot_ref.set_ydata(self.y_data)
        #graph colour
        self._plot_ref.set_color('#ffffff')

        self.canvas.draw()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.showFullScreen()
    sys.exit(app.exec())
