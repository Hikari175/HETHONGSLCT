import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from predictor import Predictor
import nltk

class SpellingCorrectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Spelling Correction App")
        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        self.correct_button = QPushButton("Correct")
        self.correct_button.clicked.connect(self.correctText)
        layout.addWidget(self.correct_button)

        self.result_edit = QTextEdit()
        layout.addWidget(self.result_edit)

        self.setLayout(layout)

        self.loadModel()

    def loadModel(self):
        print("Loading model ...")
        nltk.download('punkt')
        self.model = Predictor(weight_path='weights/seq2seq.pth', have_att=True)

    def correctText(self):
        text_input = self.text_edit.toPlainText().strip()
        text_corrected = self.model.spelling_correct(text_input)
        self.result_edit.setText(text_corrected)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    spelling_correction_app = SpellingCorrectionApp()
    spelling_correction_app.show()
    sys.exit(app.exec_())
