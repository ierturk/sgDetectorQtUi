#include "mainwindow.h"
#include "ui_mainwindow.h"

Ort::Env ortEnv = Ort::Env(nullptr);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow) {

    ui->setupUi(this);
    qtimerRight = new QTimer(this);
    qtimerLeft = new QTimer(this);

    connect(ui->btnPlayRight, &QAbstractButton::clicked, this, [this]{on_btnPlay_clicked(true);});
    connect(ui->btnPlayLeft, &QAbstractButton::clicked, this, [this]{on_btnPlay_clicked(false);});
    connect(ui->btnPauseRight, &QAbstractButton::clicked, this, [this]{on_btnPause_clicked(true);});
    connect(ui->btnPauseLeft, &QAbstractButton::clicked, this, [this]{on_btnPause_clicked(false);});


    ortEnv = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OrtEnv");

    ortNetRight = new OrtNet();
    ortNetLeft = new OrtNet();

    ortNetRight->Init("/home/ierturk/Work/REPOs/ssd/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx");
    ortNetLeft->Init("/home/ierturk/Work/REPOs/ssd/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx");
}

MainWindow::~MainWindow() {
    delete ui;
}


QString MainWindow::getFormattedTime(int timeInSeconds) {
    int seconds = (int) (timeInSeconds) % 60 ;
    int minutes = (int) ((timeInSeconds / 60) % 60);
    int hours   = (int) ((timeInSeconds / (60*60)) % 24);
    QTime t(hours, minutes, seconds);
    if (hours == 0 )
        return t.toString("mm:ss");
    else
        return t.toString("h:mm:ss");
}


void MainWindow::processFrameAndUpdateGUI(bool side) {

    cv::Mat frame;
    if(side) {
        captureRight.read(frame);
        if(frame.empty()) return;

        ortNetRight->setInputTensor(frame);
        ortNetRight->forward();
        ui->lblTimerRight->setText(
                    getFormattedTime(
                        (int)captureRight.get(cv::CAP_PROP_POS_FRAMES)
                        / (int)captureRight.get(cv::CAP_PROP_FPS)));

        ui->lblPlayerRight->setPixmap(QPixmap::fromImage(ortNetRight->getProcessedFrame()));
        ui->lblPlayerRight->setScaledContents(true);
        ui->lblPlayerRight->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );
    } else {
        captureLeft.read(frame);
        if(frame.empty()) return;

        ortNetLeft->setInputTensor(frame);
        ortNetLeft->forward();
        ui->lblTimerLeft->setText(
                    getFormattedTime(
                        (int)captureLeft.get(cv::CAP_PROP_POS_FRAMES)
                        / (int)captureLeft.get(cv::CAP_PROP_FPS)));

        ui->lblPlayerLeft->setPixmap(QPixmap::fromImage(ortNetLeft->getProcessedFrame()));
        ui->lblPlayerLeft->setScaledContents(true);
        ui->lblPlayerLeft->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );

    }
}

void MainWindow::on_btnPlay_clicked(bool side)
{
    QString fname = QFileDialog::getOpenFileName(
                        this,
                        tr("Open Images"),
                        "/home/ierturk/Work/REPOs/data/farmData/",
                        tr("mp4 File (*.mp4);; avi File (*.avi)"));

    std::string stream = "filesrc location=" + fname.toStdString() + " ! decodebin ! videoconvert ! videoflip method=clockwise ! appsink";

    if(side) {
        captureRight.open(stream, cv::CAP_GSTREAMER);
        if(captureRight.isOpened() == false) {
            std::cout << "error: Right side capture is not accessed successfully";
            return;
        } else {

            ui->lblTimerRight->setText(
                        getFormattedTime(
                            (int)captureRight.get(cv::CAP_PROP_POS_FRAMES)
                            / (int)captureRight.get(cv::CAP_PROP_FPS)));

            connect(qtimerRight, &QTimer::timeout, this, [this]{processFrameAndUpdateGUI(true);});
            qtimerRight->start();
        }
    } else {
        captureLeft.open(stream, cv::CAP_GSTREAMER);
        if(captureLeft.isOpened() == false) {
            std::cout << "error: Left side capture is not accessed successfully";
            return;
        } else {

            ui->lblTimerLeft->setText(
                        getFormattedTime(
                            (int)captureLeft.get(cv::CAP_PROP_POS_FRAMES)
                            / (int)captureLeft.get(cv::CAP_PROP_FPS)));

            connect(qtimerLeft, &QTimer::timeout, this, [this]{processFrameAndUpdateGUI(false);});
            qtimerLeft->start();
        }
    }
}

void MainWindow::on_btnPause_clicked(bool side) {
    if(side) {
        if(qtimerRight->isActive()) {
            qtimerRight->stop();
            ui->btnPauseRight->setText("Resume");
        } else {
            qtimerRight->start();
            ui->btnPauseRight->setText("Pause");
        }
    } else {
        if(qtimerLeft->isActive()) {
            qtimerLeft->stop();
            ui->btnPauseLeft->setText("Resume");
        } else {
            qtimerLeft->start();
            ui->btnPauseLeft->setText("Pause");
        }

    }
}
