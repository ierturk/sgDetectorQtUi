#include "mainwindow.h"
#include "ui_mainwindow.h"

// Ort::Env ortEnv = Ort::Env(nullptr);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

    ui->setupUi(this);
    captureTimerRight = new QTimer(this);
    captureTimerLeft = new QTimer(this);
    processTimerRight = new QTimer(this);
    processTimerLeft = new QTimer(this);
    viewerTimer = new QTimer(this);

    connect(ui->btnLoadRight, &QAbstractButton::clicked, this, [this]{on_btnLoad_clicked(true);});
    connect(ui->btnLoadLeft, &QAbstractButton::clicked, this, [this]{on_btnLoad_clicked(false);});
    connect(ui->btnPlayRight, &QAbstractButton::clicked, this, [this]{on_btnPlay_clicked(true);});
    connect(ui->btnPlayLeft, &QAbstractButton::clicked, this, [this]{on_btnPlay_clicked(false);});

    connect(viewerTimer, &QTimer::timeout, this, [this]{updateGUI();});
    viewerTimer->start(30);


    ortNet = new OrtNet();
    ortNet->Init("/home/ierturk/Work/REPOs/ssd/ssdIE/outputs/mobilenet_v2_ssd320_clk_trainval2019/model_040000.onnx");

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


void MainWindow::captureFrame(bool side) {
    cv::Mat frame;
    if(side) {
        captureRight.read(frame);
        if(frame.empty()) return;
        // cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        capturedFrameQueueRight.push(frame.clone());
    } else {
        captureLeft.read(frame);
        if(frame.empty()) return;
        // cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        capturedFrameQueueLeft.push(frame.clone());
    }
}

void MainWindow::processFrame(bool side) {
    if(side) {
        cv::Mat frame;
        if (!capturedFrameQueueRight.empty())
        {
            frame = capturedFrameQueueRight.get();
            capturedFrameQueueRight.clear();
        }

        if (!frame.empty())
        {
            ortNet->setInputTensor(frame, side);
            ortNet->forward(side);
            resultQueueRight.push(ortNet->getProcessedFrame(side));
        }

    } else {
        cv::Mat frame;
        if (!capturedFrameQueueLeft.empty())
        {
            frame = capturedFrameQueueLeft.get();
            capturedFrameQueueLeft.clear();
        }

        if (!frame.empty())
        {
            ortNet->setInputTensor(frame, side);
            ortNet->forward(side);
            resultQueueLeft.push(ortNet->getProcessedFrame(side));
        }
    }
}


void MainWindow::updateGUI() {

    if(!resultQueueRight.empty()) {
        ui->lblTimerRight->setText(
                    getFormattedTime(
                        (int)captureRight.get(cv::CAP_PROP_POS_FRAMES)
                        / (int)captureRight.get(cv::CAP_PROP_FPS)));

        QImage image = resultQueueRight.get();
        QPainter qPainter(&image);
        QFontInfo font = qPainter.fontInfo();
        qPainter.setFont(QFont(font.family(), 32));
        QPen penHText(QColor("#00e0fc"));
        qPainter.setPen(penHText);
        qPainter.drawText(10, 620, QString("Camera: %1 FPS").arg(capturedFrameQueueRight.getFPS()));
        qPainter.drawText(10, 660, QString("Network: %1 FPS").arg(resultQueueRight.getFPS()));
        qPainter.drawText(10, 700,QString("Skipped frames: %1 FPS").arg(capturedFrameQueueRight.counter - resultQueueRight.counter));

        ui->lblPlayerRight->setPixmap(QPixmap::fromImage(image));
        ui->lblPlayerRight->setScaledContents(true);
        ui->lblPlayerRight->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );

    }

    if(!resultQueueLeft.empty()) {
        ui->lblTimerLeft->setText(
                    getFormattedTime(
                        (int)captureLeft.get(cv::CAP_PROP_POS_FRAMES)
                        / (int)captureLeft.get(cv::CAP_PROP_FPS)));

        QImage image = resultQueueLeft.get();
        QPainter qPainter(&image);
        QFontInfo font = qPainter.fontInfo();
        qPainter.setFont(QFont(font.family(), 32));
        QPen penHText(QColor("#00e0fc"));
        qPainter.setPen(penHText);
        qPainter.drawText(10, 620, QString("Camera: %1 FPS").arg(capturedFrameQueueLeft.getFPS()));
        qPainter.drawText(10, 660, QString("Network: %1 FPS").arg(resultQueueLeft.getFPS()));
        qPainter.drawText(10, 700,QString("Skipped frames: %1 FPS").arg(capturedFrameQueueLeft.counter - resultQueueLeft.counter));

        ui->lblPlayerLeft->setPixmap(QPixmap::fromImage(image));
        ui->lblPlayerLeft->setScaledContents(true);
        ui->lblPlayerLeft->setSizePolicy( QSizePolicy::Ignored, QSizePolicy::Ignored );

    }
}

void MainWindow::on_btnLoad_clicked(bool side)
{
/*
    bool ok;
    QString videoURL = QInputDialog::getText(this, tr("Please provide a valid vide url"),
                                         tr("Video URL"), QLineEdit::Normal,
                                         "http://", &ok);

    if (ok && !videoURL.isEmpty())
        std::cout << videoURL.toStdString() << std::endl;
    else
        std::cout << "Please provide a valid vide url!" << std::endl;
*/

    if(side) {
        std::string stream = "http://localhost:8080/b8d7ea94eb5b4cc9f6c9df963cfae5be/mp4/pnOOUNJfOo/RightCam/s.mp4";
        captureRight.open(stream);
        if(captureRight.isOpened() == false) {
            std::cout << "error: Right side capture is not accessed successfully";
            return;
        } else {
            ui->lblTimerRight->setText(
                        getFormattedTime(
                            (int)captureRight.get(cv::CAP_PROP_POS_FRAMES)
                            / (int)captureRight.get(cv::CAP_PROP_FPS)));

            connect(captureTimerRight, &QTimer::timeout, this, [this]{captureFrame(true);});
            captureTimerRight->start(30);
            connect(processTimerRight, &QTimer::timeout, this, [this]{processFrame(true);});
        }
    } else {
        std::string stream = "http://localhost:8080/b8d7ea94eb5b4cc9f6c9df963cfae5be/mp4/pnOOUNJfOo/LeftCam/s.mp4";
        captureLeft.open(stream);
        if(captureLeft.isOpened() == false) {
            std::cout << "error: Left side capture is not accessed successfully";
            return;
        } else {
            ui->lblTimerLeft->setText(
                        getFormattedTime(
                            (int)captureLeft.get(cv::CAP_PROP_POS_FRAMES)
                            / (int)captureLeft.get(cv::CAP_PROP_FPS)));

            connect(captureTimerLeft, &QTimer::timeout, this, [this]{captureFrame(false);});
            captureTimerLeft->start(30);
            connect(processTimerLeft, &QTimer::timeout, this, [this]{processFrame(false);});
        }
    }
}

void MainWindow::on_btnPlay_clicked(bool side) {
    if(side) {
        if(processTimerRight->isActive()) {
            processTimerRight->stop();
            ui->btnPlayRight->setText("Play");
        } else {
            processTimerRight->start(30);
            ui->btnPlayRight->setText("Pause");
        }
    } else {
        if(processTimerLeft->isActive()) {
            processTimerLeft->stop();
            ui->btnPlayLeft->setText("Play");
        } else {
            processTimerLeft->start(30);
            ui->btnPlayLeft->setText("Pause");
        }
    }
}
