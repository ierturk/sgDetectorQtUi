#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QTimer>
#include <QTime>
#include <QFileDialog>
#include <QInputDialog>
#include <QLineEdit>
#include <QPainter>
#include <QPen>

#include <OrtNet.h>
#include <QueueFPS.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    double getCurrentFrame();
    double getNumberOfFrames();
    double getFrameRate();

public slots:

private:
    void captureFrame(bool side);
    void processFrame(bool side);
    void updateGUI();

    Ui::MainWindow *ui;
    OrtNet* ortNet;

    cv::VideoCapture captureRight;
    QTimer* captureTimerRight;
    QTimer* processTimerRight;

    cv::VideoCapture captureLeft;
    QTimer* captureTimerLeft;
    QTimer* processTimerLeft;

    QTimer* viewerTimer;

    void on_btnLoad_clicked(bool side);
    void on_btnPlay_clicked(bool side);
    QString getFormattedTime(int timeInSeconds);

    QueueFPS<cv::Mat> capturedFrameQueueRight;
    QueueFPS<QImage> resultQueueRight;
    QueueFPS<cv::Mat> capturedFrameQueueLeft;
    QueueFPS<QImage> resultQueueLeft;

};
#endif // MAINWINDOW_H
