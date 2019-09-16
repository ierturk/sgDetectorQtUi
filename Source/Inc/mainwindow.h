#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QTimer>
#include <QTime>
#include <QFileDialog>

// #include<iostream>
// #include<fstream>
// #include<string.h>

#include <OrtNet.h>


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
    void processFrameAndUpdateGUI();

private slots:
    void on_loadButton_clicked();
    void on_playButton_clicked();
    void on_pushButton_3_clicked();
    QString getFormattedTime(int timeInSeconds);
    void on_horizontalSlider_sliderPressed();
    void on_horizontalSlider_sliderReleased();
    void on_horizontalSlider_sliderMoved(int position);
    void setCurrentFrame( int frameNumber);
    void on_actionOpen_triggered();
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat matOriginal;
    cv::VideoCapture capWebcam;
    QImage qimgOriginal;
    QTimer* qtimer;
    OrtNet* ortNet;
};
#endif // MAINWINDOW_H
