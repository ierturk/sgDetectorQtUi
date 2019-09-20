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
    void processFrameAndUpdateGUI(bool side);

private:
    Ui::MainWindow *ui;
    cv::VideoCapture captureLeft;
    cv::VideoCapture captureRight;
    QTimer* qtimerLeft;
    QTimer* qtimerRight;
    // OrtNet* ortNetLeft;
    // OrtNet* ortNetRight;
    OrtNet* ortNet;

    void on_btnPlay_clicked(bool side);
    void on_btnPause_clicked(bool side);
    QString getFormattedTime(int timeInSeconds);

};
#endif // MAINWINDOW_H
