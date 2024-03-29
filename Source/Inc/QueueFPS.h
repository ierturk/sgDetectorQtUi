#ifndef QUEUEFPS_H
#define QUEUEFPS_H

#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>

template <typename T>
class QueueFPS : public std::queue<T> {
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry) {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1) {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get() {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS() {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    cv::TickMeter tm;
    std::mutex mutex;
};


#endif // QUEUEFPS_H
