#ifndef LOGGER_H
#define LOGGER_H

#include <cstdio>
#include <iostream>
#include <fstream>
#include <chrono>

#ifdef CDFCI_LOG
#define LOG_FILE std::remove("log.txt");

#define LOG_NUMBER(message, number) \
    do { \
        std::ofstream logfile("log.txt", std::ios_base::app); \
        if (!logfile.is_open()) { \
            std::cerr << "Failed to open log file!" << std::endl; \
            break; \
        } \
        logfile << "[" << __FILE__ << ":" << __LINE__ << "] " << message << ": " << number << std::endl; \
        logfile.close(); \
    } while (false)

#define LOG_VECTOR(message, a, n) \
    do { \
        std::ofstream logfile("log.txt", std::ios_base::app); \
        if (!logfile.is_open()) { \
            std::cerr << "Failed to open log file!" << std::endl; \
            break; \
        } \
        logfile << "[" << __FILE__ << ":" << __LINE__ << "] " << message << std::endl; \
        logfile << "Vector dimension " << (n) << std::endl; \
        for (int i = 0; i < (n); i++) { \
            logfile << a[i] << std::endl; \
        } \
        logfile.close(); \
    } while (false)

#define LOG_MATRIX(message, A, n) \
    do { \
        std::ofstream logfile("log.txt", std::ios_base::app); \
        if (!logfile.is_open()) { \
            std::cerr << "Failed to open log file!" << std::endl; \
            break; \
        } \
        logfile << "[" << __FILE__ << ":" << __LINE__ << "] " << message << std::endl; \
        logfile << "Matrix dimension " << (n) << std::endl; \
        for (int i = 0; i < (n); i++) { \
            for (int j = 0; j < (n); j++) logfile << A[i*n + j] << " "; \
            logfile << std::endl; \
        } \
        logfile.close(); \
    } while (false)

#define LOG_WAVEFUNCTIONVECTOR(message, wfv) \
    do { \
        std::ofstream logfile("log.txt", std::ios_base::app); \
        if (!logfile.is_open()) { \
            std::cerr << "Failed to open log file!" << std::endl; \
            break; \
        } \
        logfile << "[" << __FILE__ << ":" << __LINE__ << "] " << message << std::endl; \
        logfile << "Vector dimension " << (wfv).size() << std::endl; \
        for (int i = 0; i < (wfv).size(); i++) { \
            logfile << "wfv[" << i << "]: " << wfv[i].first << "(" << wfv[i].second[0] << ", " << wfv[i].second[1] << ")" << std::endl; \
        } \
        logfile.close(); \
    } while (false)

#else
#define LOG_FILE
#define LOG_NUMBER(message, number)
#define LOG_VECTOR(message, a, n)
#define LOG_MATRIX(message, A, n)
#define LOG_WAVEFUNCTIONVECTOR(message, wfv)
#endif

struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> time_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> time_end;
    std::vector<double> data;

    Timer() {
        data = std::vector<double>(1, 0.0);
    }

    Timer(int n) {
        data = std::vector<double>(n, 0.0);
    }

    void start() {
        time_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        time_end = std::chrono::high_resolution_clock::now();
    }

    void save() {
        std::chrono::duration<double> time_elpse = time_end - time_start;
        data[0] += time_elpse.count();
    }

    void save(int i) {
        if (i >= data.size()) {
            printf("Error: invalid timer number (%d >= %d).\n", i, (int) data.size());
            return;
        }
        std::chrono::duration<double> time_elpse = time_end - time_start;
        data[i] += time_elpse.count();
    }

    void lap() {
        time_end = std::chrono::high_resolution_clock::now();
        save();
        time_start = std::chrono::high_resolution_clock::now();
    }

    void lap(int i) {
        time_end = std::chrono::high_resolution_clock::now();
        save(i);
        time_start = std::chrono::high_resolution_clock::now();
    }

    void reset() {
        data.clear();
    }

    void log(std::string timer_name) {
        std::cout << "[LOG] " << timer_name << ": ";
        for (int i = 0; i < data.size(); i++)
            printf("%.2f ", data[i]);
        std::cout << std::endl;
    }
};

#endif