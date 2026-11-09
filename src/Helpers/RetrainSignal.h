#ifndef RETRAIN_SIGNAL_H_
#define RETRAIN_SIGNAL_H_

#include <atomic>

// in main.cpp.
extern std::atomic<bool> g_retrainRequested;

constexpr int RETRAIN_EXIT_CODE = 42;

#endif /* RETRAIN_SIGNAL_H_ */
