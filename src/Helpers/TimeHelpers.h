#ifndef TIME_HELPERS_H_
#define TIME_HELPERS_H_

#include <time.h>
#include <Windows.h>

#ifdef WINDOWS
#define cpSleep(sleepMs) Sleep(sleepMs);
#elif defined( LINUX )
#define cpSleep(sleepMs) usleep(sleepMs * 1000);
#endif

int clock_gettime(timespec &spec);
int GetTimeDiff(timespec TimeGreater, timespec TimeSmaller);
void efficientWait(int waitTime);

#endif
