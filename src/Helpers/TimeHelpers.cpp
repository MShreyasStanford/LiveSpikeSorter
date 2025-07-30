#include <stdint.h>
#include "TimeHelpers.h"

#define _MAX_SPIN_COUNT 1'000'000

int clock_gettime(timespec &spec)		//C-file part
{
	__int64 wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
	wintime -= 116444736000000000i64;				//1jan1601 to 1jan1970
	spec.tv_sec = wintime / 10000000i64;			//seconds
	spec.tv_nsec = wintime % 10000000i64 * 100;	//nano-seconds
	return 0;
}


//Returns time difference in milliseconds
int GetTimeDiff(timespec TimeGreater, timespec TimeSmaller) {
	int secs  = TimeGreater.tv_sec  - TimeSmaller.tv_sec;
	int nsecs = TimeGreater.tv_nsec - TimeSmaller.tv_nsec;

	return secs * 1000 + nsecs / 1'000'000;
};

/* Windows' Sleep() is too variable (i.e. a Sleep(5) can result in a 5-15ms sleep).
Spin waiting is not ideal because it unnecessarily burns CPU, so this function offers 
a compromise. If higher resolution than 1 milliseconds is necessary, could change to
microseconds */
void efficientWait(int waitTime) {
	timespec initTime, currTime; // Timespec variables
	clock_gettime(initTime); // Get the current time
	clock_gettime(currTime);
	uint64_t prev = __rdtsc(); // Get current cpu cycle count
	while(GetTimeDiff(currTime, initTime) < waitTime){
		clock_gettime(currTime); // Update current time
		do {
			_mm_pause(); // tells processor that the calling thread is in a busy wait loop
		} while (__rdtsc() - prev < _MAX_SPIN_COUNT);
	}
}
