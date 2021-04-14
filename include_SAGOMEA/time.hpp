#pragma once

#include <sys/time.h>
#include <cstddef>

long long getCurrentTimeStampInMilliSeconds();
long long getMilliSecondsRunningSinceTimeStamp (long long timestampStart);
