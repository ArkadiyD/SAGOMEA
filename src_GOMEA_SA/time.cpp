#include "time.hpp"

long long getCurrentTimeStampInMilliSeconds()
{
  struct timeval tv;
  long long result;

  gettimeofday (&tv, NULL);
  result = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);

  return  result;
}

long long getMilliSecondsRunningSinceTimeStamp (long long startTimestamp)
{
  long long timestamp_now, difference;

  timestamp_now = getCurrentTimeStampInMilliSeconds();

  difference = timestamp_now-startTimestamp;

  return difference;
}
