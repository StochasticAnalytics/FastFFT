# Simple debugging and profiling

## Debug vs Releases

*cis*TEM defines several preprocessor macros that can be used to debug code, without any cost to the release build. Those defined in /src/define.h are:

```code
#ifdef DEBUG
#define MyDebugPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); StackDump dump(NULL); dump.Walk(2);}
#define MyPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);StackDump dump(NULL); dump.Walk(2);}
#define MyDebugPrint(...)	{wxPrintf(__VA_ARGS__); wxPrintf("\n");}
#define MyDebugAssertTrue(cond, msg, ...) {if ((cond) != true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define MyDebugAssertFalse(cond, msg, ...) {if ((cond) == true) { wxPrintf("\n" msg, ##__VA_ARGS__); wxPrintf("\nFailed Assert at %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__); DEBUG_ABORT;}}
#define DEBUG_ABORT {StackDump dump(NULL); dump.Walk(1); abort();}
#else
#define MyPrintWithDetails(...)	{wxPrintf(__VA_ARGS__); wxPrintf("From %s:%i\n%s\n", __FILE__,__LINE__,__PRETTY_FUNCTION__);}
#define MyDebugPrintWithDetails(...)
#define MyDebugPrint(...)
#define MyDebugAssertTrue(cond, msg, ...)
#define MyDebugAssertFalse(cond, msg, ...)
#define DEBUG_ABORT exit(-1);
#endif
```

MyPrintWithDetails(""); is especially useful for tracking down exactly where something is going wrong in the event the backtrace is not enough.

## Profiling

simple profiling in *cis*TEM is accomplished with the StopWatch class. This is too complicated for a preprocessor macro to handle appropriatly, so to handle the release/debug cases, we have introduced two namespaces:

### namespace cisTEM_timer, cisTEM_timer_noop

They can be used and mixed as follows:
1) in global declarative region, just put in "using cistem_timer". Then any calls to StopWatch::methods will be as normal
2) in global declarative region, place "using cistem_timer(_noop)" under control of a preprocessor define, such that the "noop" namespace can be invoked(or not) with a configure option.
🧨 The noop namespace defines all the methods as inline such that the compiler optimizes them out, and there is ***no function call inserted into the compiled assembly***
3) In combination with the previous (control via preprocesser define) you can also instantiate a StopWatch object by directly using the namespace

E.g.

```code
#include core_headers.h

#ifdef ENABLE_PROFILING
using namespace cistem_timer
#else
using namespace cistem_timer_noop
#endif
-----
program variables etc.
-----
cistem_timer::StopWatch always_on_timer; // invoke namespace directly, this timer always runs
Stopwatch profiler; // relies on the using directive, gated by the preprocessor define

```