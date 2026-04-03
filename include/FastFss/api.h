#ifndef FAST_FSS_API_H
#define FAST_FSS_API_H

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(FAST_FSS_BUILD)
#define FAST_FSS_API __declspec(dllexport)
#else
#define FAST_FSS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FAST_FSS_API __attribute__((visibility("default")))
#else
#define FAST_FSS_API
#endif

#endif
