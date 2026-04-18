/* docs/examples/quickstart_common.h
 *
 * Shared allocator wrapper for the quickstart examples. Every example in
 * docs/examples/ routes its allocations through qs_realloc so the code
 * visibly follows tdc's (user, ptr, n) allocator convention:
 *
 *     qs_realloc(user, NULL, n)   -> allocate n bytes
 *     qs_realloc(user, p, 0)      -> free p
 *     qs_realloc(user, p, n)      -> grow p to n bytes (may move)
 */

#ifndef QS_COMMON_H
#define QS_COMMON_H

#include "tdc.h"
#include <stdio.h>
#include <stdlib.h>

static void *qs_realloc(void *user, void *ptr, size_t n) {
    (void)user;
    if (n == 0) { free(ptr); return NULL; }
    return realloc(ptr, n);
}

static inline tdc_buffer qs_buffer(void) {
    tdc_buffer b = {0};
    b.realloc_fn = qs_realloc;
    return b;
}

static inline void qs_buffer_free(tdc_buffer *b) {
    qs_realloc(NULL, b->data, 0);
    b->data = NULL;
    b->size = 0;
    b->capacity = 0;
}

static inline int qs_check(tdc_status s, const char *label) {
    if (s == TDC_OK) return 0;
    fprintf(stderr, "%s failed: %s (%d)\n", label, tdc_strerror(s), (int)s);
    return 1;
}

#endif
