// SPDX-License-Identifier: MIT
// Plugin-only phase after Turnip: the Metal bridge self-test (bridge/NativeMetalBridge.rs) on the same Turnip, on an 8 MiB worker under a
// watchdog. Markers 0x6501..0x650f come from the bridge itself; the result line goes to GPU7 ("M ..."), a stuck worker is reported there
// as "MW <last marker> <seconds>".
// After a passing self-test the worker runs the frontend checks (bridge/FrontendChecks.h) through the public Metal API on the native frontend
// (NativeBridgeMetalFrontend.mm) over the same Turnip: markers 0x6600 (device), 0x6601..0x660f, then 0x66f0 pass / 0x66f1 fail. Its verdict
// ends the same GPU7 line: "F0" (with the first unimplemented selector, if any), "F<failed> <marker> <first failure>", "Fo"/"Fx" + reason
// when the device or the fixtures were refused.
#pragma once
#include "bridge/NativeMetalBridge.h"
#include "bridge/NativeMetalObjects.h"
#include "bridge/FrontendChecks.h"
#include "NativeBridgeMetalFrontend.h"
#include "GpuScanoutClient.h"
#include <fcntl.h>
#include <pthread.h>
#include <pwd.h>        /* getpwuid: the shader-cache ladder's first candidate is the user's Library/Caches */
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace bridgephase {
struct Job {
    const char *turnip;
    char cache[512];
    NmbSelftest result;
    uint32_t code;
    frontendchecks::Report frontend;
    bool frontendRan, frontendOpened;
    char missing[64], error[48];
    volatile uint32_t last;
    volatile bool done;
};
static Gpu *gGpu;
static Job *gJob;
static id gDevice;   // the frontend device of the checks, kept for the scanout stage
static void mark(uint32_t value) {
    if (gJob) gJob->last = value;
    if (gGpu) gGpu->mark(value);
}
static void missingSelector(const char *cls, const char *selector) {
    if (gJob && !gJob->missing[0]) snprintf(gJob->missing, sizeof gJob->missing, "FM %.20s %.36s", cls, selector);
}
// The frontend checks on the native frontend over Turnip (QuartzCore is already translated by the self-test, so it is not repeated).
static void frontend(Job *job) {
    mark(0x6600);
    id device = native_bridge_metal_open(job->turnip, job->cache, 0, missingSelector);
    // The runtime's reasons are per thread: keep this worker's for the report.
    if (!device) {strlcpy(job->error, nmb_last_error(), sizeof job->error); return;}
    job->frontendOpened = true;
    gDevice = device;
    size_t bytes = 0;
    const void *air = nmb_fixture_air(&bytes);
    id library = native_bridge_metal_library_with_air(air, bytes);
    air = nmb_function_constant_air(&bytes);
    id fcLibrary = native_bridge_metal_library_with_air(air, bytes);
    if (!library || !fcLibrary) {strlcpy(job->error, nmb_last_error(), sizeof job->error); return;}
    job->frontend.mark = mark;
    job->frontendRan = true;
    // An exception escaping a worker thread aborts the whole host process (SpringBoard died that way, int-075/083): report it as a failed check.
    @try {
        frontendchecks::run(device, library, fcLibrary, nil, &job->frontend);
    } @catch (NSException *exception) {
        ++job->frontend.failed;
        snprintf(job->frontend.first, sizeof job->frontend.first, "exception %.40s", exception.name.UTF8String ?: "?");
    }
}
static void *worker(void *arg) {
    Job *job = static_cast<Job *>(arg);
    // The target's own compositor library, read in place (never copied anywhere).
    const uint8_t *qc = nullptr;
    size_t qcBytes = 0;
    const int fd = open("/System/Library/Frameworks/QuartzCore.framework/default.metallib", O_RDONLY);
    struct stat st = {};
    if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
        void *m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m != MAP_FAILED) {qc = static_cast<const uint8_t *>(m); qcBytes = (size_t)st.st_size;}
    }
    if (fd >= 0) close(fd);
    job->code = nmb_selftest(job->turnip, job->cache, qc, qcBytes, mark, &job->result);
    if (qc) munmap(const_cast<uint8_t *>(qc), qcBytes);
    if (!job->code) frontend(job);
    __atomic_store_n(&job->done, true, __ATOMIC_RELEASE);
    return nullptr;
}
// Serves the display driver's frames with the GPU on `device` (the kernel offers them only with sheng-fb-gpu=1). The calls go through a hand-off-only
// connection to the accelerator (open type 0x4148: no GPU slot is taken): the original sandbox profiles admit that class and not the presenter's own
// service. Reports on GPU7 only when the state changes: "P<step> e<hex>" (0 no connection, 1 no selector, 2 mode refused, 3 scanout not
// mappable/importable, 4 serving), then "PF <frames>/<failures> r<last result> v<verified>/<differing rows> t<ms>" at the first frame, every fifth after
// and the first failure, "PX ..." when serving stops. Returns when serving stops.
struct ScanoutContext {
    Gpu link;
    Gpu *marker;
};
static void serveScanout(Gpu &gpu, id device) {
    // This thread carries every frame to the screen; at the default priority it waited behind the launching apps' threads.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    ScanoutContext context;
    context.marker = &gpu;
    Gpu &link = context.link;
    link.call = gpu.call; link.map = gpu.map; link.unmap = gpu.unmap; link.openFn = gpu.openFn; link.closeFn = gpu.closeFn; link.port = gpu.port;
    gpuscanout::Stats st = {};
    char line[64];
    gpu.mark(0x6700);
    const int openResult = link.openFn ? link.openFn(link.port, mach_task_self(), 0x4148, &link.conn) : -1;
    if (openResult != 0 || !link.conn) {
        snprintf(line, sizeof line, "P0 e%x", (unsigned)openResult);
    } else if (gpu.mark(0x6701), !link.probe()) {
        snprintf(line, sizeof line, "P1");
    } else {
        gpu.mark(0x6702);
        gpuscanout::Transport transport = {&context,
            [](void *c, uint32_t method, const void *in, size_t is, void *out, size_t os) -> int {
                return static_cast<ScanoutContext *>(c)->link.method(0x7000 + method, in, is, out, os);
            },
            [](void *c, uint64_t *address, uint64_t *length) -> int {
                Gpu &g = static_cast<ScanoutContext *>(c)->link;
                return g.map(g.conn, 0x5343, mach_task_self(), address, length, 0x101);   // anywhere + inhibit cache
            },
            [](void *c, uint32_t value) {static_cast<ScanoutContext *>(c)->marker->mark(value);}};
        gpuscanout::Client client;
        const bool opened = client.open((id<MTLDevice>)device, transport, &st);
        snprintf(line, sizeof line, "P%d e%x", st.step, (unsigned)st.error);
        streamText7(gpu, line);
        if (opened) {
            uint64_t reportedFrames = 0, reportedFailures = 0;
            while (client.serveOne(500)) {
                const bool frame = st.frames && (st.frames == 1 || st.frames - reportedFrames >= 5);
                if (frame || (st.failures && !reportedFailures)) {
                    reportedFrames = st.frames ? st.frames : reportedFrames;
                    reportedFailures = st.failures ? st.failures : reportedFailures;
                    const uint64_t served = st.frames + st.failures;
                    snprintf(line, sizeof line, "PF %llu/%llu r%d v%llu/%llu t%u a%llu", (unsigned long long)st.frames, (unsigned long long)st.failures,
                             st.lastResult, (unsigned long long)st.verified, (unsigned long long)st.mismatchRows, st.lastMs,
                             (unsigned long long)(served ? st.totalMs / served : 0));
                    streamText7(gpu, line);
                }
            }
            snprintf(line, sizeof line, "PX %llu/%llu r%d e%x", (unsigned long long)st.frames, (unsigned long long)st.failures, st.lastResult,
                     (unsigned)st.error);
        }
    }
    streamText7(gpu, line);
}
static void run(Gpu &gpu, const char *turnipPath) {
    gpu.mark(0x6500);
    Job *job = new Job{};
    job->turnip = turnipPath;
    // The translated-shader cache root must be writable by the process this runs in. The plugin's own device path already
    // ladders these five candidates (NativeAdrenoMetalPlugin.mm:271-303) because the per-user cache directory is not always
    // usable under a system process's sandbox; this path had one hard-coded choice and discarded mkdir's result, and
    // metal2vulkan refuses to open the device when its shader-cache Directory::open cannot mkdir or probe-write it. That
    // refusal is the whole of the "M s6501 ... ShaderCache" row: 0x6501 is this step (bridge/NativeMetalBridge.rs:306-310),
    // not a GPU fault, and it is independent of the engine-write work. Same candidates and order as the plugin; the plugin
    // additionally retries the device open once per directory, which this single-shot path cannot, so it takes the first
    // directory that is writable.
    {
        char candidates[5][512] = {};
        if (!confstr(_CS_DARWIN_USER_CACHE_DIR, candidates[0], sizeof candidates[0] - 16)) candidates[0][0] = 0;
        if (!confstr(_CS_DARWIN_USER_TEMP_DIR, candidates[1], sizeof candidates[1] - 16)) candidates[1][0] = 0;
        strlcpy(candidates[2], "/tmp/", sizeof candidates[2]);
        strlcpy(candidates[3], "/private/var/tmp/", sizeof candidates[3]);
        if (const passwd *user = getpwuid(getuid()); user && user->pw_dir && user->pw_dir[0]) {
            snprintf(candidates[4], sizeof candidates[4] - 16, "%s/Library/Caches/com.apple.backboardd", user->pw_dir);
            mkdir(candidates[4], 0700);
            strlcat(candidates[4], "/", sizeof candidates[4]);
        }
        static const unsigned order[] = {4, 0, 1, 2, 3};
        job->cache[0] = 0;
        for (unsigned i : order) {
            if (!candidates[i][0]) continue;
            strlcat(candidates[i], "m2v", sizeof candidates[i]);
            mkdir(candidates[i], 0700);
            if (access(candidates[i], W_OK) != 0) continue;
            strlcpy(job->cache, candidates[i], sizeof job->cache);
            break;
        }
        if (!job->cache[0]) strlcpy(job->cache, "/tmp/m2v", sizeof job->cache);
    }
    gGpu = &gpu;
    gJob = job;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8u << 20);
    pthread_t thread;
    const bool started = pthread_create(&thread, &attr, worker, job) == 0;
    pthread_attr_destroy(&attr);
    if (!started) {gpu.mark(0x65fd); return;}
    static const unsigned checkpoints[] = {30, 90, 180};
    unsigned next = 0;
    for (unsigned tenths = 0; !__atomic_load_n(&job->done, __ATOMIC_ACQUIRE); ++tenths) {
        usleep(100000);
        if (next < 3 && tenths == checkpoints[next] * 10) {
            char text[64];
            snprintf(text, sizeof text, "MW %x %us", job->last, checkpoints[next]);
            streamText7(gpu, text);
            ++next;
        }
    }
    pthread_join(thread, nullptr);
    const NmbSelftest &r = job->result;
    // One GPU7 line (the page keeps only the latest): the self-test's counts, then after a passing self-test the frontend's verdict.
    char text[64];
    const int at = snprintf(text, sizeof text, "M s%x c%u r%u i%u q%u k%u/%u p%u/%u", r.stage, r.compute_bad, r.render_bad, r.import_bad,
                            r.qc_functions, r.qc_kernels_ok, r.qc_kernels_failed, r.qc_pairs_ok, r.qc_pairs_failed);
    char *rest = text + (at > 0 && at < (int)sizeof text ? at : 0);
    const size_t room = sizeof text - (size_t)(rest - text);
    // A step that fails before any count exists has an empty counts field, and the 33 characters it leaves cut "device:
    // ShaderCache(ShaderCacheError..." mid-word -- the one text that identifies the failure. Such a step spends the line on the
    // message instead; a later step keeps its counts, because a 0x6504-0x6506 failure already carries compute/render/import
    // (bridge/NativeMetalBridge.rs:318,323,328), which are the information there. A second streamText7 call cannot carry the rest:
    // it starts a new GPU7 line and the page keeps only the latest, so it would replace this one (NativeAdrenoPublish.cpp:909-926).
    const bool counted = r.compute_bad || r.render_bad || r.import_bad || r.qc_functions || r.qc_kernels_ok ||
                         r.qc_kernels_failed || r.qc_pairs_ok || r.qc_pairs_failed;
    if (job->code && !counted) snprintf(text, sizeof text, "M s%x %.54s", r.stage, r.message);
    else if (job->code) snprintf(rest, room, " %s", r.message);
    else if (!job->frontendOpened) snprintf(rest, room, " Fo %s", job->error);
    else if (!job->frontendRan) snprintf(rest, room, " Fx %s", job->error);
    else if (job->frontend.failed) snprintf(rest, room, " F%u %s", job->frontend.failed, job->frontend.first);
    else snprintf(rest, room, " F0%s%s", job->missing[0] ? " " : "", job->missing[0] ? job->missing + 3 : "");
    streamText7(gpu, text);
    const bool ok = !job->code && !r.compute_bad && !r.render_bad && !r.import_bad;
    gpu.mark(ok ? 0x65f0 : 0x65f1);
    if (!job->code) gpu.mark(job->frontendRan && !job->frontend.failed ? 0x66f0 : 0x66f1);
    const bool scanout = ok && job->frontendRan && !job->frontend.failed && gDevice;
    gJob = nullptr;
    delete job;
    // After the checks' verdict has been on the page for a while: serve the display driver's frames with the GPU.
    if (scanout) {
        sleep(8);
        serveScanout(gpu, gDevice);
    }
    gGpu = nullptr;
}
}  // namespace bridgephase
