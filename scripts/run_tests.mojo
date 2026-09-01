from std.python import Python

def main() raises:
    var glob = Python.import_module("glob")
    var os = Python.import_module("os")
    var pty = Python.import_module("pty")
    var subprocess = Python.import_module("subprocess")
    var sys = Python.import_module("sys")
    var re = Python.import_module("re")
    var time = Python.import_module("time")
    var py_str = Python.import_module("builtins").str
    var py_builtins = Python.import_module("builtins")

    var test_files_obj = glob.glob("tests/test_*.mojo")
    var test_files = py_builtins.sorted(test_files_obj)

    var total_files = Int(String(len(test_files)))
    if total_files == 0:
        print("No test files found in tests/")
        return

    var conda_prefix = os.environ.get("CONDA_PREFIX", "")
    var linker_flags = Python.list()
    if conda_prefix:
        linker_flags.append("-Xlinker")
        linker_flags.append("-L" + String(conda_prefix) + "/lib")
        linker_flags.append("-Xlinker")
        linker_flags.append("-llapack")
        linker_flags.append("-Xlinker")
        linker_flags.append("-lblas")

    var cpu_cnt = Int(String(os.cpu_count()))
    var sys_argv = sys.argv
    var is_single_thread = False
    var custom_jobs = 0

    var argv_len = Int(String(len(sys_argv)))
    for idx in range(argv_len):
        var arg_str = String(sys_argv[idx])
        if (
            arg_str == "--single-threaded"
            or arg_str == "-j1"
            or arg_str == "--sequential"
        ):
            is_single_thread = True
        elif arg_str == "-j" or arg_str == "--jobs":
            if idx + 1 < argv_len:
                try:
                    custom_jobs = Int(String(sys_argv[idx + 1]))
                except:
                    pass

    var max_workers = 1 if is_single_thread else (
        custom_jobs if custom_jobs > 0 else (cpu_cnt if cpu_cnt <= 8 else 8)
    )

    var start_time = time.time()
    var summary_pattern = re.compile(
        r"Summary\s+\[\s*[\d.]+\s*\]\s+(\d+)\s+tests"
        r" run:\s+(\d+)\s+passed\s*,\s*(\d+)\s+failed\s*,\s*(\d+)\s+skipped"
    )

    var active_jobs = Python.list()
    var file_idx = 0
    var total_tests = 0
    var total_passed = 0
    var total_failed = 0
    var total_skipped = 0
    var failed_files = Python.list()

    while file_idx < total_files or len(active_jobs) > 0:
        while file_idx < total_files and len(active_jobs) < max_workers:
            var test_file = test_files[file_idx]
            file_idx += 1

            var cmd = Python.list()
            cmd.append("mojo")
            cmd.append("run")
            cmd.append("-I")
            cmd.append(".")
            cmd.extend(linker_flags)
            cmd.append(test_file)

            var pty_pair = pty.openpty()
            var master_fd = pty_pair[0]
            var s_fd = pty_pair[1]

            var proc = subprocess.Popen(
                cmd, stdout=s_fd, stderr=s_fd, close_fds=True
            )
            os.close(s_fd)
            var master_file = os.fdopen(master_fd, "r")

            var job = Python.dict()
            job["file"] = test_file
            job["proc"] = proc
            job["file_obj"] = master_file
            job["file_tests"] = 0
            job["file_passed"] = 0
            job["file_failed"] = 0
            job["file_skipped"] = 0
            active_jobs.append(job)

        var remaining_jobs = Python.list()
        for j in range(len(active_jobs)):
            var job = active_jobs[j]
            var master_file = job["file_obj"]
            var proc = job["proc"]
            var test_file = job["file"]

            while True:
                var line_str: String
                try:
                    var line_obj = master_file.readline()
                    if not line_obj:
                        break
                    line_str = String(line_obj)
                except:
                    break

                sys.stdout.write(line_str)
                sys.stdout.flush()

                var clean_line = String(
                    re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line_str)
                )
                var res = summary_pattern.search(clean_line)
                if res:
                    job["file_tests"] = Int(String(res.group(1)))
                    job["file_passed"] = Int(String(res.group(2)))
                    job["file_failed"] = Int(String(res.group(3)))
                    job["file_skipped"] = Int(String(res.group(4)))

            if proc.poll() is not None:
                try:
                    master_file.close()
                except:
                    pass

                var returncode = Int(String(proc.returncode))
                var f_tests = Int(String(job["file_tests"]))
                var f_passed = Int(String(job["file_passed"]))
                var f_failed = Int(String(job["file_failed"]))
                var f_skipped = Int(String(job["file_skipped"]))

                total_tests += f_tests
                total_passed += f_passed
                total_failed += f_failed
                total_skipped += f_skipped

                if returncode != 0 or f_failed > 0:
                    failed_files.append(test_file)
            else:
                remaining_jobs.append(job)

        active_jobs = remaining_jobs
        if len(active_jobs) > 0 and file_idx >= total_files:
            time.sleep(0.01)

    var elapsed_val = Float64(String(time.time() - start_time))
    var failed_files_count = Int(String(len(failed_files)))
    var passed_files = total_files - failed_files_count
    var is_success = failed_files_count == 0 and total_failed == 0

    var RESET = "\033[0m"
    var BOLD = "\033[1m"
    var FG_GREEN = "\033[38;5;48m"
    var FG_RED = "\033[38;5;203m"
    var FG_YELLOW = "\033[38;5;221m"
    var FG_CYAN = "\033[38;5;75m"
    var FG_WHITE = "\033[38;5;255m"
    var FG_MUTED = "\033[38;5;242m"
    var BG_GREEN = "\033[48;5;40;38;5;232;1m"
    var BG_RED = "\033[48;5;196;38;15;1m"

    var hr = String(py_str("─") * 48)

    print("\n" + FG_MUTED + hr + RESET)
    if is_success:
        print(
            BG_GREEN
            + " [PASS] "
            + RESET
            + " "
            + FG_GREEN
            + BOLD
            + "All test suites passed successfully"
            + RESET
        )
    else:
        print(
            BG_RED
            + " [FAIL] "
            + RESET
            + " "
            + FG_RED
            + BOLD
            + "Test suite execution failed"
            + RESET
        )
    print()

    var bar_width = 24
    var pass_ratio = 1.0
    if total_tests > 0:
        pass_ratio = Float64(total_passed) / Float64(total_tests)
    var filled = Int(pass_ratio * Float64(bar_width))
    var empty = bar_width - filled

    var filled_bar = String(py_str("━") * filled)
    var empty_bar = String(py_str("━") * empty)
    var pct_str = String(Int(pass_ratio * 100))

    print(
        "  "
        + FG_MUTED
        + "Progress:  "
        + RESET
        + FG_GREEN
        + filled_bar
        + FG_RED
        + empty_bar
        + RESET
        + " "
        + FG_MUTED
        + pct_str
        + "%"
        + RESET
    )

    print("  " + FG_WHITE + BOLD + "Suites:    " + RESET, end="")
    if passed_files > 0:
        print(
            FG_GREEN + BOLD + String(passed_files) + " passed" + RESET, end=""
        )
        if failed_files_count > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if failed_files_count > 0:
        print(
            FG_RED + BOLD + String(failed_files_count) + " failed" + RESET,
            end="",
        )
    print(FG_MUTED + " (" + String(total_files) + " total)" + RESET)

    print("  " + FG_WHITE + BOLD + "Tests:     " + RESET, end="")
    if total_passed > 0:
        print(
            FG_GREEN + BOLD + String(total_passed) + " passed" + RESET, end=""
        )
        if total_failed > 0 or total_skipped > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if total_failed > 0:
        print(FG_RED + BOLD + String(total_failed) + " failed" + RESET, end="")
        if total_skipped > 0:
            print(FG_MUTED + ", " + RESET, end="")
    if total_skipped > 0:
        print(
            FG_YELLOW + BOLD + String(total_skipped) + " skipped" + RESET,
            end="",
        )
    print(FG_MUTED + " (" + String(total_tests) + " total)" + RESET)

    var time_str = String(py_str("{:.2f}").format(elapsed_val))
    print(
        "  "
        + FG_WHITE
        + BOLD
        + "Duration:  "
        + RESET
        + FG_CYAN
        + time_str
        + "s"
        + RESET
    )
    print(FG_MUTED + hr + RESET)

    if not is_success:
        print("\n" + FG_RED + BOLD + "FAILED SUITES" + RESET)
        for i in range(failed_files_count):
            print(
                "  "
                + FG_RED
                + "│"
                + RESET
                + "  "
                + FG_RED
                + "✖"
                + RESET
                + " "
                + FG_WHITE
                + String(failed_files[i])
                + RESET
            )
        print(FG_RED + "└" + String(py_str("─") * 44) + RESET + "\n")
        sys.exit(1)
