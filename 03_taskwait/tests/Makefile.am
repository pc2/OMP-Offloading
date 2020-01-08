if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = taskwait_test_00.sh

TESTS = $(check_SCRIPTS)

taskwait_test_00.sh: $(top_srcdir)/src/taskwait
	echo "${cmdrun} taskwait_real_00.sh" > $@
	chmod +x $@

CLEANFILES = taskwait_test_00.sh
