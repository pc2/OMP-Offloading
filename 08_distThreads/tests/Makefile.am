if HAVE_CCSALLOC
cmdrun = ${CCSALLOC}
else
cmdrun = `which bash`
endif

check_SCRIPTS = distThreads_test_00.sh

TESTS = $(check_SCRIPTS)

distThreads_test_00.sh: $(top_srcdir)/src/distThreads
	echo "${cmdrun} distThreads_real_00.sh" > $@
	chmod +x $@

CLEANFILES = distThreads_test_00.sh
